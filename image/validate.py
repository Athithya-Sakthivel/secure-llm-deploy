#!/usr/bin/env python3
"""Validate the pinned model lock and locally downloaded model artifacts.

The validator checks model metadata and referenced files without loading the
multi-gigabyte target/drafter weights into memory. For Prompt Guard it creates
an ONNX Runtime CPU session and runs one short tokenizer-produced smoke input.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Final

LOCK_KEYS: Final[dict[str, tuple[str, str]]] = {
    "target": ("TARGET_MODEL_ID", "TARGET_MODEL_REVISION"),
    "eagle": ("EAGLE_MODEL_ID", "EAGLE_MODEL_REVISION"),
    "guard": ("GUARD_MODEL_ID", "GUARD_MODEL_REVISION"),
}
REVISION_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-fA-F]{40}\Z")
MODEL_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
DEFAULT_LOCK_FILE = Path("/etc/vllm/models.lock")
GUARD_ONNX_NAME = "model.QUInt8.quant.onnx"


def read_lock(path: Path) -> dict[str, str]:
    """Parse and validate the exact six-key KEY=VALUE model manifest."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read model lock {path}: {exc}") from exc

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"Invalid model-lock line {line_number}: expected KEY=VALUE"
            )
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            raise ValueError(
                f"Empty key or value in model-lock line {line_number}"
            )
        if key in values:
            raise ValueError(f"Duplicate lock key: {key}")
        values[key] = value

    expected_keys = {key for pair in LOCK_KEYS.values() for key in pair}
    missing = sorted(expected_keys - values.keys())
    unknown = sorted(values.keys() - expected_keys)
    if missing:
        raise ValueError(
            "Model lock is missing required keys: " + ", ".join(missing)
        )
    if unknown:
        raise ValueError(
            "Model lock contains unknown keys: " + ", ".join(unknown)
        )

    for id_key, revision_key in LOCK_KEYS.values():
        model_id = values[id_key]
        revision = values[revision_key]
        if MODEL_ID_RE.fullmatch(model_id) is None:
            raise ValueError(
                f"{id_key} is not a valid Hugging Face repository ID: "
                f"{model_id!r}"
            )
        if REVISION_RE.fullmatch(revision) is None:
            raise ValueError(
                f"{revision_key} must be a full 40-character commit SHA"
            )
        values[revision_key] = revision.lower()
    return values


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _require_file(path: Path) -> None:
    """Require a nonempty regular file that the current process can actually read."""
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Required artifact is missing or empty: {path}")
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise ValueError(
            f"Required artifact is missing or unreadable: {path}"
        ) from exc


def _require_files(directory: Path, filenames: tuple[str, ...]) -> None:
    for filename in filenames:
        _require_file(directory / filename)


def _validate_target(model_dir: Path) -> None:
    """Validate Qwen3 architecture, AWQ metadata and all indexed weight shards."""
    _require_files(
        model_dir,
        (
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        ),
    )
    config = _read_json(model_dir / "config.json")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not any(
        isinstance(name, str)
        and name.casefold() == "qwen3forcausallm"
        for name in architectures
    ):
        raise ValueError(
            "Target config must declare Qwen3ForCausalLM; "
            f"found {architectures!r}"
        )
    if config.get("model_type") != "qwen3":
        raise ValueError(
            "Target config must declare model_type='qwen3'; "
            f"found {config.get('model_type')!r}"
        )

    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError(
            "Target config has no quantization_config; expected AWQ"
        )
    if str(quantization.get("quant_method", "")).casefold() != "awq":
        raise ValueError(
            "Target config must declare "
            "quantization_config.quant_method='awq'; "
            f"found {quantization.get('quant_method')!r}"
        )
    bits = quantization.get("bits")
    if bits is not None and bits != 4:
        raise ValueError(
            f"Expected a 4-bit AWQ target; found bits={bits!r}"
        )

    index_path = model_dir / "model.safetensors.index.json"
    weight_map = _read_json(index_path).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"Target safetensors index has no valid weight_map: {index_path}"
        )

    shard_names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
            raise ValueError(
                f"Invalid tensor-to-shard mapping in {index_path}"
            )
        if (
            not shard_name
            or PurePosixPath(shard_name).name != shard_name
            or "\\" in shard_name
            or shard_name in {".", ".."}
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError(
                f"Unsafe or unexpected shard path in target index: "
                f"{shard_name!r}"
            )
        shard_names.add(shard_name)

    for shard_name in sorted(shard_names):
        _require_file(model_dir / shard_name)


def _validate_eagle(model_dir: Path) -> None:
    """Validate the known EAGLE 3.1 Qwen3 drafter config contract."""
    _require_files(
        model_dir,
        ("config.json", "config.py", "model.safetensors"),
    )
    config_path = model_dir / "config.json"
    config = _read_json(config_path)
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not any(
        isinstance(name, str)
        and name.casefold() == "eagle3draftmodel"
        for name in architectures
    ):
        raise ValueError(
            "EAGLE config must declare architecture Eagle3DraftModel; "
            f"found {architectures!r} in {config_path}"
        )
    if config.get("speculators_model_type") != "eagle3":
        raise ValueError(
            "EAGLE config must declare speculators_model_type='eagle3'; "
            f"found {config.get('speculators_model_type')!r}"
        )
    if config.get("eagle31") is not True:
        raise ValueError(
            f"EAGLE config must declare eagle31=true: {config_path}"
        )
    speculators_config = config.get("speculators_config")
    if (
        not isinstance(speculators_config, dict)
        or str(speculators_config.get("algorithm", "")).casefold() != "eagle3"
    ):
        raise ValueError(
            "EAGLE config must declare "
            f"speculators_config.algorithm='eagle3': {config_path}"
        )
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict) or not any(
        isinstance(value, str) and value.startswith("config.")
        for value in auto_map.values()
    ):
        raise ValueError(
            f"EAGLE config is missing its config.py auto_map entry: "
            f"{config_path}"
        )


def _numpy_dtype(onnx_type: str, np: Any) -> Any:
    """Map supported ONNX input tensor types to NumPy dtypes."""
    type_map = {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    try:
        return type_map[onnx_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Prompt Guard ONNX input type: {onnx_type}"
        ) from exc


def _validate_guard(model_dir: Path) -> None:
    """Validate label metadata and execute one real CPU ONNX inference."""
    onnx_path = model_dir / GUARD_ONNX_NAME
    _require_files(
        model_dir,
        (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            GUARD_ONNX_NAME,
            "LICENSE",
            "NOTICE",
        ),
    )
    config = _read_json(model_dir / "config.json")
    id2label = config.get("id2label")
    if not isinstance(id2label, dict):
        raise ValueError(
            "Prompt Guard config is missing id2label metadata: "
            f"{model_dir / 'config.json'}"
        )
    labels = {
        str(key): str(value).strip().casefold()
        for key, value in id2label.items()
    }
    if labels.get("0") != "benign" or labels.get("1") != "malicious":
        raise ValueError(
            "Prompt Guard must map class 0 to BENIGN and class 1 to "
            f"MALICIOUS; found {id2label!r}"
        )
    num_labels = config.get("num_labels", len(id2label))
    if num_labels != 2:
        raise ValueError(
            "Prompt Guard must be a two-class classifier; "
            f"found num_labels={num_labels!r}"
        )

    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ValueError(
            "Guard validation requires numpy, onnxruntime, and tokenizers "
            "in the validator's Python environment"
        ) from exc

    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ValueError(
            "Cannot open Prompt Guard ONNX graph with CPUExecutionProvider: "
            f"{exc}"
        ) from exc

    providers = session.get_providers()
    if "CPUExecutionProvider" not in providers:
        raise ValueError(
            "Prompt Guard session is not using CPUExecutionProvider: "
            f"{providers!r}"
        )
    input_metadata = session.get_inputs()
    if not input_metadata:
        raise ValueError(
            f"Prompt Guard ONNX graph has no inputs: {onnx_path}"
        )

    try:
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        encoded = tokenizer.encode("Hello, world.")
    except Exception as exc:
        raise ValueError(
            f"Cannot load or run the Prompt Guard tokenizer: {exc}"
        ) from exc
    if not encoded.ids:
        raise ValueError(
            "Prompt Guard tokenizer returned no token IDs for the smoke input"
        )

    token_values: dict[str, list[int]] = {
        "input_ids": encoded.ids,
        "attention_mask": encoded.attention_mask,
        "token_type_ids": encoded.type_ids,
    }
    feed: dict[str, Any] = {}
    seen_input_ids = False
    for metadata in input_metadata:
        name = metadata.name.casefold()
        if name not in token_values:
            raise ValueError(
                f"Unsupported Prompt Guard ONNX input {metadata.name!r}; "
                "the validator supports input_ids, attention_mask, and "
                "token_type_ids"
            )
        values = token_values[name]
        if not values:
            raise ValueError(
                f"Tokenizer produced empty {metadata.name} input"
            )
        shape = metadata.shape
        if len(shape) != 2:
            raise ValueError(
                f"Expected rank-2 Prompt Guard input {metadata.name!r}; "
                f"shape={shape!r}"
            )
        if isinstance(shape[1], int) and shape[1] > 0:
            required_length = shape[1]
            values = values[:required_length]
            pad_id = config.get("pad_token_id", 0)
            pad_value = (
                int(pad_id)
                if name == "input_ids" and isinstance(pad_id, int)
                else 0
            )
            values = values + [pad_value] * (required_length - len(values))
        dtype = _numpy_dtype(metadata.type, np)
        feed[metadata.name] = np.asarray([values], dtype=dtype)
        if name == "input_ids":
            seen_input_ids = True
    if not seen_input_ids:
        raise ValueError(
            f"Prompt Guard ONNX graph has no input_ids input: {onnx_path}"
        )

    try:
        outputs = session.run(None, feed)
    except Exception as exc:
        raise ValueError(
            f"Prompt Guard ONNX smoke inference failed: {exc}"
        ) from exc
    if not outputs:
        raise ValueError("Prompt Guard ONNX graph returned no outputs")
    logits = np.asarray(outputs[0])
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError(
            "Prompt Guard's first ONNX output must be a two-class matrix "
            f"shaped (1, 2); found {logits.shape!r}"
        )
    if not np.isfinite(logits).all():
        raise ValueError(
            "Prompt Guard ONNX smoke inference returned non-finite values"
        )


def validate(model_root: Path, lock: dict[str, str]) -> None:
    """Run validators in deterministic order and print the lock identity."""
    validators = {
        "target": _validate_target,
        "eagle": _validate_eagle,
        "guard": _validate_guard,
    }
    for name, validator in validators.items():
        model_dir = model_root / name
        if not model_dir.is_dir():
            raise ValueError(f"Model directory is missing: {model_dir}")
        validator(model_dir)
        id_key, revision_key = LOCK_KEYS[name]
        print(
            f"OK: {name}: {lock[id_key]}@{lock[revision_key]}",
            flush=True,
        )
    print(
        "Model manifest and local model artifacts passed validation.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Secure LLM Deploy model artifacts."
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_ROOT", "/opt/models"),
        help="Directory containing target/, eagle/, and guard/",
    )
    parser.add_argument(
        "--lock-file",
        default=os.environ.get("MODEL_LOCK_FILE", str(DEFAULT_LOCK_FILE)),
        help="Path to the pinned KEY=VALUE model lock",
    )
    args = parser.parse_args()
    try:
        lock = read_lock(Path(args.lock_file))
        validate(Path(args.model_dir), lock)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())