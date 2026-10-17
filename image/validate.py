
"""Validate the pinned model manifest and downloaded model artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Final

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


def read_lock(path: Path) -> dict[str, str]:
    """Read a KEY=VALUE manifest without executing it."""
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

    expected_keys = {
        key
        for id_key, revision_key in LOCK_KEYS.values()
        for key in (id_key, revision_key)
    }

    missing = sorted(expected_keys - values.keys())
    if missing:
        raise ValueError(
            "Model lock is missing required keys: " + ", ".join(missing)
        )

    unknown = sorted(values.keys() - expected_keys)
    if unknown:
        raise ValueError(
            "Model lock contains unknown keys: " + ", ".join(unknown)
        )

    for name, (id_key, revision_key) in LOCK_KEYS.items():
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
    """Read a JSON object and reject malformed or unexpected top-level data."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return payload


def _require_nonempty_readable(path: Path) -> None:
    """Require a nonempty regular file with read permission bits."""
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ValueError(f"Required artifact is missing: {path}") from exc

    if not path.is_file() or metadata.st_size <= 0:
        raise ValueError(f"Required artifact is missing or empty: {path}")

    if not metadata.st_mode & 0o444:
        raise ValueError(f"Artifact has no read permission bits: {path}")


def _require_files(directory: Path, filenames: tuple[str, ...]) -> None:
    """Validate a group of required artifact files."""
    for filename in filenames:
        _require_nonempty_readable(directory / filename)


def _validate_target(model_dir: Path) -> None:
    """Validate the Qwen3 target config, tokenizer, and safetensors weights."""
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"

    _require_files(model_dir, ("config.json", "tokenizer.json"))
    config = _read_json(config_path)

    architectures = config.get("architectures")
    model_type = config.get("model_type")

    if not isinstance(architectures, list) or not any(
        isinstance(name, str) and name.casefold() == "qwen3forcausallm"
        for name in architectures
    ):
        raise ValueError(
            "Target config must declare Qwen3ForCausalLM; "
            f"found {architectures!r} in {config_path}"
        )

    if not isinstance(model_type, str) or model_type.casefold() != "qwen3":
        raise ValueError(
            f"Target config must declare model_type='qwen3'; "
            f"found {model_type!r} in {config_path}"
        )

    index_path = model_dir / "model.safetensors.index.json"
    shard_files = sorted(model_dir.glob("*.safetensors"))

    if not shard_files:
        raise ValueError(f"No target safetensors weights found in {model_dir}")

    for shard_path in shard_files:
        _require_nonempty_readable(shard_path)

    if index_path.is_file():
        _require_nonempty_readable(index_path)
        index = _read_json(index_path)
        weight_map = index.get("weight_map")

        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"Target safetensors index has no valid weight_map: {index_path}"
            )

        expected_shards: set[str] = set()

        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(
                shard_name, str
            ):
                raise ValueError(
                    f"Invalid tensor mapping in target index: {index_path}"
                )

            if Path(shard_name).name != shard_name:
                raise ValueError(
                    f"Unsafe shard path in target index: {shard_name!r}"
                )

            expected_shards.add(shard_name)

        for shard_name in sorted(expected_shards):
            _require_nonempty_readable(model_dir / shard_name)

    # Keep the local assignment explicit so static analysis catches any
    # accidental future removal of the tokenizer validation above.
    del tokenizer_path


def _validate_eagle(model_dir: Path) -> None:
    """Validate an EAGLE-3 speculator checkpoint, not a standalone draft LM."""
    _require_files(model_dir, ("config.json", "model.safetensors"))
    config_path = model_dir / "config.json"
    config = _read_json(config_path)

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not any(
        isinstance(name, str) and "eagle3" in name.casefold()
        for name in architectures
    ):
        raise ValueError(
            "Speculator config must declare an EAGLE-3 architecture; "
            f"found {architectures!r} in {config_path}"
        )

    if config.get("model_type") != "llama":
        raise ValueError(
            "The pinned Qwen3-4B EAGLE-3 checkpoint is expected to use "
            f"model_type='llama'; found {config.get('model_type')!r}"
        )


def _validate_guard(model_dir: Path) -> None:
    """Validate the Vela guard's CPU ONNX inference artifacts."""
    required_files = (
        "config.json",
        "tokenizer.json",
        "label_mapping.json",
        "onnx/model.onnx",
        "onnx/model.onnx.data",
    )
    _require_files(model_dir, required_files)

    # Ensure text metadata is parseable JSON, not merely a nonempty file.
    _read_json(model_dir / "config.json")
    _read_json(model_dir / "label_mapping.json")


def validate(model_root: Path, lock: dict[str, str]) -> None:
    """Validate all downloaded artifacts against the supplied model lock."""
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

    print("Model manifest and local model artifacts passed validation.")


def main() -> int:
    """Validate a model root using a pinned lock manifest."""
    parser = argparse.ArgumentParser(
        description="Validate Secure LLM Deploy model artifacts."
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_ROOT", "/opt/models"),
        help="Directory containing target/, eagle/, and guard/ artifacts.",
    )
    parser.add_argument(
        "--lock-file",
        default=os.environ.get("MODEL_LOCK_FILE", str(DEFAULT_LOCK_FILE)),
        help="Path to the pinned KEY=VALUE model lock.",
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
