#!/usr/bin/env python3
"""Download only the commit-pinned model artifacts required by this image.

The model lock is parsed as data; it is never sourced or executed as shell code.
The Prompt Guard ONNX file is normalized to ``model.QUInt8.quant.onnx`` so
runtime paths remain stable even if the upstream repository uses another name
for the same quantized artifact (for example, ``model.quant.onnx``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from huggingface_hub import snapshot_download

LOCK_FILE = Path(os.environ.get("MODEL_LOCK_FILE", "/etc/vllm/models.lock"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/opt/models"))
REVISION_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-fA-F]{40}\Z")
MODEL_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
GUARD_ONNX_NAME: Final[str] = "model.QUInt8.quant.onnx"
GUARD_ONNX_CANDIDATES: Final[tuple[str, ...]] = (
    GUARD_ONNX_NAME,
    "model.quant.onnx",
    "model.onnx",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Allow-list and required files for one model directory."""

    directory: str
    id_key: str
    revision_key: str
    allow_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


MODEL_SPECS: Final[tuple[ModelSpec, ...]] = (
    ModelSpec(
        directory="target",
        id_key="TARGET_MODEL_ID",
        revision_key="TARGET_MODEL_REVISION",
        allow_patterns=(
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors.index.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.jinja",
            "*.tiktoken",
            "vocab.*",
            "merges.*",
            "LICENSE*",
            "NOTICE*",
        ),
        required_files=(
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        ),
    ),
    ModelSpec(
        directory="eagle",
        id_key="EAGLE_MODEL_ID",
        revision_key="EAGLE_MODEL_REVISION",
        # config.py is imported by the pinned checkpoint's auto_map. Training
        # optimizer/scheduler checkpoints are intentionally not downloaded.
        allow_patterns=(
            "config.json",
            "config.py",
            "model.safetensors",
            "LICENSE*",
            "NOTICE*",
            "README*",
        ),
        required_files=("config.json", "config.py", "model.safetensors"),
    ),
    ModelSpec(
        directory="guard",
        id_key="GUARD_MODEL_ID",
        revision_key="GUARD_MODEL_REVISION",
        # Accept the canonical filename and common upstream ONNX export names;
        # _normalize_guard_onnx() creates the canonical path after download.
        allow_patterns=(
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.QUInt8.quant.onnx",
            "model.quant.onnx",
            "model.onnx",
            "LICENSE",
            "NOTICE",
            "README*",
            "USE_POLICY.md",
        ),
        required_files=(
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "LICENSE",
            "NOTICE",
        ),
    ),
)


def _read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object and reject malformed or non-object JSON."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def read_lock(path: Path) -> dict[str, str]:
    """Parse and validate a strict KEY=VALUE model lock without executing it."""
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
                f"Invalid lock-file line {line_number}: expected KEY=VALUE"
            )
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            raise ValueError(
                f"Empty key or value in lock-file line {line_number}"
            )
        if key in values:
            raise ValueError(f"Duplicate key in model lock: {key}")
        values[key] = value

    expected_keys = {
        key
        for spec in MODEL_SPECS
        for key in (spec.id_key, spec.revision_key)
    }
    missing = sorted(expected_keys - values.keys())
    unknown = sorted(values.keys() - expected_keys)
    if missing:
        raise ValueError(
            f"Model lock is missing required keys: {', '.join(missing)}"
        )
    if unknown:
        raise ValueError(
            f"Model lock contains unknown keys: {', '.join(unknown)}"
        )

    for spec in MODEL_SPECS:
        repo_id = values[spec.id_key]
        revision = values[spec.revision_key]
        if MODEL_ID_RE.fullmatch(repo_id) is None:
            raise ValueError(
                f"Invalid Hugging Face repository ID in "
                f"{spec.id_key}: {repo_id!r}"
            )
        if REVISION_RE.fullmatch(revision) is None:
            raise ValueError(
                f"{spec.revision_key} must be a full 40-character "
                f"commit SHA; got {revision!r}"
            )
        values[spec.revision_key] = revision.lower()
    return values


def _require_nonempty_file(path: Path) -> None:
    """Require a nonempty regular file that the current process can open."""
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
        _require_nonempty_file(directory / filename)


def _validate_target_index(destination: Path) -> None:
    """Ensure every tensor shard named in the target index exists safely."""
    index_path = destination / "model.safetensors.index.json"
    index = _read_json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Target index has no valid weight_map: {index_path}")

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
        _require_nonempty_file(destination / shard_name)


def _validate_eagle_config(destination: Path) -> None:
    """Check the known EAGLE-3.1 config contract for the pinned Qwen3 drafter."""
    config_path = destination / "config.json"
    config = _read_json_object(config_path)
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not any(
        isinstance(name, str) and name.casefold() == "eagle3draftmodel"
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
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict) or not any(
        isinstance(value, str) and value.startswith("config.")
        for value in auto_map.values()
    ):
        raise ValueError(
            f"EAGLE config is missing its config.py auto_map entry: "
            f"{config_path}"
        )
    _require_nonempty_file(destination / "config.py")


def _normalize_guard_onnx(destination: Path) -> None:
    """Normalize one known quantized ONNX filename and remove duplicate variants."""
    available: list[Path] = []
    for filename in GUARD_ONNX_CANDIDATES:
        path = destination / filename
        if path.is_file() and path.stat().st_size > 0:
            available.append(path)
    if not available:
        expected = ", ".join(GUARD_ONNX_CANDIDATES)
        raise ValueError(
            f"No nonempty Prompt Guard ONNX artifact found in {destination}; "
            f"tried {expected}"
        )

    canonical = destination / GUARD_ONNX_NAME
    source = next(
        (path for path in available if path.name == GUARD_ONNX_NAME),
        available[0],
    )
    if source != canonical:
        os.replace(source, canonical)
    _require_nonempty_file(canonical)
    for path in available:
        if path != canonical and path.exists():
            path.unlink()


def main() -> int:
    """Download and structurally validate all lock-pinned model artifacts."""
    lock = read_lock(LOCK_FILE)
    token = os.environ.get("HF_TOKEN") or None
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS:
        repo_id = lock[spec.id_key]
        revision = lock[spec.revision_key]
        destination = MODEL_ROOT / spec.directory

        # These are fixed child paths; no lock value is interpolated into a path.
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id}@{revision} -> {destination}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(spec.allow_patterns),
            token=token,
        )
        shutil.rmtree(destination / ".cache", ignore_errors=True)

        if spec.directory == "guard":
            _normalize_guard_onnx(destination)
        _require_files(destination, spec.required_files)

        if spec.directory == "target":
            _validate_target_index(destination)
        elif spec.directory == "eagle":
            _validate_eagle_config(destination)

    print(
        "All pinned model artifacts downloaded and structurally validated.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc