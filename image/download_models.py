"""Download only the pinned artifacts required by Secure LLM Deploy.

Model revisions are read from /etc/vllm/models.lock. Target and EAGLE
weights are separate artifacts: the EAGLE checkpoint is not a standalone
language model and does not require a separate draft tokenizer.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from huggingface_hub import snapshot_download

LOCK_FILE = Path(os.environ.get("MODEL_LOCK_FILE", "/etc/vllm/models.lock"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/opt/models"))
REVISION_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-fA-F]{40}\Z")


@dataclass(frozen=True)
class ModelSpec:
    """Build-time metadata for one model artifact."""

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
        ),
        required_files=("config.json", "tokenizer.json"),
    ),
    ModelSpec(
        directory="eagle",
        id_key="EAGLE_MODEL_ID",
        revision_key="EAGLE_MODEL_REVISION",
        # The speculator needs its configuration and weight file. Do not
        # require or download a separate tokenizer for this checkpoint.
        allow_patterns=("config.json", "model.safetensors"),
        required_files=("config.json", "model.safetensors"),
    ),
    ModelSpec(
        directory="guard",
        id_key="GUARD_MODEL_ID",
        revision_key="GUARD_MODEL_REVISION",
        # Vela guard inference uses ONNX Runtime on CPU. Download the ONNX
        # graph and its external-data sidecar, not the 1.23 GB HF weights.
        allow_patterns=(
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "label_mapping.json",
            "jailbreak_type_mapping.json",
            "onnx/model.onnx",
            "onnx/model.onnx.data",
        ),
        required_files=(
            "config.json",
            "tokenizer.json",
            "label_mapping.json",
            "onnx/model.onnx",
            "onnx/model.onnx.data",
        ),
    ),
)


def read_lock(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE lock file without executing it."""
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
            raise ValueError(f"Empty key or value in lock-file line {line_number}")
        if key in values:
            raise ValueError(f"Duplicate key in model lock: {key}")
        values[key] = value

    required_keys = {
        key
        for spec in MODEL_SPECS
        for key in (spec.id_key, spec.revision_key)
    }
    missing = sorted(required_keys - values.keys())
    if missing:
        raise ValueError(f"Model lock is missing required keys: {', '.join(missing)}")

    unknown = sorted(values.keys() - required_keys)
    if unknown:
        raise ValueError(f"Model lock contains unknown keys: {', '.join(unknown)}")

    return values


def _validate_target_artifacts(destination: Path) -> None:
    """Validate the target's shard index and every referenced weight shard."""
    index_path = destination / "model.safetensors.index.json"
    if not index_path.is_file():
        shards = list(destination.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"No target safetensors weights were downloaded into {destination}")
        return

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid target safetensors index: {index_path}") from exc

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Target safetensors index has no valid weight_map: {index_path}")

    for shard_name in sorted(set(weight_map.values())):
        if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
            raise ValueError(f"Unsafe shard path in target index: {shard_name!r}")
        if not (destination / shard_name).is_file():
            raise ValueError(f"Target weight shard is missing: {shard_name}")


def _validate_eagle_config(destination: Path) -> None:
    """Reject a normal draft LM accidentally placed in the EAGLE directory."""
    config_path = destination / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid EAGLE config: {config_path}") from exc

    architectures = config.get("architectures", []) if isinstance(config, dict) else []
    if not isinstance(architectures, list) or not any(
        isinstance(architecture, str) and "eagle3" in architecture.casefold()
        for architecture in architectures
    ):
        raise ValueError(
            "The configured speculator does not declare an EAGLE-3 architecture; "
            f"found {architectures!r} in {config_path}"
        )


def _validate_files(destination: Path, required_files: tuple[str, ...]) -> None:
    missing = [name for name in required_files if not (destination / name).is_file()]
    if missing:
        raise ValueError(
            f"Required downloaded artifacts are missing in {destination}: "
            + ", ".join(missing)
        )


def main() -> None:
    """Download pinned target, EAGLE-3, and guard artifacts."""
    lock = read_lock(LOCK_FILE)
    token = os.environ.get("HF_TOKEN") or None
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS:
        repo_id = lock[spec.id_key]
        revision = lock[spec.revision_key]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
            raise ValueError(f"Invalid Hugging Face repository ID in {spec.id_key}: {repo_id!r}")
        if REVISION_RE.fullmatch(revision) is None:
            raise ValueError(
                f"{spec.revision_key} must be a full 40-character commit SHA; got {revision!r}"
            )

        destination = MODEL_ROOT / spec.directory
        # Remove stale files if this script is rerun in an existing build stage.
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)

        print(
            f"Downloading {repo_id}@{revision} -> {destination} "
            f"(artifact set: {spec.directory})",
            flush=True,
        )
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(spec.allow_patterns),
            token=token,
        )

        # huggingface_hub may leave download bookkeeping beneath local_dir.
        shutil.rmtree(destination / ".cache", ignore_errors=True)
        _validate_files(destination, spec.required_files)

        if spec.directory == "target":
            _validate_target_artifacts(destination)
        elif spec.directory == "eagle":
            _validate_eagle_config(destination)

    print("All pinned model artifacts downloaded and validated.", flush=True)


if __name__ == "__main__":
    main()
