#!/usr/bin/env python3
"""Validate the vLLM runtime and model lock file at build time."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import sys
from typing import NoReturn

LOCK_PATH = os.environ.get("VLLM_LOCK_FILE", "/etc/vllm/model.lock")
SHA_RE = re.compile(r"[0-9a-f]{40}")
MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")

EXPECTED_VLLM = "0.31.0"
EXPECTED_CUDA_PREFIX = "12.9"

EXPECTED_LOCK_KEYS = {"MODEL_ID", "MODEL_REVISION"}


def fail(message: str) -> NoReturn:
    """Exit with a consistent validation error."""
    sys.exit(f"ERROR: {message}")


def check_vllm() -> None:
    """Verify the installed vLLM version."""
    try:
        installed = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        fail("base image does not contain vLLM")

    # Ignore the PEP 440 local version segment, such as +cu129.
    base_version = installed.split("+", maxsplit=1)[0]

    if base_version != EXPECTED_VLLM:
        fail(f"expected vLLM {EXPECTED_VLLM}, found {installed!r}")


def check_cuda() -> None:
    """Verify that PyTorch is importable and uses the expected CUDA version."""
    try:
        torch_module = importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError) as exc:
        fail(f"PyTorch import failed: {exc}")

    torch_version: object = getattr(torch_module, "version", None)
    cuda_version: object = getattr(torch_version, "cuda", None)

    actual = cuda_version if isinstance(cuda_version, str) else ""
    valid_version = actual == EXPECTED_CUDA_PREFIX or actual.startswith(f"{EXPECTED_CUDA_PREFIX}.")

    if not valid_version:
        fail(f"expected CUDA {EXPECTED_CUDA_PREFIX}.x, found {actual!r}")


def check_lock() -> None:
    """Validate the model ID and immutable revision in the lock file."""
    if not os.path.isfile(LOCK_PATH):
        fail(f"lock file missing at {LOCK_PATH!r}")

    values: dict[str, str] = {}

    try:
        with open(LOCK_PATH, encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                key, separator, value = line.partition("=")

                if not separator:
                    fail(f"malformed lock line {line_number}: {line!r}")

                key = key.strip()
                value = value.strip()

                if key not in EXPECTED_LOCK_KEYS:
                    fail(f"unexpected key {key!r} on lock line {line_number}")

                if key in values:
                    fail(f"duplicate key {key!r} on lock line {line_number}")

                values[key] = value

    except OSError as exc:
        fail(f"could not read lock file {LOCK_PATH!r}: {exc}")

    missing_keys = EXPECTED_LOCK_KEYS.difference(values)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        fail(f"lock file is missing required keys: {missing}")

    model_id = values["MODEL_ID"]
    revision = values["MODEL_REVISION"]

    if MODEL_ID_RE.fullmatch(model_id) is None:
        fail(f"MODEL_ID is not a valid namespace/repository ID: {model_id!r}")

    if SHA_RE.fullmatch(revision) is None:
        fail(f"MODEL_REVISION must be a 40-character lowercase commit SHA, got {revision!r}")

    print(f"validated lock: {model_id}@{revision}")


def main() -> None:
    """Run all runtime validation checks."""
    check_vllm()
    check_cuda()
    check_lock()
    print("runtime validation passed")


if __name__ == "__main__":
    main()
