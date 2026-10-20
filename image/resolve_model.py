#!/usr/bin/env python3
"""Resolve a Hugging Face repository ref to an immutable commit SHA.

This maintainer tool emits a generic MODEL_ID/MODEL_REVISION fragment. It is
not used in production image builds; those use the committed image/model.lock.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import suppress
from pathlib import Path

SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
HF_API = "https://huggingface.co/api/models/{model_id}/revision/{revision}"
USER_AGENT = "secure-llm-deploy-model-resolver/1.0"


class ResolverError(RuntimeError):
    """Expected input or API failure suitable for a concise CLI message."""


def resolve(
    model_id: str,
    revision: str = "main",
    *,
    timeout: float = 30.0,
) -> str:
    """Return the immutable commit SHA for a public or token-accessible HF repo ref."""
    model_id = model_id.strip()
    revision = revision.strip()
    if MODEL_ID_RE.fullmatch(model_id) is None:
        raise ResolverError(f"invalid Hugging Face repository ID: {model_id!r}")
    if not revision or "\n" in revision or "\r" in revision:
        raise ResolverError("revision must be a nonempty single-line value")
    if timeout <= 0:
        raise ResolverError("timeout must be greater than zero")

    url = HF_API.format(
        model_id=urllib.parse.quote(model_id, safe="/"),
        revision=urllib.parse.quote(revision, safe=""),
    )
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: object = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ResolverError(
            f"Hugging Face API returned HTTP {exc.code} for {model_id!r} at ref {revision!r}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ResolverError(f"could not reach Hugging Face API: {exc.reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ResolverError(f"Hugging Face API returned invalid JSON: {exc}") from exc
    except (TimeoutError, OSError) as exc:
        raise ResolverError(f"network error while contacting Hugging Face API: {exc}") from exc

    if not isinstance(payload, dict):
        raise ResolverError("Hugging Face API response must be a JSON object")
    sha = payload.get("sha")
    if not isinstance(sha, str) or SHA_RE.fullmatch(sha) is None:
        raise ResolverError(f"API response has no valid commit SHA for {model_id!r}")
    return sha.lower()


def _atomic_write(output_path: Path, content: str) -> None:
    """Atomically replace the output file using a temporary file in its directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temp_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temp_name, output_path)
        temp_name = None
    finally:
        if temp_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temp_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a Hugging Face model ref to a pinned commit SHA."
    )
    parser.add_argument("--model-id", required=True, help="Repository ID, e.g. Qwen/Qwen3-8B")
    parser.add_argument(
        "--revision", default="main", help="Branch, tag, or commit SHA (default: main)"
    )
    parser.add_argument("--output", required=True, help="Destination for KEY=VALUE lock fragment")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    model_id = args.model_id.strip()
    revision = args.revision.strip()
    output_name = args.output.strip()
    if not output_name:
        parser.error("--output must not be empty")

    try:
        sha = resolve(model_id, revision, timeout=args.timeout)
        _atomic_write(Path(output_name), f"MODEL_ID={model_id}\nMODEL_REVISION={sha}\n")
    except (OSError, ResolverError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"resolved {model_id}@{revision} -> {sha}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())