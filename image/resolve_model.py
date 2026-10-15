#!/usr/bin/env python3
"""Resolve a Hugging Face model ID to its immutable commit SHA.

Writes a shell-sourceable lock file consumed by the runtime stage
and container entrypoint. Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

SHA_RE = re.compile(r"[0-9a-f]{40}")
MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
HF_API = "https://huggingface.co/api/models/{}"


class ResolverArgs(argparse.Namespace):
    """Typed command-line arguments."""

    model_id: str
    output: str


def resolve(model_id: str) -> str:
    """Return the 40-character commit SHA for a Hugging Face model."""
    encoded_model_id = urllib.parse.quote(model_id, safe="/")
    url = HF_API.format(encoded_model_id)

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "secure-llm-deploy/resolver",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload: object = json.load(response)
    except urllib.error.HTTPError as exc:
        sys.exit(f"ERROR: Hugging Face API returned HTTP {exc.code} for {model_id!r}")
    except urllib.error.URLError as exc:
        sys.exit(f"ERROR: could not reach Hugging Face API: {exc.reason}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        sys.exit(f"ERROR: Hugging Face API returned invalid JSON: {exc}")

    if not isinstance(payload, dict):
        sys.exit("ERROR: Hugging Face API response must be a JSON object")

    raw_sha = payload.get("sha")
    if not isinstance(raw_sha, str):
        sys.exit(f"ERROR: API response has no valid SHA for {model_id!r}")

    sha = raw_sha.strip()
    if SHA_RE.fullmatch(sha) is None:
        sys.exit(f"ERROR: malformed SHA {sha!r} for {model_id!r}")

    return sha


def main() -> None:
    """Parse arguments, resolve the revision, and write the lock file."""
    parser = argparse.ArgumentParser(description="Resolve a Hugging Face model ID to a commit SHA.")
    parser.add_argument(
        "--model-id",
        required=True,
        type=str,
        help="Hugging Face repository ID in namespace/repository format",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Destination path for the lock file",
    )

    args = parser.parse_args(namespace=ResolverArgs())
    model_id = args.model_id.strip()
    output_path = args.output.strip()

    if MODEL_ID_RE.fullmatch(model_id) is None:
        sys.exit(f"ERROR: MODEL_ID must be a valid namespace/repository ID; got {model_id!r}")

    if not output_path:
        sys.exit("ERROR: output path must not be empty")

    sha = resolve(model_id)

    try:
        with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"MODEL_ID={model_id}\n")
            handle.write(f"MODEL_REVISION={sha}\n")
    except OSError as exc:
        sys.exit(f"ERROR: could not write lock file {output_path!r}: {exc}")

    print(f"resolved {model_id} -> {sha}", file=sys.stderr)


if __name__ == "__main__":
    main()
