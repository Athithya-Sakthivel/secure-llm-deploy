#!/usr/bin/env python3
"""Resolve a Hugging Face repository revision to an immutable commit SHA.

This maintainer utility writes a small KEY=VALUE file. Runtime builds should
consume the committed image/model.lock with all three model revisions pinned;
they must not resolve moving branches during production builds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
MODEL_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
HF_API = "https://huggingface.co/api/models/{}"


class ResolverArgs(argparse.Namespace):
    """Typed command-line arguments."""

    model_id: str
    output: str
    revision: str


def resolve(model_id: str, revision: str = "main") -> str:
    """Return the full commit SHA for a repository ref (branch, tag, or SHA)."""
    encoded_model_id = urllib.parse.quote(model_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"{HF_API.format(encoded_model_id)}/revision/{encoded_revision}"
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
        sys.exit(
            f"ERROR: Hugging Face API returned HTTP {exc.code} "
            f"for {model_id!r} at revision {revision!r}"
        )
    except urllib.error.URLError as exc:
        sys.exit(f"ERROR: could not reach Hugging Face API: {exc.reason}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        sys.exit(f"ERROR: Hugging Face API returned invalid JSON: {exc}")

    if not isinstance(payload, dict):
        sys.exit("ERROR: Hugging Face API response must be a JSON object")

    raw_sha = payload.get("sha")
    if not isinstance(raw_sha, str) or SHA_RE.fullmatch(raw_sha.strip()) is None:
        sys.exit(f"ERROR: API response has no valid commit SHA for {model_id!r}")

    return raw_sha.strip()


def main() -> None:
    """Resolve a repository ref and write a shell-sourceable lock file."""
    parser = argparse.ArgumentParser(
        description="Resolve a Hugging Face model revision to a commit SHA."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        type=str,
        help="Hugging Face repository ID, e.g. Qwen/Qwen3-4B",
    )
    parser.add_argument(
        "--revision",
        default="main",
        type=str,
        help="Repository branch, tag, or commit SHA to resolve (default: main)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Destination path for the two-line lock file",
    )

    args = parser.parse_args(namespace=ResolverArgs())
    model_id = args.model_id.strip()
    revision = args.revision.strip()
    output_name = args.output.strip()

    if MODEL_ID_RE.fullmatch(model_id) is None:
        sys.exit(f"ERROR: invalid MODEL_ID repository path: {model_id!r}")
    if not revision or "\n" in revision or "\r" in revision:
        sys.exit("ERROR: revision must be a nonempty single-line value")
    if not output_name:
        sys.exit("ERROR: output path must not be empty")

    sha = resolve(model_id, revision)
    output_path = Path(output_name)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        temporary_path.write_text(
            f"MODEL_ID={model_id}\nMODEL_REVISION={sha}\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_path.replace(output_path)
    except OSError as exc:
        sys.exit(f"ERROR: could not write lock file {str(output_path)!r}: {exc}")

    print(f"resolved {model_id}@{revision} -> {sha}", file=sys.stderr)


if __name__ == "__main__":
    main()
