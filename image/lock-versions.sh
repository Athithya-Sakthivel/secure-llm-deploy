#!/usr/bin/env bash
# lock-versions.sh — Regenerate uv.lock and sync the venv.
#
# Usage:
#   bash image/lock-versions.sh
#   bash image/lock-versions.sh --upgrade

set -Eeuo pipefail

# This script lives in image/, but pyproject.toml and uv.lock live
# in the repository root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

UV_VERSION="0.12.18"
export UV_LINK_MODE=copy

# Ensure project metadata exists.
[[ -f pyproject.toml ]] || {
  echo "ERROR: pyproject.toml not found in $PROJECT_ROOT" >&2
  exit 1
}

# Install pinned uv if missing or wrong version.
if ! command -v uv >/dev/null 2>&1 \
  || [[ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]]; then
  echo "Installing uv $UV_VERSION..."
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
    | env UV_UNMANAGED_INSTALL=/usr/local/bin sh
fi

# Create the project virtual environment if needed.
if [[ ! -d .venv ]]; then
  echo "Creating Python 3.14 virtual environment..."
  uv venv --python 3.14
fi

# Regenerate the lockfile.
if [[ "${1:-}" == "--upgrade" ]]; then
  echo "Upgrading dependencies..."
  uv lock --upgrade
else
  echo "Regenerating lockfile..."
  uv lock
fi

# Verify lockfile and sync runtime + all dependency groups.
echo "Checking lockfile..."
uv lock --check

echo "Syncing environment..."
uv sync --locked --all-groups

echo "✓ Dependencies locked and virtual environment synced"
echo "Activate with: source $PROJECT_ROOT/.venv/bin/activate"