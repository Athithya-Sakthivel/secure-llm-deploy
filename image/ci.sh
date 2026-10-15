#!/usr/bin/env bash
set -Eeuo pipefail


uv lock --check
uv sync --locked --group dev

uv run python -m compileall -q image/src/secure_llm_deploy tests
uv run ruff check . --fix
uv run mypy --no-incremental

uv run pytest --collect-only -q
uv run pytest -q


cd "$(dirname "$0")"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
IMAGE_TAG="${IMAGE_TAG:-secure-llm-vllm:0.31.0}"
PLATFORM="${PLATFORM:-linux/amd64}"

docker buildx build \
  --platform "${PLATFORM}" \
  --load \
  --build-arg MODEL_ID="${MODEL_ID}" \
  --tag "${IMAGE_TAG}" \
  .

echo "built: ${IMAGE_TAG}  (model: ${MODEL_ID})"