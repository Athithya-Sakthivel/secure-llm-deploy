#!/usr/bin/env bash
set -Eeuo pipefail

LOCK_FILE="${VLLM_LOCK_FILE:-/etc/vllm/model.lock}"
[[ -r "${LOCK_FILE}" ]] || { printf 'ERROR: lock file not readable: %s\n' "${LOCK_FILE}" >&2; exit 64; }

# shellcheck disable=SC1090
source "${LOCK_FILE}"

: "${MODEL_ID:?MODEL_ID missing from lock file}"
: "${MODEL_REVISION:?MODEL_REVISION missing from lock file}"
: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8000}"

if [[ ! "${VLLM_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] || (( VLLM_PORT > 65535 )); then
  printf 'ERROR: VLLM_PORT must be 1..65535; got %q\n' "${VLLM_PORT}" >&2
  exit 64
fi

args=(
  serve "${MODEL_ID}"
  --revision "${MODEL_REVISION}"
  --tokenizer-revision "${MODEL_REVISION}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
)

[[ -n "${VLLM_API_KEY:-}"          ]] && args+=(--api-key "${VLLM_API_KEY}")
[[ -n "${VLLM_DTYPE:-}"            ]] && args+=(--dtype "${VLLM_DTYPE}")
[[ -n "${VLLM_MAX_MODEL_LEN:-}"    ]] && args+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
[[ -n "${VLLM_GPU_MEM_UTIL:-}"     ]] && args+=(--gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}")
[[ -n "${VLLM_TRUST_REMOTE_CODE:-}" ]] && args+=(--trust-remote-code)

exec vllm "${args[@]}" "$@"