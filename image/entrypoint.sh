#!/usr/bin/env bash
set -Eeuo pipefail

readonly LOCK_FILE="/etc/vllm/models.lock"
readonly UVICORN="/opt/app/.venv/bin/uvicorn"

_die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 64
}

# Validate and normalize a port. Normalization prevents strings such as 08000
# and 8000 from passing the distinct-port check while naming the same port.
_validate_port() {
    local name="$1"
    local value="$2"
    local port

    [[ "$value" =~ ^[0-9]{1,5}$ ]] ||
        _die "${name} must be a numeric port between 1 and 65535."

    port=$((10#${value}))
    (( port >= 1 && port <= 65535 )) ||
        _die "${name} must be between 1 and 65535."

    printf '%s' "$port"
}

_validate_single_line() {
    local name="$1"
    local value="$2"

    [[ -n "$value" && "$value" != *$'\n'* && "$value" != *$'\r'* ]] ||
        _die "${name} must be a nonempty single-line value."
}

# This manifest is validated during image build and parsed as data, never
# sourced or evaluated as shell code.
[[ -r "$LOCK_FILE" && -s "$LOCK_FILE" ]] ||
    _die "Pinned model manifest is missing or unreadable: ${LOCK_FILE}"

export SERVE_MODE="${SERVE_MODE:-baseline}"
case "$SERVE_MODE" in
    baseline|speculative) ;;
    *) _die "SERVE_MODE must be baseline or speculative." ;;
esac

# Keep these paths synchronized with image/download_models.py,
# image/validate.py and secure_llm_deploy.config.Settings.
export MODEL_ROOT="${MODEL_ROOT:-/opt/models}"
export MODEL_LOCK_FILE="$LOCK_FILE"
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-${MODEL_ROOT}/target}"
export TARGET_TOKENIZER_PATH="${TARGET_TOKENIZER_PATH:-${TARGET_MODEL_PATH}/tokenizer.json}"
export EAGLE_MODEL_PATH="${EAGLE_MODEL_PATH:-${MODEL_ROOT}/eagle}"
export GUARD_MODEL_PATH="${GUARD_MODEL_PATH:-${MODEL_ROOT}/guard/model.QUInt8.quant.onnx}"
export GUARD_TOKENIZER_PATH="${GUARD_TOKENIZER_PATH:-${MODEL_ROOT}/guard/tokenizer.json}"
export GUARD_CONFIG_PATH="${GUARD_CONFIG_PATH:-${MODEL_ROOT}/guard/config.json}"
export GUARD_THRESHOLD="${GUARD_THRESHOLD:-0.5}"
export GUARD_MAX_INPUT_TOKENS="${GUARD_MAX_INPUT_TOKENS:-6144}"
export GUARD_WINDOW_TOKENS="${GUARD_WINDOW_TOKENS:-512}"
export GUARD_INTRA_OP_THREADS="${GUARD_INTRA_OP_THREADS:-2}"
export GUARD_FAILURE_POLICY="${GUARD_FAILURE_POLICY:-closed}"
export GUARD_OVERLONG_POLICY="${GUARD_OVERLONG_POLICY:-reject}"

export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="$(_validate_port GATEWAY_PORT "${GATEWAY_PORT:-8000}")"
VLLM_PORT="$(_validate_port VLLM_PORT "${VLLM_PORT:-8001}")"
export GATEWAY_PORT VLLM_PORT
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-vllm}"

[[ "$VLLM_HOST" == "127.0.0.1" ]] ||
    _die "VLLM_HOST must be 127.0.0.1; the internal server must not be exposed."

[[ "$GATEWAY_PORT" != "$VLLM_PORT" ]] ||
    _die "GATEWAY_PORT and VLLM_PORT must differ."

_validate_single_line GATEWAY_HOST "$GATEWAY_HOST"
_validate_single_line VLLM_EXECUTABLE "$VLLM_EXECUTABLE"

[[ "$VLLM_EXECUTABLE" != *[[:space:]]* ]] ||
    _die "VLLM_EXECUTABLE must not contain whitespace."

: "${GATEWAY_API_KEY:?GATEWAY_API_KEY must be configured}"
: "${VLLM_API_KEY:?VLLM_API_KEY must be configured}"

_validate_single_line GATEWAY_API_KEY "$GATEWAY_API_KEY"
_validate_single_line VLLM_API_KEY "$VLLM_API_KEY"

(( ${#GATEWAY_API_KEY} >= 32 )) ||
    _die "GATEWAY_API_KEY must contain at least 32 characters."

(( ${#VLLM_API_KEY} >= 32 )) ||
    _die "VLLM_API_KEY must contain at least 32 characters."

[[ "$GATEWAY_API_KEY" != "$VLLM_API_KEY" ]] ||
    _die "GATEWAY_API_KEY and VLLM_API_KEY must be different."

[[ -x "$UVICORN" ]] ||
    _die "Gateway virtual environment is missing its uvicorn executable."

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

umask 027

printf 'Starting secure-llm-deploy gateway (mode=%s, port=%s).\n' \
    "$SERVE_MODE" "$GATEWAY_PORT"

# Uvicorn owns the gateway process; the application lifespan supervises vLLM.
exec "$UVICORN" \
    secure_llm_deploy.main:app \
    --host "$GATEWAY_HOST" \
    --port "$GATEWAY_PORT" \
    --workers 1 \
    --no-access-log \
    --log-level info