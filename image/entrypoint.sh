#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 64
}

validate_port() {
    local name="$1"
    local value="$2"

    [[ "$value" =~ ^[0-9]{1,5}$ ]] ||
        die "${name} must be a numeric port between 1 and 65535."

    local port=$((10#$value))
    (( port >= 1 && port <= 65535 )) ||
        die "${name} must be between 1 and 65535."
}

# This is build metadata; never execute it as shell code.
LOCK_FILE="/etc/vllm/models.lock"

[[ -r "$LOCK_FILE" && -s "$LOCK_FILE" ]] ||
    die "Pinned model manifest is missing or unreadable: ${LOCK_FILE}"

export SERVE_MODE="${SERVE_MODE:-baseline}"
case "$SERVE_MODE" in
    baseline|speculative)
        ;;
    *)
        die "SERVE_MODE must be baseline or speculative."
        ;;
esac

# The gateway is externally exposed; vLLM must remain private.
export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
export GATEWAY_PORT="${GATEWAY_PORT:-8000}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8001}"
export VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-vllm}"

[[ "$VLLM_HOST" == "127.0.0.1" ]] ||
    die "VLLM_HOST must be 127.0.0.1; the internal server must not be exposed."

validate_port GATEWAY_PORT "$GATEWAY_PORT"
validate_port VLLM_PORT "$VLLM_PORT"

[[ "$GATEWAY_PORT" != "$VLLM_PORT" ]] ||
    die "GATEWAY_PORT and VLLM_PORT must differ."

[[ -n "$VLLM_EXECUTABLE" && "$VLLM_EXECUTABLE" != *$'\n'* ]] ||
    die "VLLM_EXECUTABLE must be a nonempty single-line executable name."

: "${GATEWAY_API_KEY:?GATEWAY_API_KEY must be configured}"
: "${VLLM_API_KEY:?VLLM_API_KEY must be configured}"

[[ "$GATEWAY_API_KEY" != *$'\n'* && "$GATEWAY_API_KEY" != *$'\r'* ]] ||
    die "GATEWAY_API_KEY must be a single-line value."

[[ "$VLLM_API_KEY" != *$'\n'* && "$VLLM_API_KEY" != *$'\r'* ]] ||
    die "VLLM_API_KEY must be a single-line value."

(( ${#GATEWAY_API_KEY} >= 32 )) ||
    die "GATEWAY_API_KEY must contain at least 32 characters."

(( ${#VLLM_API_KEY} >= 32 )) ||
    die "VLLM_API_KEY must contain at least 32 characters."

[[ "$GATEWAY_API_KEY" != "$VLLM_API_KEY" ]] ||
    die "GATEWAY_API_KEY and VLLM_API_KEY must be different."

[[ -x /opt/app/.venv/bin/uvicorn ]] ||
    die "Gateway virtual environment is missing its uvicorn executable."

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

umask 027

printf 'Starting secure-llm-deploy gateway (mode=%s, port=%s).\n' \
    "$SERVE_MODE" "$GATEWAY_PORT"

# Uvicorn owns the gateway process. The gateway lifespan starts and stops
# its managed vLLM child process.
exec /opt/app/.venv/bin/uvicorn \
    secure_llm_deploy.main:app \
    --host "$GATEWAY_HOST" \
    --port "$GATEWAY_PORT" \
    --workers 1 \
    --no-access-log \
    --log-level info