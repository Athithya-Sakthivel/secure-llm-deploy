# syntax=docker/dockerfile:1.7.1@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Keep the vLLM version and digest pinned together in a reviewed update.
FROM vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb

USER 0:0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/app

ENV SERVE_MODE=baseline \
    MODEL_ROOT=/opt/models \
    MODEL_LOCK_FILE=/etc/vllm/models.lock \
    TARGET_MODEL_PATH=/opt/models/target \
    TARGET_TOKENIZER_PATH=/opt/models/target/tokenizer.json \
    EAGLE_MODEL_PATH=/opt/models/eagle \
    GUARD_MODEL_PATH=/opt/models/guard/model.QUInt8.quant.onnx \
    GUARD_TOKENIZER_PATH=/opt/models/guard/tokenizer.json \
    GUARD_CONFIG_PATH=/opt/models/guard/config.json \
    GUARD_THRESHOLD=0.5 \
    GUARD_MAX_INPUT_TOKENS=6144 \
    GUARD_WINDOW_TOKENS=512 \
    GUARD_INTRA_OP_THREADS=2 \
    GUARD_FAILURE_POLICY=closed \
    GUARD_OVERLONG_POLICY=reject \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8000 \
    VLLM_HOST=127.0.0.1 \
    VLLM_PORT=8001 \
    VLLM_EXECUTABLE=vllm \
    VLLM_MAX_MODEL_LEN=4096 \
    VLLM_GPU_MEMORY_UTILIZATION=0.80 \
    VLLM_NUM_SPECULATIVE_TOKENS=2 \
    VLLM_MAX_NUM_SEQS=2 \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_CACHE_ROOT=/tmp/vllm-cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/app/.venv \
    UV_CACHE_DIR=/root/.cache/uv \
    PYTHONPATH=/opt/app/image/src \
    PATH=/opt/app/.venv/bin:${PATH}

LABEL org.opencontainers.image.source="https://github.com/Athithya-Sakthivel/secure-llm-deploy" \
      org.opencontainers.image.description="Secure LLM gateway with pinned Qwen3-8B AWQ, EAGLE 3.1 and Prompt Guard 2 ONNX artifacts"

RUN install -d -m 0755 \
        /etc/vllm \
        /opt/app/image \
        /opt/uv/python \
        /opt/models/target \
        /opt/models/eagle \
        /opt/models/guard \
        /tmp/huggingface \
        /tmp/vllm-cache \
        /tmp/torchinductor

# Resolve dependencies before copying application source. This keeps model
# downloads cacheable across ordinary gateway-source changes. The application
# itself is installed in a later layer after image/src/ has been copied.
COPY --chown=0:0 pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    set -Eeuo pipefail; \
    BASE_VLLM="$(PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin command -v vllm)"; \
    [[ -x "${BASE_VLLM}" ]] || { printf 'ERROR: base vllm executable not found\n' >&2; exit 1; }; \
    BASE_SHEBANG="$(head -n 1 "${BASE_VLLM}")"; \
    [[ "${BASE_SHEBANG}" == '#!'* ]] || { printf 'ERROR: cannot identify vLLM Python interpreter\n' >&2; exit 1; }; \
    BASE_INTERPRETER="${BASE_SHEBANG#\#!}"; \
    if [[ "${BASE_INTERPRETER}" == /usr/bin/env\ * ]]; then BASE_INTERPRETER="${BASE_INTERPRETER#/usr/bin/env }"; BASE_INTERPRETER="${BASE_INTERPRETER%% *}"; BASE_PYTHON="$(PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin command -v "${BASE_INTERPRETER}")"; else BASE_PYTHON="${BASE_INTERPRETER%% *}"; fi; \
    [[ -x "${BASE_PYTHON}" ]] || { printf 'ERROR: vLLM Python interpreter is not executable\n' >&2; exit 1; }; \
    "${BASE_PYTHON}" -m pip install --no-cache-dir --disable-pip-version-check "uv==0.13.1"; \
    uv python install 3.13; \
    uv sync --locked --no-dev --no-install-project --python 3.13; \
    /opt/app/.venv/bin/python -c 'import numpy, onnxruntime, tokenizers'

# The lock is data, never shell-sourced. download_models.py uses the vLLM
# image's Python/Hugging Face Hub; validate.py runs in the pinned gateway venv
# where NumPy, ONNX Runtime and tokenizers are installed from uv.lock.
COPY --chown=0:0 image/model.lock /etc/vllm/models.lock
COPY --chown=0:0 image/download_models.py image/validate.py /opt/app/image/

RUN chmod 0444 \
    /etc/vllm/models.lock \
    /opt/app/image/download_models.py \
    /opt/app/image/validate.py

# HF_TOKEN is optional and exists only for this BuildKit RUN. It is never baked
# into ENV, copied into the image, or passed to the runtime container.
RUN --mount=type=secret,id=hf_token,required=false \
    set -Eeuo pipefail; \
    BASE_VLLM="$(PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin command -v vllm)"; \
    [[ -x "${BASE_VLLM}" ]] || { printf 'ERROR: base vllm executable not found\n' >&2; exit 1; }; \
    BASE_SHEBANG="$(head -n 1 "${BASE_VLLM}")"; \
    [[ "${BASE_SHEBANG}" == '#!'* ]] || { printf 'ERROR: cannot identify vLLM Python interpreter\n' >&2; exit 1; }; \
    BASE_INTERPRETER="${BASE_SHEBANG#\#!}"; \
    if [[ "${BASE_INTERPRETER}" == /usr/bin/env\ * ]]; then BASE_INTERPRETER="${BASE_INTERPRETER#/usr/bin/env }"; BASE_INTERPRETER="${BASE_INTERPRETER%% *}"; BASE_PYTHON="$(PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin command -v "${BASE_INTERPRETER}")"; else BASE_PYTHON="${BASE_INTERPRETER%% *}"; fi; \
    [[ -x "${BASE_PYTHON}" ]] || { printf 'ERROR: vLLM Python interpreter is not executable\n' >&2; exit 1; }; \
    "${BASE_PYTHON}" -c 'import huggingface_hub' || { printf 'ERROR: huggingface_hub is not installed in the vLLM Python environment\n' >&2; exit 1; }; \
    if [[ -r /run/secrets/hf_token ]]; then export HF_TOKEN="$(</run/secrets/hf_token)"; fi; \
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0; \
    "${BASE_PYTHON}" /opt/app/image/download_models.py; \
    unset HF_TOKEN || true; \
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
    /opt/app/.venv/bin/python /opt/app/image/validate.py --model-dir /opt/models --lock-file /etc/vllm/models.lock; \
    find /tmp/huggingface -mindepth 1 -delete

# Copy application code only after the weight-download layer so source-only
# changes do not invalidate the multi-gigabyte model layer.
COPY --chown=0:0 image/src/ /opt/app/image/src/
COPY --chown=0:0 image/entrypoint.sh /opt/app/image/entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    set -Eeuo pipefail; \
    uv sync --locked --no-dev --no-editable --python 3.13; \
    test -x /opt/app/.venv/bin/uvicorn; \
    BASE_VLLM="$(PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin command -v vllm)"; \
    BASE_SHEBANG="$(head -n 1 "${BASE_VLLM}")"; \
    BASE_INTERPRETER="${BASE_SHEBANG#\#!}"; \
    if [[ "${BASE_INTERPRETER}" == /usr/bin/env\ * ]]; then BASE_INTERPRETER="${BASE_INTERPRETER#/usr/bin/env }"; BASE_INTERPRETER="${BASE_INTERPRETER%% *}"; BASE_PYTHON="$(PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin command -v "${BASE_INTERPRETER}")"; else BASE_PYTHON="${BASE_INTERPRETER%% *}"; fi; \
    "${BASE_PYTHON}" -m pip uninstall -y uv; \
    chmod 0555 /opt/app/image/entrypoint.sh; \
    find /opt/app/image/src -type d -exec chmod 0555 {} +; \
    find /opt/app/image/src -type f -exec chmod 0444 {} +; \
    find /opt/app/.venv -type d -exec chmod 0555 {} +; \
    find /opt/app/.venv -type f -perm /111 -exec chmod 0555 {} +; \
    find /opt/app/.venv -type f ! -perm /111 -exec chmod 0444 {} +; \
    find /opt/uv/python -type d -exec chmod 0555 {} +; \
    find /opt/uv/python -type f -perm /111 -exec chmod 0555 {} +; \
    find /opt/uv/python -type f ! -perm /111 -exec chmod 0444 {} +; \
    find /opt/models -type d -exec chmod 0555 {} +; \
    find /opt/models -type f -exec chmod 0444 {} +; \
    chmod 0444 /etc/vllm/models.lock; \
    chown -R 2000:0 /tmp/huggingface /tmp/vllm-cache /tmp/torchinductor; \
    chmod 0770 /tmp/huggingface /tmp/vllm-cache /tmp/torchinductor

# UID 2000 / GID 0 is supported by the pinned vLLM base image. Model/application
# files are read-only; only explicit temporary cache directories are writable.
USER 2000:0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD ["/opt/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

ENTRYPOINT ["/opt/app/image/entrypoint.sh"]