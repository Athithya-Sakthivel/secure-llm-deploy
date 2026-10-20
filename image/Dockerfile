# syntax=docker/dockerfile:1.7.1@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Update the tag and digest together, after reviewing the upstream release.
FROM vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb

USER 0:0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/app

# Keep these paths synchronized with image/model.lock, the build scripts, and
# secure_llm_deploy.config.Settings. Runtime model loading is offline.
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

# Resolve the interpreter behind the pinned image's `vllm` entry point once.
# The temporary wrapper lets build steps use that same interpreter without
# assuming a hard-coded Python path. It is removed before the image is sealed.
RUN set -Eeuo pipefail; \
    VLLM_BIN="$(command -v vllm || true)"; \
    [[ -n "${VLLM_BIN}" && -x "${VLLM_BIN}" ]] || { printf 'ERROR: vllm executable is missing from the base image\n' >&2; exit 1; }; \
    VLLM_SHEBANG="$(head -n 1 "${VLLM_BIN}")"; \
    [[ "${VLLM_SHEBANG}" == '#!'* ]] || { printf 'ERROR: cannot identify the Python interpreter used by vllm\n' >&2; exit 1; }; \
    VLLM_INTERPRETER="${VLLM_SHEBANG#\#!}"; \
    if [[ "${VLLM_INTERPRETER}" == /usr/bin/env\ * ]]; then \
      VLLM_INTERPRETER="${VLLM_INTERPRETER#/usr/bin/env }"; \
      VLLM_INTERPRETER="${VLLM_INTERPRETER%% *}"; \
      VLLM_PYTHON="$(command -v "${VLLM_INTERPRETER}" || true)"; \
    else \
      VLLM_PYTHON="${VLLM_INTERPRETER%% *}"; \
    fi; \
    [[ -x "${VLLM_PYTHON}" ]] || { printf 'ERROR: vLLM Python interpreter is not executable: %s\n' "${VLLM_PYTHON}" >&2; exit 1; }; \
    printf '#!/bin/sh\nexec "%s" "$@"\n' "${VLLM_PYTHON}" > /usr/local/bin/vllm-python; \
    chmod 0555 /usr/local/bin/vllm-python

# Copy dependency metadata first. The lock must be up-to-date; --locked is
# intentional so builds cannot silently resolve a different dependency graph.
COPY --chown=0:0 pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -Eeuo pipefail; \
    /usr/local/bin/vllm-python -m pip install --no-cache-dir --disable-pip-version-check 'uv==0.13.1'; \
    uv --version | grep -Fx 'uv 0.13.1'; \
    uv python install 3.13; \
    uv sync --locked --no-dev --no-install-project --python 3.13; \
    /opt/app/.venv/bin/python -c 'import numpy, onnxruntime, tokenizers'

# The lock is parsed as KEY=VALUE data, never sourced as shell code. The model
# download and validation scripts are part of the build context and repository.
COPY --chown=0:0 image/model.lock /etc/vllm/models.lock
COPY --chown=0:0 image/download_models.py image/validate.py /opt/app/image/
RUN chmod 0444 /etc/vllm/models.lock /opt/app/image/download_models.py /opt/app/image/validate.py

# Download model snapshots at immutable commit revisions. BuildKit mounts an
# optional token only for this instruction; no Hub token is saved in ENV or a
# later layer. Public repositories should build without a token.
RUN --mount=type=secret,id=hf_token,required=false \
    set -Eeuo pipefail; \
    /usr/local/bin/vllm-python -c 'import huggingface_hub' || { printf 'ERROR: huggingface_hub is unavailable in the vLLM Python environment\n' >&2; exit 1; }; \
    if [[ -r /run/secrets/hf_token ]]; then export HF_TOKEN="$(</run/secrets/hf_token)"; fi; \
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0; \
    /usr/local/bin/vllm-python /opt/app/image/download_models.py; \
    unset HF_TOKEN || true; \
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
    /opt/app/.venv/bin/python /opt/app/image/validate.py --model-dir /opt/models --lock-file /etc/vllm/models.lock; \
    find /tmp/huggingface -mindepth 1 -delete

# Copy app code after the model layer so ordinary source changes do not force
# multi-gigabyte model downloads to repeat.
COPY --chown=0:0 image/src/ /opt/app/image/src/
COPY --chown=0:0 image/entrypoint.sh /opt/app/image/entrypoint.sh

# Install the package from the committed lock, remove the temporary package
# manager from the base interpreter, and make application/model assets read-only.
# Preserve executable mode on venv launchers and the installed Python runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -Eeuo pipefail; \
    uv sync --locked --no-dev --no-editable --python 3.13; \
    test -x /opt/app/.venv/bin/uvicorn; \
    /usr/local/bin/vllm-python -m pip uninstall -y uv; \
    rm -f /usr/local/bin/vllm-python; \
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

# One gateway worker owns and supervises one vLLM child process. Only port 8000
# is published; vLLM remains bound to loopback at 127.0.0.1:8001.
USER 2000:0
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD ["/opt/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

ENTRYPOINT ["/opt/app/image/entrypoint.sh"]