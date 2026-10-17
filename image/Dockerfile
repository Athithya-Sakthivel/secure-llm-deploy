# syntax=docker/dockerfile:1.7.1@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Pinned CUDA 12.9 vLLM runtime. Update the version and digest together
# through a reviewed dependency/supply-chain change.
FROM vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb

USER 0:0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/app

ENV SERVE_MODE=baseline \
    MODEL_ROOT=/opt/models \
    TARGET_MODEL_PATH=/opt/models/target \
    TARGET_TOKENIZER_PATH=/opt/models/target/tokenizer.json \
    EAGLE_MODEL_PATH=/opt/models/eagle \
    GUARD_MODEL_PATH=/opt/models/guard/onnx/model.onnx \
    GUARD_TOKENIZER_PATH=/opt/models/guard/tokenizer.json \
    GUARD_LABEL_MAPPING_PATH=/opt/models/guard/label_mapping.json \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8000 \
    VLLM_HOST=127.0.0.1 \
    VLLM_PORT=8001 \
    VLLM_EXECUTABLE=vllm \
    VLLM_MAX_MODEL_LEN=4096 \
    VLLM_GPU_MEMORY_UTILIZATION=0.80 \
    VLLM_NUM_SPECULATIVE_TOKENS=2 \
    VLLM_MAX_NUM_SEQS=2 \
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
    PYTHONPATH=/opt/app/image/src \
    PATH=/opt/app/.venv/bin:${PATH}

LABEL org.opencontainers.image.source="https://github.com/Athithya-Sakthivel/secure-llm-deploy" \
      org.opencontainers.image.description="Secure LLM gateway with pinned Qwen3-4B, EAGLE-3, and Vela guard artifacts"

RUN install -d -m 0755 \
        /etc/vllm \
        /opt/app/image \
        /opt/uv/python \
        /opt/models/target \
        /opt/models/eagle \
        /opt/models/guard/onnx \
        /tmp/huggingface \
        /tmp/vllm-cache \
        /tmp/torchinductor

# ---------------------------------------------------------------------------
# Download the pinned model artifacts. This layer is deliberately independent
# of application source changes so Docker can reuse it between code releases.
# ---------------------------------------------------------------------------

COPY --chown=0:0 image/model.lock /etc/vllm/models.lock
COPY --chown=0:0 image/download_models.py image/validate.py /opt/app/image/

RUN chmod 0444 /etc/vllm/models.lock \
    && chmod 0444 /opt/app/image/download_models.py /opt/app/image/validate.py

# Public model repositories do not need HF_TOKEN. If an authorized build
# requires a token, pass it with BuildKit --secret id=hf_token,...
# The secret is mounted temporarily and is never copied into the image.
RUN --mount=type=secret,id=hf_token,required=false \
    set -Eeuo pipefail; \
    if [[ -r /run/secrets/hf_token ]]; then \
        export HF_TOKEN="$(</run/secrets/hf_token)"; \
    fi; \
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0; \
    python3 /opt/app/image/download_models.py; \
    python3 /opt/app/image/validate.py \
        --model-dir /opt/models \
        --lock-file /etc/vllm/models.lock; \
    unset HF_TOKEN || true; \
    find /tmp/huggingface -mindepth 1 -delete

# ---------------------------------------------------------------------------
# Install the gateway from the repository's locked Python dependency graph.
# The gateway has its own Python 3.13 environment; the pinned vLLM image keeps
# its own Python environment for the inference subprocess.
# ---------------------------------------------------------------------------

COPY --chown=0:0 pyproject.toml uv.lock README.md ./
COPY --chown=0:0 image/src/ image/src/
COPY --chown=0:0 image/entrypoint.sh image/entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    set -Eeuo pipefail; \
    python3 -m pip install \
        --no-cache-dir \
        --disable-pip-version-check \
        "uv==0.13.1"; \
    uv python install 3.13; \
    uv sync --locked --no-dev --no-editable --python 3.13; \
    test -x /opt/app/.venv/bin/uvicorn; \
    python3 -m pip uninstall -y uv; \
    chmod 0555 /opt/app/image/entrypoint.sh; \
    find /opt/app/image/src -type d -exec chmod 0555 {} +; \
    find /opt/app/image/src -type f -exec chmod 0444 {} +; \
    find /opt/app/.venv -type d -exec chmod 0555 {} +; \
    find /opt/app/.venv -type f -exec chmod 0444 {} +; \
    find /opt/models -type d -exec chmod 0555 {} +; \
    find /opt/models -type f -exec chmod 0444 {} +; \
    chmod 0444 /etc/vllm/models.lock; \
    chown -R 2000:0 /tmp/huggingface /tmp/vllm-cache /tmp/torchinductor; \
    chmod 0770 /tmp/huggingface /tmp/vllm-cache /tmp/torchinductor

# All model files and the application are read-only to the runtime user.
# Writable model caches are confined to the temporary filesystem.
USER 2000:0

EXPOSE 8000

# Readiness includes actual guard and vLLM readiness. The startup grace period
# accommodates model loading on GPUs with limited memory or slower storage.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD ["/opt/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

ENTRYPOINT ["/opt/app/image/entrypoint.sh"]