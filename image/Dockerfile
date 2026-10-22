# syntax=docker/dockerfile:1.7.1@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Keep the tag and digest synchronized after reviewing the upstream release.
FROM vllm/vllm-openai:v0.32.0-cu129@sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f

USER 0:0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/app

# Patch the OS package with a published Ubuntu fix. Do this in the same RUN as
# apt-get update so the package index cannot be reused from a stale layer.
# Fail closed if the configured Ubuntu repositories do not supply the minimum
# fixed version identified by the Trivy findings.
RUN set -Eeuo pipefail; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade linux-libc-dev; \
    LINUX_LIBC_DEV_VERSION="$(dpkg-query -W -f='${Version}' linux-libc-dev)"; \
    printf 'linux-libc-dev installed version: %s\n' "${LINUX_LIBC_DEV_VERSION}"; \
    dpkg --compare-versions "${LINUX_LIBC_DEV_VERSION}" ge '6.8.0-147.147' || { \
      printf 'ERROR: linux-libc-dev %s is older than required fix 6.8.0-147.147\n' "${LINUX_LIBC_DEV_VERSION}" >&2; \
      exit 1; \
    }; \
    rm -rf /var/lib/apt/lists/*

# Reuse the Python runtime shipped with vLLM. uv is also already present in
# this digest-pinned base image, so do not pip-install/replace either one.
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
    UV_PROJECT_ENVIRONMENT=/opt/app/.venv \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/opt/app/image/src \
    PATH=/opt/app/.venv/bin:${PATH}

LABEL org.opencontainers.image.source="https://github.com/Athithya-Sakthivel/secure-llm-deploy" \
      org.opencontainers.image.description="Secure LLM gateway with pinned Qwen3-8B AWQ, EAGLE 3.1 and Prompt Guard 2 ONNX artifacts"

RUN install -d -m 0755 \
      /etc/vllm \
      /opt/app/image \
      /opt/models/target \
      /opt/models/eagle \
      /opt/models/guard \
      /tmp/huggingface \
      /tmp/vllm-cache \
      /tmp/torchinductor

# Resolve both (a) the interpreter used by the base image's vllm entry point
# and (b) its underlying Python executable. Keep the first path intact because
# it may point into a vLLM virtual environment and is needed for Hub downloads.
# The gateway venv is based on the same Python version but stays package-isolated.
RUN set -Eeuo pipefail; \
    VLLM_BIN="$(command -v vllm || true)"; \
    [[ -n "${VLLM_BIN}" && -x "${VLLM_BIN}" ]] || { printf 'ERROR: vllm executable is missing from the base image\n' >&2; exit 1; }; \
    VLLM_SHEBANG="$(head -n 1 "${VLLM_BIN}")"; \
    [[ "${VLLM_SHEBANG}" == '#!'* ]] || { printf 'ERROR: cannot identify Python interpreter used by vllm\n' >&2; exit 1; }; \
    if [[ "${VLLM_SHEBANG}" == '#!/usr/bin/env '* ]]; then \
      VLLM_INTERPRETER="${VLLM_SHEBANG#\#!/usr/bin/env }"; \
      if [[ "${VLLM_INTERPRETER}" == -S\ * ]]; then VLLM_INTERPRETER="${VLLM_INTERPRETER#-S }"; fi; \
      VLLM_INTERPRETER="${VLLM_INTERPRETER%% *}"; \
      VLLM_PYTHON="$(command -v "${VLLM_INTERPRETER}" || true)"; \
    else \
      VLLM_INTERPRETER="${VLLM_SHEBANG#\#!}"; \
      VLLM_PYTHON="${VLLM_INTERPRETER%% *}"; \
    fi; \
    [[ -n "${VLLM_PYTHON}" && -x "${VLLM_PYTHON}" ]] || { printf 'ERROR: vLLM Python interpreter is not executable: %s\n' "${VLLM_PYTHON}" >&2; exit 1; }; \
    APP_PYTHON="$("${VLLM_PYTHON}" -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))')"; \
    [[ -n "${APP_PYTHON}" && -x "${APP_PYTHON}" ]] || { printf 'ERROR: base Python executable is not available: %s\n' "${APP_PYTHON}" >&2; exit 1; }; \
    VLLM_VERSION="$("${VLLM_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"; \
    APP_VERSION="$("${APP_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"; \
    [[ "${VLLM_VERSION}" == "${APP_VERSION}" ]] || { printf 'ERROR: vLLM Python %s differs from base Python %s\n' "${VLLM_VERSION}" "${APP_VERSION}" >&2; exit 1; }; \
    "${VLLM_PYTHON}" -c 'import sys, torch, vllm; from importlib.metadata import version; print("vLLM interpreter:", sys.executable); print("Python:", sys.version.split()[0]); print("PyTorch:", torch.__version__); print("vLLM:", version("vllm"))'; \
    "${APP_PYTHON}" -c 'import sys; assert (3, 12) <= sys.version_info[:2] < (3, 15), f"Unsupported Python version: {sys.version}"; print("Gateway venv base Python:", sys.version.split()[0])'; \
    printf '%s\n' "${VLLM_PYTHON}" > /etc/vllm/vllm-python.path; \
    printf '%s\n' "${APP_PYTHON}" > /etc/vllm/python-base.path; \
    chmod 0444 /etc/vllm/vllm-python.path /etc/vllm/python-base.path; \
    uv --version

# LMCache is an optional vLLM integration and this deployment does not configure
# its connector or multiprocess cache server. Remove it instead of replacing a
# native CUDA/PyTorch extension with a wheel whose ABI is not validated against
# this exact pinned vLLM image. Also fail if package/module files remain.
RUN set -Eeuo pipefail; \
    VLLM_PYTHON="$(cat /etc/vllm/vllm-python.path)"; \
    "${VLLM_PYTHON}" -m pip uninstall --yes lmcache; \
    if "${VLLM_PYTHON}" -m pip show lmcache >/dev/null 2>&1; then \
      printf 'ERROR: lmcache distribution is still installed\n' >&2; \
      exit 1; \
    fi; \
    "${VLLM_PYTHON}" -c 'import importlib.util; assert importlib.util.find_spec("lmcache") is None, "lmcache module remains installed"'; \
    "${VLLM_PYTHON}" -c 'import torch, vllm; print("vLLM smoke test passed after removing optional lmcache.", "torch=" + torch.__version__)'

# Copy project metadata before dependencies so source changes do not invalidate
# this layer. Build from the repository root, because these files live there.
COPY --chown=0:0 pyproject.toml uv.lock README.md ./

# Dependencies first for better layer caching. --locked deliberately fails if
# pyproject.toml and uv.lock disagree; it never silently resolves new versions.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -Eeuo pipefail; \
    APP_PYTHON="$(cat /etc/vllm/python-base.path)"; \
    uv sync --locked --no-dev --no-install-project --python "${APP_PYTHON}"; \
    /opt/app/.venv/bin/python -c 'import numpy, onnxruntime, tokenizers'

# All files below are relative to the repository-root build context.
COPY --chown=0:0 image/model.lock /etc/vllm/models.lock
COPY --chown=0:0 image/download_models.py image/validate.py /opt/app/image/
RUN chmod 0444 /etc/vllm/models.lock /opt/app/image/download_models.py /opt/app/image/validate.py

# Model downloads happen at build time against immutable commit revisions.
# An optional Hugging Face token is mounted as a BuildKit secret and is not
# persisted in ENV or any image layer. Public model repositories need no token.
RUN --mount=type=secret,id=hf_token,required=false \
    set -Eeuo pipefail; \
    VLLM_PYTHON="$(cat /etc/vllm/vllm-python.path)"; \
    "${VLLM_PYTHON}" -c 'import huggingface_hub'; \
    if [[ -r /run/secrets/hf_token ]]; then export HF_TOKEN="$(</run/secrets/hf_token)"; fi; \
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0; \
    "${VLLM_PYTHON}" /opt/app/image/download_models.py; \
    unset HF_TOKEN || true; \
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
    /opt/app/.venv/bin/python /opt/app/image/validate.py --model-dir /opt/models --lock-file /etc/vllm/models.lock; \
    find /tmp/huggingface -mindepth 1 -delete

# Copy app code only after model downloads so ordinary source edits do not
# invalidate the multi-gigabyte model-download layer.
COPY --chown=0:0 image/src/ /opt/app/image/src/
COPY --chown=0:0 image/entrypoint.sh /opt/app/image/entrypoint.sh

# Install the gateway as a non-editable package; keep its dependencies locked.
# Smoke-import the gateway stack before sealing the image.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -Eeuo pipefail; \
    APP_PYTHON="$(cat /etc/vllm/python-base.path)"; \
    uv sync --locked --no-dev --no-editable --python "${APP_PYTHON}"; \
    test -x /opt/app/.venv/bin/uvicorn; \
    /opt/app/.venv/bin/python -c 'import fastapi, httpx, numpy, onnxruntime, pydantic, pydantic_settings, tokenizers, uvicorn; import secure_llm_deploy.main; print("Gateway import smoke test passed.")'; \
    chmod 0555 /opt/app/image/entrypoint.sh; \
    find /opt/app/image/src -type d -exec chmod 0555 {} +; \
    find /opt/app/image/src -type f -exec chmod 0444 {} +; \
    find /opt/app/.venv -type d -exec chmod 0555 {} +; \
    find /opt/app/.venv -type f -perm /111 -exec chmod 0555 {} +; \
    find /opt/app/.venv -type f ! -perm /111 -exec chmod 0444 {} +; \
    find /opt/models -type d -exec chmod 0555 {} +; \
    find /opt/models -type f -exec chmod 0444 {} +; \
    chmod 0444 /etc/vllm/models.lock; \
    rm -f /etc/vllm/vllm-python.path /etc/vllm/python-base.path; \
    chown -R 2000:0 /tmp/huggingface /tmp/vllm-cache /tmp/torchinductor; \
    chmod 0770 /tmp/huggingface /tmp/vllm-cache /tmp/torchinductor

# The pinned vLLM image supports UID 2000 in group 0. Gateway process owns and
# supervises one vLLM child; publish only port 8000, with vLLM on loopback.
USER 2000:0
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD ["/opt/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

ENTRYPOINT ["/opt/app/image/entrypoint.sh"]
