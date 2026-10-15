# syntax=docker/dockerfile:1.7.1@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
#
# Production vLLM runtime.
# Base image and model revision are both pinned. Updating either pin is an
# intentional reviewed change; builds do not resolve model HEAD dynamically.

FROM vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb AS runtime

USER 0:0

ENV HOME=/home/vllm \
    XDG_CACHE_HOME=/home/vllm/.cache \
    HF_HOME=/home/vllm/.cache/huggingface \
    HF_HUB_CACHE=/home/vllm/.cache/huggingface/hub \
    VLLM_CACHE_ROOT=/home/vllm/.cache/vllm \
    TRITON_CACHE_DIR=/home/vllm/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/home/vllm/.cache/torchinductor \
    VLLM_LOCK_FILE=/etc/vllm/model.lock \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_LOGGING_LEVEL=INFO \
    VLLM_HOST=0.0.0.0 \
    VLLM_PORT=8000

COPY --chown=0:0 --chmod=0444 model.lock /etc/vllm/model.lock
COPY --chown=0:0 --chmod=0555 validate.py /usr/local/bin/vllm-validate
COPY --chown=0:0 --chmod=0555 entrypoint.sh /usr/local/bin/vllm-container-entrypoint

RUN python3 /usr/local/bin/vllm-validate \
 && install -d -o 2000 -g 0 -m 0750 /home/vllm /home/vllm/.cache \
 && install -d -o 2000 -g 0 -m 0770 \
      /home/vllm/.cache/huggingface \
      /home/vllm/.cache/huggingface/hub \
      /home/vllm/.cache/vllm \
      /home/vllm/.cache/triton \
      /home/vllm/.cache/torchinductor

LABEL org.opencontainers.image.title="secure-llm-deploy vLLM runtime" \
      org.opencontainers.image.version="0.31.0" \
      org.opencontainers.image.description="vLLM CUDA 12.9 runtime with a pinned model revision" \
      org.opencontainers.image.source="https://github.com/athithya-sakthivel/secure-llm-deploy"

WORKDIR /home/vllm
USER 2000:0
EXPOSE 8000/tcp
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD ["python3", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('VLLM_PORT', '8000') + '/health', timeout=4).close()"]

ENTRYPOINT ["/usr/local/bin/vllm-container-entrypoint"]