# syntax=docker/dockerfile:1.7
# =============================================================================
# Stage 1: Builder
# Compiles only your application-specific dependencies.
# Uses the full CUDA "devel" image with build tools.
# =============================================================================
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS builder

# Pin Python to a specific patch version for reproducibility.
ARG PYTHON_VERSION=3.11.17

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip wheel

# Build wheels for your application-specific dependencies only.
# vLLM itself will be installed from a pre-built wheel in the runtime stage.
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# Uses the minimal CUDA "runtime" image. No compilers, no build tools.
# =============================================================================
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04 AS runtime

# Pin all versions for full reproducibility.
ARG PYTHON_VERSION=3.11.17
ARG VLLM_VERSION=0.31.0

# Create a non-root user with a high UID for Kubernetes compatibility.
# This is a critical security control: if the process is compromised,
# it cannot escalate to root on the host.
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid appgroup --create-home --shell /bin/bash appuser

# Install only the minimal runtime dependencies.
# tini is used as an init system to handle signals properly (SIGTERM for
# graceful shutdown in Kubernetes).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Copy application wheels from the builder stage.
COPY --from=builder /wheels /wheels

# Install application dependencies from local wheels (no network needed).
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Install vLLM from PyPI (pre-built wheel, CUDA 12.9 variant).
# The pip resolver will select the correct wheel for the CUDA 12.9 base image.
RUN pip install --no-cache-dir vllm==${VLLM_VERSION}

# Copy application code with correct ownership.
WORKDIR /app
COPY --chown=appuser:appgroup . .

# Switch to the non-root user.
USER appuser

# Environment variables for vLLM.
ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000
EXPOSE 8000

# Use tini as the entrypoint for proper signal handling.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: launch the vLLM OpenAI-compatible server.
# The model can be overridden at runtime.
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen3.8-2B-Distill-AWQ", \
     "--host", "0.0.0.0", \
     "--port", "8000"]