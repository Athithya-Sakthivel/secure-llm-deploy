#!/usr/bin/env bash
set -Eeuo pipefail

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
export PATH="$BIN_DIR:$PATH"

UV_VERSION="0.12.17"

curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

uv --version

# Persist PATH without duplicates
touch "$HOME/.bashrc"
grep -Fqx 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" ||
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

# Install GitHub CLI
if ! command -v gh >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y gh
fi

# Install Kind v0.33.0
KIND_VERSION="v0.33.0"
if ! "$BIN_DIR/kind" version 2>/dev/null | grep -Fq "$KIND_VERSION"; then
  curl -fsSL "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64" \
    -o "$BIN_DIR/kind.tmp"
  chmod +x "$BIN_DIR/kind.tmp"
  mv "$BIN_DIR/kind.tmp" "$BIN_DIR/kind"
fi

# Install kubectl v1.35.0
KUBECTL_VERSION="v1.35.0"
if ! "$BIN_DIR/kubectl" version --client -o yaml 2>/dev/null |
  grep -Fq "gitVersion: ${KUBECTL_VERSION}"; then
  curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    -o "$BIN_DIR/kubectl.tmp"
  chmod +x "$BIN_DIR/kubectl.tmp"
  mv "$BIN_DIR/kubectl.tmp" "$BIN_DIR/kubectl"
fi

# Create the cluster only if it doesn't exist
if ! "$BIN_DIR/kind" get clusters | grep -qx "secure-llm"; then
  "$BIN_DIR/kind" create cluster \
    --name secure-llm \
    --image kindest/node:v1.35.0 \
    --wait 180s
fi

# Verify installation and cluster
gh --version
"$BIN_DIR/kind" version
"$BIN_DIR/kubectl" version --client
"$BIN_DIR/kubectl" get nodes
"$BIN_DIR/kubectl" get pods -A