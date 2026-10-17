#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_DIR="$SCRIPT_DIR/model"
MODEL_REPO="Athithya117/Llama-Prompt-Guard-2-22M-onnx"
DOWNLOAD_MARKER="$MODEL_DIR/.download-complete"

echo "==> Evaluation directory: $SCRIPT_DIR"

# Create and activate the isolated Python environment.
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install evaluation dependencies.
echo "==> Installing Python dependencies..."
python -m pip install -q -r "$SCRIPT_DIR/requirements.txt"
python -m pip install -q "huggingface_hub>=0.30"

# Verify that the Hugging Face CLI is available.
if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: Hugging Face CLI ('hf') is unavailable." >&2
    exit 1
fi

# Download the complete model repository when needed.
mkdir -p "$MODEL_DIR"

if [[ ! -f "$DOWNLOAD_MARKER" ]]; then
    echo "==> Downloading full model repository: $MODEL_REPO"
    hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
else
    echo "==> Model repository download already completed."
fi

# Support either quantized filename.
if [[ -s "$MODEL_DIR/model.QUInt8.quant.onnx" ]]; then
    MODEL_FILE="$MODEL_DIR/model.QUInt8.quant.onnx"
elif [[ -s "$MODEL_DIR/model.quant.onnx" ]]; then
    MODEL_FILE="$MODEL_DIR/model.quant.onnx"
else
    echo "ERROR: quantized ONNX model not found in $MODEL_DIR" >&2
    find "$MODEL_DIR" -maxdepth 2 -type f -printf '%P\n' >&2
    exit 1
fi

# Validate the files required by the evaluator.
for file in tokenizer.json config.json; do
    if [[ ! -s "$MODEL_DIR/$file" ]]; then
        echo "ERROR: missing required model file: $MODEL_DIR/$file" >&2
        exit 1
    fi
done

# Mark complete only after the required files have been verified.
touch "$DOWNLOAD_MARKER"

echo "==> Model ready: $MODEL_FILE"

# Diagnostic checks without importing Transformers or PyTorch.
python - <<'PY'
import onnxruntime
import tokenizers
import datasets

print("onnxruntime:", onnxruntime.__version__)
print("tokenizers:", tokenizers.__version__)
print("datasets:", datasets.__version__)
print("Available ORT providers:", onnxruntime.get_available_providers())
PY

echo "==> Bootstrap complete."
echo "    Next step: bash \"$SCRIPT_DIR/run.sh\""