#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_DIR="$SCRIPT_DIR/model"

# Defaults can be overridden using environment variables.
CONFIGS="${EVAL_CONFIGS:-notinject,xstest,slabs_test}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: virtual environment not found. Run bootstrap.sh first." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Select the available quantized model filename.
if [[ -s "$MODEL_DIR/model.QUInt8.quant.onnx" ]]; then
    MODEL_PATH="$MODEL_DIR/model.QUInt8.quant.onnx"
elif [[ -s "$MODEL_DIR/model.quant.onnx" ]]; then
    MODEL_PATH="$MODEL_DIR/model.quant.onnx"
else
    echo "ERROR: quantized ONNX model not found. Run bootstrap.sh first." >&2
    exit 1
fi

for file in tokenizer.json config.json; do
    if [[ ! -s "$MODEL_DIR/$file" ]]; then
        echo "ERROR: required file missing: $MODEL_DIR/$file" >&2
        exit 1
    fi
done

echo "==> Starting Prompt Guard evaluation"
echo "    Model:       $MODEL_PATH"
echo "    Tokenizer:   $MODEL_DIR"
echo "    Configs:     $CONFIGS"
echo "    Max samples: $MAX_SAMPLES per config"
echo

exec python "$SCRIPT_DIR/eval.py" \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_DIR" \
    --configs "$CONFIGS" \
    --max-samples "$MAX_SAMPLES" \
    "$@"

