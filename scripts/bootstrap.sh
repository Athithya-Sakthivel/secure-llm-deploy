

sudo apt install gh 

docker buildx build --platform linux/amd64 --load \
  --build-arg VLLM_BASE_IMAGE='vllm/vllm-openai:v0.31.0-cu129@sha256:<full-digest>' \
  --build-arg MODEL_ID="$MODEL_ID" \
  --build-arg MODEL_REVISION="$MODEL_REVISION" \
  --tag qwen-vllm:0.31.0 .
