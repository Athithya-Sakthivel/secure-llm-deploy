# Secure LLM Deploy — production image

Secure LLM Deploy packages a local vLLM inference server behind an authenticated
FastAPI gateway. The image contains pinned model artifacts, a CPU ONNX prompt
guard, request/response safety filters, PII redaction, rate limiting, auditing,
and two inference modes.

## Architecture

The container exposes only the gateway on port 8000.

- **Gateway:** authenticates callers and applies request limits, guard checks,
  PII redaction, output filtering, and audit logging.
- **vLLM:** launched and managed by `VLLMProcess`, listening only on
  `127.0.0.1:8001`.
- **Target model:** pinned `Qwen/Qwen3-4B`, served using FP16.
- **Speculator:** pinned `AngelSlim/Qwen3-4B_eagle3`, enabled only in
  `SERVE_MODE=speculative` using vLLM's `eagle3` speculative-decoding method.
- **Prompt guard:** pinned `vllm-sr/Vela-1.0-Encoder-307M-Guard`, executed by
  ONNX Runtime on CPU.

The image's default is `SERVE_MODE=baseline`. The same image supports
`SERVE_MODE=speculative` at runtime; it does not need to be rebuilt to change
between those modes.

The committed `image/model.lock` is the source of truth for the target,
EAGLE-3 speculator, and guard repository revisions. Image builds must not
resolve model revisions from moving branches.

## Build the image locally

Build from the **repository root**, not from `image/`:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  --file image/Dockerfile \
  --tag secure-llm-deploy:local \
  .
```

The public model repositories do not require a Hugging Face token. If a reviewed
build needs a token for gated/private model access, pass it using BuildKit's
secret mount rather than a Docker `ARG` or an image environment variable:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  --file image/Dockerfile \
  --secret id=hf_token,src="$HOME/.cache/huggingface/token" \
  --tag secure-llm-deploy:local \
  .
```

Keep any token source file outside Git and protect it with filesystem
permissions. The secret mount is available only during the model-download
build step.

The image build downloads models into `/opt/models/{target,eagle,guard}`,
validates their manifests and required files, and makes the artifacts
read-only. It then installs the gateway using the root `pyproject.toml` and
`uv.lock`. A lockfile change must be intentional and reviewed.

## Runtime configuration

The container requires these two environment variables:

| Variable | Requirement |
|---|---|
| `GATEWAY_API_KEY` | Client-facing Bearer token, at least 32 characters |
| `VLLM_API_KEY` | Internal vLLM Bearer token, at least 32 characters, distinct from the gateway key |

Use separately generated secrets. Do not reuse the GHCR publishing token or
the Hugging Face token as an API key. Runtime validation rejects missing,
short, or identical API keys.

Important configuration variables:

| Variable | Default | Purpose |
|---|---|---|
| `SERVE_MODE` | `baseline` | `baseline` or `speculative` |
| `GATEWAY_HOST` | `0.0.0.0` | Gateway bind address |
| `GATEWAY_PORT` | `8000` | Public gateway port |
| `VLLM_HOST` | `127.0.0.1` | Internal server bind address; other values are rejected |
| `VLLM_PORT` | `8001` | Internal server port |
| `VLLM_MAX_MODEL_LEN` | `4096` | vLLM model context limit |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.80` | vLLM GPU-memory allocation fraction |
| `VLLM_NUM_SPECULATIVE_TOKENS` | `2` | EAGLE-3 speculative token count |
| `VLLM_MAX_NUM_SEQS` | `2` | Maximum concurrent vLLM sequences |

These values are initial settings for testing a 16 GB T4 GPU, not a guarantee
of memory fit or throughput. Increasing context length, concurrent sequences,
or speculative tokens changes memory pressure and latency. Tune one variable
at a time using representative requests.

The gateway validates the input/output budget against `MAX_MODEL_LEN`.
A request must fit the gateway's input and output limits as well as vLLM's
actual model context limit.

The EAGLE-3 speculator is optional in baseline mode but required in speculative
mode. Both modes require the target and guard artifacts. The EAGLE checkpoint
does not need a separate draft-model tokenizer.

## Local smoke tests

Run the smoke script on a machine with Docker, NVIDIA Container Toolkit, a
working NVIDIA driver, and enough GPU memory for the pinned target and
speculator.

Run baseline mode:

```bash
SERVE_MODE=baseline ./scripts/smoke-tests.sh
```

Run EAGLE-3 mode:

```bash
SERVE_MODE=speculative ./scripts/smoke-tests.sh
```

Run both modes sequentially:

```bash
SERVE_MODE=both ./scripts/smoke-tests.sh
```

The script checks:

1. The container becomes ready.
2. Liveness works.
3. `/v1/models` rejects unauthenticated requests.
4. Authenticated model listing works.
5. Non-streaming completions return valid output.
6. Streaming completions produce the terminal SSE `[DONE]` event.
7. A synthetic prompt-injection request is blocked.

By default, the script binds to `127.0.0.1` on an automatically assigned host
port, creates temporary API keys, and removes the test container at exit.
It does not send the synthetic test prompt to an external service.

Useful overrides:

```bash
IMAGE=secure-llm-deploy:local \
SMOKE_STARTUP_TIMEOUT=1200 \
SERVE_MODE=both \
./scripts/smoke-tests.sh
```

`SMOKE_TEST_GUARD=0` skips the known-attack smoke test; use that only for
diagnostics and do not treat the run as a complete security smoke test.
`SMOKE_KEEP_CONTAINER=1` leaves successful test containers running. When
keeping containers for both modes, use automatic host port allocation rather
than one fixed `SMOKE_PORT`.

The smoke tests exercise actual inference and therefore require a working GPU.
Passing unit tests or building the image does not prove that model startup
succeeds on a particular driver/GPU combination.

## Release pipeline and GHCR

The release workflow is `.github/workflows/release.yaml`. When building this
Dockerfile, its `docker/build-push-action` step must specify:

```yaml
with:
  context: .
  file: image/Dockerfile
```

Keep the action pinned to its reviewed full commit SHA. The workflow must
continue to build for `linux/amd64`, push to
`ghcr.io/athithya-sakthivel/secure-llm-deploy`, scan the exact image digest,
enforce the configured Trivy HIGH/CRITICAL vulnerability gate, generate an
SBOM and provenance, sign with keyless Cosign, and verify the resulting
signature. Do not weaken the security gate simply to make a build pass.

The current workflow uses these repository secrets:

- `GHCR_USER_NAME`: the lowercase GHCR publishing namespace.
- `GHCR_PAT`: a classic GitHub token with the minimum required `write:packages`
  permission for this workflow's registry authentication.

These are registry-publishing credentials, not runtime inference keys. Do not
place any of these secrets in `image/model.lock`, the Dockerfile, or an image
build argument. The model lock contains repository IDs and immutable commit
SHAs only.

Deploy the exact digest reported by a successful release, rather than relying
only on the mutable `latest` tag. Verify the signature and provenance before
deploying it.

## Operational security

- Expose only gateway port 8000. Do not publish container port 8001.
- Keep vLLM bound to loopback and do not run the inference process directly as
  the public API server.
- Use TLS and suitable ingress/network controls for remote clients.
- Keep API keys in a runtime secret manager or protected environment; do not
  commit them or include them in command-line arguments.
- Keep model artifacts local and offline at runtime. Changes to the pinned
  models, vLLM base image, or dependencies are supply-chain changes and should
  be reviewed.
- Use readiness as a dependency-aware signal. Liveness alone does not prove
  that the guard and model are ready to serve requests.
- Update pinned image/action dependencies and rerun security scans regularly.

## Upstream references

- [vLLM Docker deployment](https://docs.vllm.ai/en/v0.31.0/deployment/docker/)
- [vLLM EAGLE speculative decoding](https://docs.vllm.ai/en/v0.31.0/features/speculative_decoding/eagle/)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [GitHub artifact attestations](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations)
- [Cosign container signing](https://docs.sigstore.dev/cosign/signing/signing_with_containers/)
- [Trivy GitHub Action](https://github.com/aquasecurity/trivy-action)