# Secure LLM Deploy

> ⚠️ **EXPERIMENTAL / RESEARCH PREVIEW**
> This project is an experimental research prototype. It is **not production-ready, not yet horizontally scalable, and not supported for operational use**. The per-process rate limiter, single-GPU architecture, and in-memory coordination state do not scale beyond a single node. Security controls are complementary screening layers, not certified safeguards. Use for evaluation, benchmarking, and research only.

---

**A security hardened vLLM inference gateway delivering +57.6% throughput on NVIDIA T4 via [EAGLE 3.1](https://vllm.ai/blog/2026-05-26-eagle-3-1) speculative decoding, with CPU-only prompt screening (0–2.7% FPR), streaming PII redaction, and zero-trust container security.**

---

## Executive Summary

| Concern | Solution | Impact | Source |
|:---|:---|:---|:---|
| **Throughput** | EAGLE 3.1 speculative decoding (k=2) | **+57.6% tok/s** on Tesla T4 | [`evals/llm/eagle_31_t4_results.json`](evals/llm/eagle_31_t4_results.json) |
| **Security** | CPU-only Prompt Guard 2 (ONNX) + PII redaction | 0–2.7% FPR; zero GPU memory cost | [`evals/gaurd/results/metrics.json`](evals/gaurd/results/metrics.json) |
| **Operability** | Strict API contracts, bounded resources, structured audit | No prompt/completion data ever logged | [`image/src/secure_llm_deploy/audit.py`](image/src/secure_llm_deploy/audit.py) |

> **Deployment note:** Use `baseline` mode for simplicity; enable `speculative` (EAGLE 3.1, k=2) when GPU throughput is the binding constraint. The prompt guard is a **complementary screening layer** within defense-in-depth, not a standalone security control.

---

## Architecture

```mermaid
flowchart LR
    C[Client] -->|Bearer GATEWAY_API_KEY| G[FastAPI Gateway :8000]
    G --> A[Auth · Rate Limit · Body Bounds]
    A --> P[Prompt Guard 2 / ONNX CPU]
    P --> R[PII Redaction]
    R -->|Bearer VLLM_API_KEY · loopback| V[vLLM :8001]
    V --> O[Output Safety · PII Filter]
    O --> C
    G --> L[Metadata-Only Audit]
```

The gateway is the sole trust boundary. vLLM binds exclusively to `127.0.0.1:8001` and is never published externally.

---

## Benchmark Verdicts (NVIDIA T4)

**Environment:** vLLM 0.31.0 · Tesla T4 (14.56 GiB) · PyTorch 2.13.0+cu132 · 8 prompts × 128 tokens × 3 trials · `temperature=0.0` · `seed=42`  
**Full results:** [`evals/llm/eagle_31_t4_results.json`](evals/llm/eagle_31_t4_results.json) · [`evals/llm/eagle31_t4_summary.json`](evals/llm/eagle31_t4_summary.json)

| Configuration | Median Throughput | Speedup | Draft Acceptance | Mean Acceptance Length |
|:---|---:|---:|---:|---:|
| Baseline (autoregressive) | 35.99 tok/s | — | — | — |
| **EAGLE 3.1 (k=2)** ✅ | **56.71 tok/s** | **+57.56%** | 47.71% | 1.95 |
| EAGLE 3.1 (k=3) | 52.66 tok/s | +46.29% | 37.55% | 2.13 |

**Token fidelity:** 96.39% token-position agreement vs baseline. Single divergence attributable to BF16→FP16 cast on T4 (CC 7.5), not a correctness defect.

**Verdict:** k=2 is optimal on T4-class hardware. k=3 raises acceptance length by only +0.18 tokens but drops acceptance rate by 10pp, producing a net throughput penalty.

---

## Prompt Guard Evaluation

**Model:** [Llama Prompt Guard 2 22M (ONNX, QUInt8)](https://huggingface.co/Athithya117/Llama-Prompt-Guard-2-22M-onnx) · **Suite:** [Horizon Labs Prompt Injection Eval](https://huggingface.co/datasets/Horizon-Labs/prompt-injection-eval-suite)  
**Full results:** [`evals/gaurd/results/metrics.json`](evals/gaurd/results/metrics.json) · [`evals/gaurd/results/metrics.csv`](evals/gaurd/results/metrics.csv)

| Dataset | Samples | FPR | Recall | ROC AUC |
|:---|---:|---:|---:|---:|
| jackhhao_test | 262 | 0.0% | 89.9% | 0.996 |
| slabs_test | 2,101 | 2.7% | 38.0% | 0.868 |
| neuralchemy_test | 942 | 1.8% | 46.2% | 0.811 |
| deepset_test | 116 | 0.0% | 10.0% | 0.821 |
| agentic5k_test | 545 | 8.7% | 19.3% | 0.682 |
| bipia | 3,178 | 0.0% | 2.3% | 0.650 |
| piarena | 1,500 | 0.0% | 1.6% | 0.775 |

**Why CPU?** Published workload analysis shows ~83% GPU / ~5% CPU utilization during inference ([Applied Sciences, 2026](https://www.mdpi.com/2076-3417/16/7/3129)). The 22M-param ONNX guard (~90 MB) exploits idle CPU capacity, preserving all GPU VRAM for KV cache.

**Deployment role:** Low FPR (0–2.7%) makes blocking safe. Variable recall (1.6–89.9%) means it is **complementary**, not sufficient alone. Default policy is fail-closed (`GUARD_FAILURE_POLICY=closed`).

---

## Security Posture

| Control | Implementation |
|:---|:---|
| Authentication | Constant-time Bearer comparison; distinct gateway/internal keys (≥32 chars) |
| Request bounding | ASGI-level body enforcement; Pydantic `extra="forbid"` |
| Rate limiting | Sliding-window per client IP; max 10K tracked clients |
| Prompt screening | CPU-only ONNX; fail-closed; rejects partial scans |
| PII redaction | Local regex + configured secrets; input and streaming output |
| Output safety | Structural validation; tool-call rejection; cross-chunk PII holdback |
| Audit | Metadata-only JSON; no prompt/completion/auth content logged |
| Container | Non-root (UID 2000); read-only artifacts; offline model loading |
| Supply chain | Digest-pinned base image; `uv.lock`; `model.lock` immutable revisions |

**Explicit limitations:** Guard produces false negatives. PII redaction is pattern-based. Rate limiter is per-process. TLS, secret rotation, and network segmentation are deployment responsibilities.

---

## Quick Start

```bash
# Build
docker buildx build --platform linux/amd64 --load \
  --file image/Dockerfile --tag secure-llm-deploy:local .

# Configure
cat > .env.local <<EOF
GATEWAY_API_KEY=$(openssl rand -base64 48)
VLLM_API_KEY=$(openssl rand -base64 48)
SERVE_MODE=speculative
EOF

# Run
docker run --rm --gpus all --env-file .env.local \
  -p 127.0.0.1:8000:8000 --name secure-llm-deploy secure-llm-deploy:local
```

**Smoke tests:** `SERVE_MODE=both ./evals/llm/smoke-tests.sh`

---

## Model Provenance

| Role | Repository | Notes |
|:---|:---|:---|
| Target (official) | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Alibaba Qwen Team |
| Target (deployed) | [`Athithya117/Qwen3-8B-AWQ`](https://huggingface.co/Athithya117/Qwen3-8B-AWQ) | AWQ quantized for T4 |
| Draft (deployed) | [`Athithya117/...eagle31-qwen3arch-ckpt1`](https://huggingface.co/Athithya117/Qwen3-8B-from-Qwen3-8B_regen-speculators.eagle31-qwen3arch-ckpt1) | EAGLE 3.1, qwen3arch-aligned |
| Draft (upstream) | [`RedHatAI/Qwen3-8B-speculator.eagle3`](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3) | SpecForge-trained reference |
| Guard | [`Athithya117/Llama-Prompt-Guard-2-22M-onnx`](https://huggingface.co/Athithya117/Llama-Prompt-Guard-2-22M-onnx) | QUInt8 ONNX, CPU-only |
| EAGLE method | [`SafeAILab/EAGLE`](https://github.com/SafeAILab/EAGLE) | UC Berkeley |

All deployed models are pinned by immutable commit revision in [`image/model.lock`](image/model.lock).

---

## Release Artifacts

| Artifact | Location |
|:---|:---|
| Container image | `ghcr.io/athithya-sakthivel/secure-llm-deploy` (immutable digest) |
| Dependency lock | [`uv.lock`](uv.lock) |
| Vulnerability scan | [Trivy](https://github.com/aquasecurity/trivy-action) |
| Image signing | [Cosign](https://docs.sigstore.dev/cosign/verifying/verify/) keyless |
| SBOM + attestation | [GitHub artifact attestations](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations) |

---

## Upstream Documentation

- [vLLM Docker deployment](https://docs.vllm.ai/en/v0.31.0/deployment/docker/)
- [vLLM speculative decoding](https://docs.vllm.ai/en/v0.31.0/examples/features/speculative_decoding/)
- [EAGLE official implementation](https://github.com/SafeAILab/EAGLE)
- [NVIDIA NeMo EAGLE training recipe](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/eagle-speculative-decoding)
