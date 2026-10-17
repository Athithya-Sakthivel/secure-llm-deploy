# Prompt Guard 2: CPU-Based Prompt-Injection Screening

## Summary

This evaluation tests the quantized **Llama Prompt Guard 2 22M** model as a lightweight, CPU-based screening layer for prompt-injection and jailbreak attempts.

The model is small enough to run without a dedicated GPU. This makes it a candidate for screening inputs before they reach a GPU-hosted LLM, provided CPU capacity and end-to-end latency are validated in the target environment.

**Decision:** Treat Prompt Guard as an additional risk signal—not as the sole security control. The tested threshold produced low false-positive rates on several datasets, but attack detection varied considerably, especially for indirect prompt injections.

## Why this model and execution approach?

### Small model, CPU deployment

The 22M-parameter model is substantially smaller than the primary generative LLM it may protect. A quantized ONNX version provides a practical way to run a separate classifier on CPU, potentially using CPU capacity available alongside GPU inference.

CPU availability must be measured on the actual serving VM. A published *Applied Sciences* study reported one GPU-deployed inference workload with approximately 83% GPU utilization and 5% CPU utilization. This illustrates that CPU can be lightly used while a GPU is active; it does **not** establish that CPU is generally idle across production GPU workloads. [Source](https://www.mdpi.com/2076-3417/16/7/3129)

### ONNX Runtime on CPU

The evaluation uses ONNX Runtime with `CPUExecutionProvider` to execute the exported quantized model directly. This isolates the screening workload from GPU execution and avoids requiring a GPU for the classifier.

The measured inference throughput is for ONNX session execution only. It excludes tokenization, dataset loading, and other request-processing overhead, so it should not be interpreted as end-to-end service throughput or latency.

### Why not PyTorch or Transformers?

The evaluator uses the model’s ONNX graph and the provided `tokenizer.json` directly. It therefore does not need to load the model through PyTorch or the Transformers model stack.

This keeps the evaluation runtime focused on ONNX Runtime and the Rust-backed `tokenizers` library, reducing framework dependencies and avoiding the need to install PyTorch solely for inference. It does not, by itself, prove lower production latency or a smaller total deployment footprint; those should be measured.

### Long-context use

Prompt-injection screening can be applied to user prompts and, where the application supports it, untrusted retrieved text or tool content. For long inputs, the application must define how text is tokenized, truncated, or divided into chunks and how chunk-level flags are combined.

This evaluation does **not** establish unlimited long-context support. Validate the model’s maximum input length, truncation behavior, memory use, and latency for the context sizes used by the application.

## Model and data sources

- ONNX model and tokenizer: [Gravitee Llama Prompt Guard 2 22M ONNX](https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-22M-onnx)
- Evaluation datasets: [Horizon Labs Prompt Injection Evaluation Suite](https://huggingface.co/datasets/Horizon-Labs/prompt-injection-eval-suite)
- Evaluated model artifact: `model.QUInt8.quant.onnx`
- Runtime: ONNX Runtime, `CPUExecutionProvider`
- Tokenization: `tokenizer.json` using the `tokenizers` package

The Hugging Face model and dataset repositories are the source artifacts; the reported metrics below are from this project’s evaluation run.

## Setup

From the repository root:

```bash
bash evals/gaurd/bootstrap.sh
```

The bootstrap script creates the virtual environment, installs dependencies, and downloads the model and tokenizer files.

## Run

Evaluate the selected full dataset splits:

```bash
EVAL_CONFIGS=deepset_test,jackhhao_test,boundary_pairs_test,agentic5k_test,neuralchemy_test,slabs_test,bipia,piarena \
MAX_SAMPLES=0 \
bash evals/gaurd/run.sh
```

`MAX_SAMPLES=0` requests all available examples for each selected configuration.

## Results

The following results were measured at the evaluator’s tested decision threshold of **0**. False-positive rate (FPR) is the fraction of benign examples classified as malicious; malicious recall is the fraction of malicious examples detected.

| Dataset | Samples | FPR | Malicious recall | ROC AUC |
|---|---:|---:|---:|---:|
| deepset_test | 116 | 0.0% | 10.0% | 0.821 |
| jackhhao_test | 262 | 0.0% | 89.9% | 0.996 |
| boundary_pairs_test | 240 | 8.3% | 25.0% | 0.593 |
| agentic5k_test | 545 | 8.7% | 19.3% | 0.682 |
| neuralchemy_test | 942 | 1.8% | 46.2% | 0.811 |
| slabs_test | 2,101 | 2.7% | 38.0% | 0.868 |
| bipia | 3,178 | 0.0% | 2.3% | 0.650 |
| piarena | 1,500 | 0.0% | 1.6% | 0.775 |

These are dataset-specific measurements, not guaranteed production rates. ROC AUC describes ranking quality across thresholds; it does not mean the classifier achieves high recall at the current threshold.

### Interpretation

- **False positives were low** on several test sets, including jackhhao, deepset, neuralchemy, slabs, BIPIA, and PIArena.
- **Detection recall was inconsistent.** The classifier performed well on jackhhao but missed most malicious examples in BIPIA and PIArena at this threshold.
- Some datasets have moderate or high AUC despite low recall at threshold 0. This suggests that threshold selection may affect the operating trade-off, but it does not guarantee a threshold with both high recall and acceptably low FPR.
- Dataset composition matters. These benchmark results should not be treated as estimates of real-world traffic performance.

## Recommended role in the security architecture

Use Prompt Guard as a **supplementary screening layer**:

1. Score user prompts and relevant untrusted content before passing them to the LLM or tools.
2. Treat a positive classification as a risk signal and apply the application’s policy—for example, block, sanitize, or route for review.
3. Maintain independent protections such as tool authorization, least-privilege access, untrusted-content isolation, and output validation.
4. Tune the decision threshold on representative validation data to meet an explicit false-positive target, then report recall and FPR on held-out data.
5. Benchmark end-to-end latency and CPU utilization under realistic concurrency and context lengths before deployment.

**Conclusion:** The model is feasible to evaluate and potentially deploy as a CPU-side screening component. Its low false-positive results are promising for a layered design, but the weak and dataset-dependent attack recall means it should not be relied on to stop prompt injection by itself.

## Evaluation outputs

The run writes:

- `results/metrics.csv` — aggregate metrics
- `results/metrics.json` — aggregate metrics in JSON
- `results/predictions/` — per-example predictions in JSONL

