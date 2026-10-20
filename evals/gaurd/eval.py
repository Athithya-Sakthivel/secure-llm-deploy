#!/usr/bin/env python3
"""CPU-only Prompt Guard ONNX evaluator. No torch or transformers imports."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import onnxruntime as ort
    from datasets import get_dataset_config_names, load_dataset
    from tokenizers import Tokenizer
except ImportError as exc:
    package = getattr(exc, "name", None) or str(exc)
    print(
        f"ERROR: missing dependency {package}. Install numpy, onnxruntime, "
        "tokenizers and datasets. PyTorch/Transformers are not required.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

LOG = logging.getLogger("prompt_guard_eval")
DEFAULT_DATASET = "Horizon-Labs/prompt-injection-eval-suite"
DEFAULT_CONFIGS = "notinject,xstest,slabs_test"


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate a Prompt Guard ONNX model on CPU only."
    )
    parser.add_argument(
        "--model-path",
        help="Path to model.quant.onnx; auto-discovered if omitted.",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Directory containing tokenizer.json, or its path.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--dataset-revision",
        default="main",
        help="Dataset revision or commit hash. Pin a commit for reproducibility.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="First N rows per config; 0 means all.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ORT intra-op CPU threads; 0 lets ORT choose.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.0,
        help=(
            "Predict malicious when malicious_logit - benign_logit "
            "is greater than this threshold."
        ),
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--output-dir", default=str(here / "results"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if (
        args.max_samples < 0
        or args.threads < 0
        or args.warmup_runs < 0
        or args.batch_size < 1
        or args.max_length < 2
    ):
        parser.error(
            "max-samples, threads and warmup-runs must be >= 0; "
            "batch-size must be >= 1; max-length must be >= 2"
        )
    if not math.isfinite(args.margin_threshold):
        parser.error("--margin-threshold must be a finite number")

    args.config_list = list(
        dict.fromkeys(c.strip() for c in args.configs.split(",") if c.strip())
    )
    if not args.config_list:
        parser.error("--configs must contain at least one configuration")
    return args


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(paths: list[Path]) -> list[Path]:
    """Resolve paths and preserve their order while removing duplicates."""
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def find_model(arg: str | None, script_dir: Path) -> Path:
    """Find the ONNX model from an explicit path or known local locations."""
    override = arg or os.getenv("PROMPT_GUARD_MODEL")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Model file does not exist: {path}")

    bases = unique([Path.cwd(), script_dir, *script_dir.parents])
    candidates: list[Path] = []
    for base in bases:
        candidates.extend(
            [
                base / "model.quant.onnx",
                base / "models/model.quant.onnx",
                base / "Llama-Prompt-Guard-2-22M-onnx/model.quant.onnx",
                base / "models/Llama-Prompt-Guard-2-22M-onnx/model.quant.onnx",
            ]
        )

    for path in unique(candidates):
        if path.is_file():
            return path

    cache = Path(
        os.getenv("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")
    ).expanduser()
    for repo in (
        "models--gravitee-io--Llama-Prompt-Guard-2-22M-onnx",
        "models--Athithya117--Llama-Prompt-Guard-2-22M-onnx",
    ):
        snapshots = cache / repo / "snapshots"
        if snapshots.is_dir():
            found = list(snapshots.glob("*/model.quant.onnx"))
            if found:
                return max(found, key=lambda path: path.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "Cannot find model.quant.onnx. Pass --model-path /path/model.quant.onnx "
        "or set PROMPT_GUARD_MODEL. Model weights are not downloaded implicitly."
    )


def find_tokenizer(
    arg: str | None, model_path: Path, script_dir: Path
) -> tuple[Path, Path]:
    """Find tokenizer.json and return it with the directory containing it."""
    if arg:
        path = Path(arg).expanduser().resolve()
        if path.is_file() and path.name == "tokenizer.json":
            return path, path.parent
        if path.is_dir() and (path / "tokenizer.json").is_file():
            return path / "tokenizer.json", path
        raise FileNotFoundError(
            "--tokenizer-path must be tokenizer.json or a directory containing it: "
            f"{path}"
        )

    bases = unique(
        [
            model_path.parent,
            script_dir,
            Path.cwd(),
            *model_path.parents,
            *script_dir.parents,
        ]
    )
    for base in bases:
        candidate = base / "tokenizer.json"
        if candidate.is_file():
            return candidate, base

    raise FileNotFoundError(
        "Cannot find tokenizer.json near the model. Pass --tokenizer-path to "
        "the model directory containing the matching tokenizer."
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read optional JSON metadata; missing or malformed files become empty metadata."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("Ignoring unreadable optional metadata %s: %s", path, exc)
        return {}


def token_text(value: Any) -> str | None:
    """Extract a token's text from common tokenizer metadata representations."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def build_tokenizer(
    tokenizer_json: Path,
    tokenizer_dir: Path,
    model_dir: Path,
    max_length: int,
    fixed_length: int | None,
) -> Tokenizer:
    """Load the fast tokenizer and configure safe truncation/padding."""
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    tokenizer_config = read_json(tokenizer_dir / "tokenizer_config.json")
    special_map = read_json(tokenizer_dir / "special_tokens_map.json")
    model_config = read_json(model_dir / "config.json") or read_json(
        tokenizer_dir / "config.json"
    )

    pad_token = token_text(tokenizer_config.get("pad_token")) or token_text(
        special_map.get("pad_token")
    )
    get_padding = getattr(tokenizer, "get_padding", None)
    padding = get_padding() if callable(get_padding) else None
    pad_id = padding.get("pad_id") if isinstance(padding, dict) else None

    if pad_token is None and isinstance(padding, dict):
        candidate = padding.get("pad_token")
        if isinstance(candidate, str):
            pad_token = candidate

    if pad_token is None:
        for candidate in ("[PAD]", "<pad>", "<PAD>"):
            if tokenizer.token_to_id(candidate) is not None:
                pad_token = candidate
                break

    if pad_id is None and pad_token is not None:
        pad_id = tokenizer.token_to_id(pad_token)

    if pad_id is None and model_config.get("pad_token_id") is not None:
        try:
            pad_id = int(model_config["pad_token_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("config.json contains an invalid pad_token_id") from exc
        pad_token = tokenizer.id_to_token(pad_id)

    if pad_id is None or pad_token is None:
        raise ValueError(
            f"Cannot determine pad token from {tokenizer_dir}. Check "
            "tokenizer_config.json, special_tokens_map.json and config.json."
        )

    tokenizer.enable_truncation(max_length=max_length, direction="right")
    tokenizer.enable_padding(
        direction="right",
        pad_id=int(pad_id),
        pad_token=str(pad_token),
        length=fixed_length,
    )
    return tokenizer


def malicious_index(config: dict[str, Any]) -> int:
    """Infer which binary output label represents malicious input."""
    labels = config.get("id2label")
    if isinstance(labels, dict):
        hits: list[int] = []
        for key, label in labels.items():
            if isinstance(label, str) and any(
                word in label.lower()
                for word in ("malicious", "injection", "jailbreak", "attack", "positive")
            ):
                with contextlib.suppress(TypeError, ValueError):
                    hits.append(int(key))
        unique_hits = set(hits)
        if len(unique_hits) == 1:
            inferred = next(iter(unique_hits))
            if inferred in (0, 1):
                return inferred

    LOG.warning(
        "Cannot infer malicious output class from config.json; assuming index 1. "
        "Verify this matches the model's label mapping."
    )
    return 1


def make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    """Create an ONNX Runtime session with the CPU provider only."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.log_severity_level = 3

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(
            "CPU-only session expected; actual providers="
            f"{session.get_providers()}"
        )
    return session


def make_feed(
    session: ort.InferenceSession, encodings: list[Any]
) -> dict[str, np.ndarray]:
    """Create correctly typed model inputs for a batch of tokenizer encodings."""
    source = {
        "input_ids": np.asarray([x.ids for x in encodings], dtype=np.int64),
        "attention_mask": np.asarray(
            [x.attention_mask for x in encodings], dtype=np.int64
        ),
        "token_type_ids": np.asarray([x.type_ids for x in encodings], dtype=np.int64),
    }
    dtypes = {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
    }
    feed: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        name = item.name.lower()
        if name not in source:
            raise ValueError(
                f"Unsupported required ONNX input {item.name!r}; expected "
                "input_ids, attention_mask or token_type_ids."
            )
        if item.type not in dtypes:
            raise ValueError(
                f"Unsupported input type {item.type!r} for {item.name!r}"
            )
        feed[item.name] = source[name].astype(dtypes[item.type], copy=False)

    if not any(name.lower() == "input_ids" for name in feed):
        raise ValueError("ONNX graph has no input_ids input.")
    return feed


def predict_logits(
    session: ort.InferenceSession,
    feed: dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    """Run a batch and return finite binary logits in [batch, 2] shape."""
    names = [item.name for item in session.get_outputs()]
    outputs = session.run(None, feed)
    order = [
        i
        for i, name in enumerate(names)
        if name.lower() in ("logits", "output_logits")
    ]
    order.extend(i for i in range(len(outputs)) if i not in order)

    for i in order:
        value = np.asarray(outputs[i])
        if value.ndim == 2 and value.shape == (batch_size, 2):
            if not np.isfinite(value).all():
                raise ValueError("Model returned NaN or infinite logits.")
            return value.astype(np.float64, copy=False)

    raise ValueError(
        "Expected logits with shape [batch, 2]. Output shapes: "
        f"{[np.asarray(x).shape for x in outputs]}; names={names}"
    )


def columns(names: list[str]) -> tuple[str, str]:
    """Infer text and label columns from common dataset column names."""
    by_lower = {name.lower(): name for name in names}
    text = next(
        (
            by_lower[key]
            for key in (
                "text",
                "prompt",
                "content",
                "input",
                "query",
                "instruction",
                "message",
            )
            if key in by_lower
        ),
        None,
    )
    label = next(
        (
            by_lower[key]
            for key in (
                "label",
                "labels",
                "target",
                "is_injection",
                "injection",
                "y",
                "class",
            )
            if key in by_lower
        ),
        None,
    )
    if text is None or label is None:
        raise ValueError(f"Cannot identify text/label columns in {names}")
    return text, label


def normalize_label(value: Any, row: int) -> int:
    """Normalize common binary dataset labels: benign=0, malicious=1."""
    if isinstance(value, (int, np.integer, bool, np.bool_)) or (
        isinstance(value, (float, np.floating)) and value in (0.0, 1.0)
    ):
        result = int(value)
    elif isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "1",
            "true",
            "yes",
            "malicious",
            "injection",
            "prompt_injection",
            "jailbreak",
            "attack",
            "positive",
        }:
            result = 1
        elif normalized in {
            "0",
            "false",
            "no",
            "benign",
            "clean",
            "safe",
            "negative",
            "not_injection",
            "non_injection",
        }:
            result = 0
        else:
            raise ValueError(
                f"Row {row}: cannot map label {value!r}; "
                "expected benign=0/malicious=1."
            )
    else:
        raise ValueError(f"Row {row}: unsupported label value {value!r}.")

    if result not in (0, 1):
        raise ValueError(f"Row {row}: label must be 0 or 1, got {result}.")
    return result


def ratio(numerator: int, denominator: int) -> float | None:
    """Compute a ratio, returning None when the denominator is zero."""
    return float(numerator / denominator) if denominator else None


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute ROC AUC from ranks, averaging ranks for tied scores."""
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end

    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calculate_metrics(
    labels: list[int], predictions: list[int], scores: list[float]
) -> dict[str, Any]:
    """Calculate classification metrics and ROC AUC."""
    if not (len(labels) == len(predictions) == len(scores)):
        raise ValueError("labels, predictions and scores must have equal lengths")

    y = np.asarray(labels)
    pred = np.asarray(predictions)
    score = np.asarray(scores, dtype=np.float64)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return {
        "n_samples": len(y),
        "n_benign": int(np.sum(y == 0)),
        "n_malicious": int(np.sum(y == 1)),
        "n_predicted_benign": int(np.sum(pred == 0)),
        "n_predicted_malicious": int(np.sum(pred == 1)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": ratio(tp + tn, len(y)),
        "precision_malicious": precision,
        "recall_malicious": recall,
        "f1_malicious": ratio(2 * tp, 2 * tp + fp + fn),
        "specificity_benign": specificity,
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        "matthews_corrcoef": (tp * tn - fp * fn) / denom if denom else None,
        "roc_auc": auc_rank(y, score),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON so interrupted runs do not leave partial reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    """Atomically write the per-configuration metrics CSV."""
    fields = [
        "config",
        "n_samples",
        "n_benign",
        "n_malicious",
        "accuracy",
        "precision_malicious",
        "recall_malicious",
        "f1_malicious",
        "specificity_benign",
        "false_positive_rate",
        "false_negative_rate",
        "balanced_accuracy",
        "matthews_corrcoef",
        "roc_auc",
        "samples_per_inference_second",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results.values():
            writer.writerow(row)
    temp.replace(path)


def evaluate_config(
    name: str,
    dataset: Any,
    session: ort.InferenceSession,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    positive_index: int,
    max_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one dataset configuration and return metrics plus row results."""
    original_count = len(dataset)
    if original_count == 0:
        raise ValueError("Requested dataset split is empty.")
    if args.max_samples:
        dataset = dataset.select(range(min(original_count, args.max_samples)))

    text_col, label_col = columns(list(dataset.column_names))
    texts: list[str] = []
    labels: list[int] = []
    for i, row in enumerate(dataset):
        text = row.get(text_col)
        if text is None:
            raise ValueError(f"Row {i}: null text field {text_col!r}.")
        texts.append(text if isinstance(text, str) else str(text))
        labels.append(normalize_label(row.get(label_col), i))

    LOG.info(
        "[%s] loaded %d/%d rows (split=%s, text=%s, label=%s)",
        name,
        len(texts),
        original_count,
        args.split,
        text_col,
        label_col,
    )

    warm_size = min(args.batch_size, len(texts))
    warm_feed = make_feed(session, tokenizer.encode_batch(texts[:warm_size]))
    for _ in range(args.warmup_runs):
        predict_logits(session, warm_feed, warm_size)

    scores: list[float] = []
    predictions: list[int] = []
    rows: list[dict[str, Any]] = []
    batch_ms: list[float] = []
    inference_seconds = 0.0

    for start in range(0, len(texts), args.batch_size):
        end = min(start + args.batch_size, len(texts))
        encodings = tokenizer.encode_batch(texts[start:end])
        feed = make_feed(session, encodings)

        tick = time.perf_counter()
        logits = predict_logits(session, feed, end - start)
        elapsed = time.perf_counter() - tick
        inference_seconds += elapsed
        batch_ms.append(elapsed * 1000.0)

        margins = logits[:, positive_index] - logits[:, 1 - positive_index]
        batch_predictions = (margins > args.margin_threshold).astype(np.int64)

        # This is the two-class softmax score, not a calibrated confidence.
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(margins, -60.0, 60.0)))
        scores.extend(float(x) for x in margins)
        predictions.extend(int(x) for x in batch_predictions)

        for j, (text, label, pred, margin, probability) in enumerate(
            zip(
                texts[start:end],
                labels[start:end],
                batch_predictions,
                margins,
                probabilities,
                strict=True,
            )
        ):
            rows.append(
                {
                    "config": name,
                    "split": args.split,
                    "row_index": start + j,
                    "label": label,
                    "prediction": int(pred),
                    "correct": bool(int(pred) == label),
                    "malicious_logit_margin": float(margin),
                    "malicious_softmax_score": float(probability),
                    "text": text,
                }
            )

        if end == len(texts) or end % (args.batch_size * 10) == 0:
            LOG.info("[%s] scored %d/%d", name, end, len(texts))

    result = calculate_metrics(labels, predictions, scores)
    result.update(
        {
            "dataset": args.dataset,
            "config": name,
            "split": args.split,
            "n_dataset_rows_before_limit": original_count,
            "batch_size": args.batch_size,
            "max_length": max_length,
            "margin_threshold": args.margin_threshold,
            "inference_seconds": round(inference_seconds, 6),
            "samples_per_inference_second": (
                round(len(texts) / inference_seconds, 3)
                if inference_seconds
                else None
            ),
            "batch_latency_ms_median": (
                round(float(np.median(batch_ms)), 3) if batch_ms else None
            ),
            "batch_latency_ms_p95": (
                round(float(np.percentile(batch_ms, 95)), 3) if batch_ms else None
            ),
            "latency_note": "session.run only; excludes tokenization and dataset loading",
        }
    )
    return result, rows


def fmt(value: Any) -> str:
    """Format an optional metric for logging."""
    return "n/a" if value is None else f"{float(value):.4f}"


def main() -> int:
    """Run evaluation, persist metrics, and return a process exit code."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import tokenizers

    LOG.info(
        "Starting CPU-only evaluation | Python=%s ONNX Runtime=%s tokenizers=%s",
        sys.version.split()[0],
        ort.__version__,
        tokenizers.__version__,
    )
    LOG.info("No PyTorch or Transformers imports are used.")
    LOG.info(
        "Dataset=%s revision=%s split=%s configs=%s",
        args.dataset,
        args.dataset_revision,
        args.split,
        ",".join(args.config_list),
    )

    script_dir = Path(__file__).resolve().parent
    model_path = find_model(args.model_path, script_dir)
    tokenizer_json, tokenizer_dir = find_tokenizer(
        args.tokenizer_path, model_path, script_dir
    )
    config = read_json(model_path.parent / "config.json") or read_json(
        tokenizer_dir / "config.json"
    )
    session = make_session(model_path, args.threads)

    model_hash = sha256_file(model_path)
    LOG.info(
        "Model=%s bytes=%d sha256=%s",
        model_path,
        model_path.stat().st_size,
        model_hash,
    )
    LOG.info(
        "Tokenizer=%s; available providers=%s; session providers=%s",
        tokenizer_json,
        ort.get_available_providers(),
        session.get_providers(),
    )

    inputs = {item.name.lower(): item for item in session.get_inputs()}
    if "input_ids" not in inputs:
        raise ValueError(
            "ONNX input_ids missing; inputs="
            f"{[item.name for item in session.get_inputs()]}"
        )
    shape = inputs["input_ids"].shape
    fixed_length = (
        shape[1] if len(shape) > 1 and isinstance(shape[1], int) else None
    )
    max_length = min(args.max_length, fixed_length) if fixed_length else args.max_length

    tokenizer = build_tokenizer(
        tokenizer_json, tokenizer_dir, model_path.parent, max_length, fixed_length
    )
    positive_index = malicious_index(config)
    LOG.info(
        "max_length=%d fixed_sequence_length=%s malicious_output_index=%d",
        max_length,
        fixed_length,
        positive_index,
    )

    try:
        known_configs = list(
            get_dataset_config_names(
                args.dataset, revision=args.dataset_revision
            )
        )
        LOG.info("Dataset configs: %s", ", ".join(known_configs))
    except Exception as exc:
        known_configs = None
        LOG.warning(
            "Could not list dataset configs (%s); will try requested configs directly.",
            exc,
        )

    report: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "dataset_revision_requested": args.dataset_revision,
        "dataset_revision_note": (
            "main is mutable; pin a commit hash for strict reproducibility"
        ),
        "split": args.split,
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "tokenizer_json": str(tokenizer_json),
        "onnxruntime_version": ort.__version__,
        "providers": session.get_providers(),
        "label_convention": {"0": "benign", "1": "injection/malicious"},
        "malicious_logit_index": positive_index,
        "decision_rule": {
            "score": "malicious_logit - benign_logit",
            "operator": ">",
            "threshold": args.margin_threshold,
        },
        "arguments": {
            key: value for key, value in vars(args).items() if key != "config_list"
        },
        "results": {},
        "failures": [],
    }
    if args.dataset_revision == "main":
        LOG.warning(
            "Dataset revision 'main' is mutable. Use --dataset-revision "
            "<commit-hash> for audited runs."
        )

    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    predictions_dir = output_dir / "predictions"

    for name in args.config_list:
        if known_configs is not None and name not in known_configs:
            message = (
                f"Unknown config {name!r}; available configs: "
                f"{', '.join(known_configs)}"
            )
            LOG.error("[%s] %s", name, message)
            failures.append({"config": name, "error": message})
        else:
            try:
                dataset = load_dataset(
                    args.dataset,
                    name=name,
                    split=args.split,
                    revision=args.dataset_revision,
                )
                metrics, prediction_rows = evaluate_config(
                    name,
                    dataset,
                    session,
                    tokenizer,
                    args,
                    positive_index,
                    max_length,
                )
                results[name] = metrics

                predictions_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = predictions_dir / f"{name}.jsonl"
                temp = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
                with temp.open("w", encoding="utf-8") as handle:
                    for row in prediction_rows:
                        handle.write(
                            json.dumps(
                                row, ensure_ascii=False, allow_nan=False
                            )
                            + "\n"
                        )
                temp.replace(prediction_path)

                LOG.info(
                    "[%s] accuracy=%s precision=%s recall=%s F1=%s FPR=%s AUC=%s",
                    name,
                    fmt(metrics["accuracy"]),
                    fmt(metrics["precision_malicious"]),
                    fmt(metrics["recall_malicious"]),
                    fmt(metrics["f1_malicious"]),
                    fmt(metrics["false_positive_rate"]),
                    fmt(metrics["roc_auc"]),
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                LOG.error("[%s] failed: %s", name, message)
                if args.debug:
                    LOG.exception("Traceback for config %s", name)
                failures.append({"config": name, "error": message})

        report["results"], report["failures"] = results, failures
        write_json(output_dir / "metrics.json", report)
        write_csv(output_dir / "metrics.csv", results)

    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    report["results"], report["failures"] = results, failures
    write_json(output_dir / "metrics.json", report)
    write_csv(output_dir / "metrics.csv", results)

    LOG.info("Results directory: %s", output_dir)
    LOG.info("Successful configurations: %d/%d", len(results), len(args.config_list))
    if failures:
        LOG.error("%d config(s) failed; inspect metrics.json.", len(failures))
        return 1

    LOG.info("Evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
