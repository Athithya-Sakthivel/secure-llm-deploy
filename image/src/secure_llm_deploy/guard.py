"""CPU-only Prompt Guard 2 inference over a local ONNX model."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

GuardDecision = Literal["allow", "block"]
GuardFailurePolicy = Literal["closed", "open"]
GuardOverlongPolicy = Literal["reject", "truncate"]


class GuardError(RuntimeError):
    """Base error for prompt-guard input and inference failures."""


class GuardUnavailableError(GuardError):
    """The guard artifacts or ONNX runtime could not provide a reliable score."""


class GuardInputTooLongError(GuardError):
    """The input exceeded the guard budget and policy forbids truncation."""


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Typed, auditable result of checking one prompt.

    ``score`` is the model-derived probability for the malicious class. It is
    ``None`` when inference failed. ``decision`` already reflects the configured
    fail-open/fail-closed policy. A truncated result explicitly says so.
    """

    decision: GuardDecision
    label: str
    score: float | None
    input_tokens: int
    tokens_scanned: int
    chunks_evaluated: int
    truncated: bool = False
    reason: str = "classified"
    error_code: str | None = None

    @property
    def blocked(self) -> bool:
        """Whether the gateway should reject this prompt."""
        return self.decision == "block"

    @property
    def allowed(self) -> bool:
        """Whether the guard decision permits this prompt to continue."""
        return self.decision == "allow"


class ONNXPromptGuard:
    """Run a local classifier on CPU, discovering its ONNX I/O contract.

    ``max_input_tokens`` limits the total tokenized text. ``window_tokens`` is
    the maximum ONNX sequence length per chunk (including CLS/SEP tokens).
    Inputs longer than the model window are evaluated as independent token
    windows, so ordinary long inputs are not silently truncated. Only when the
    total configured budget is exceeded can the explicit ``overlong_policy``
    permit prefix truncation.

    Fail-open/fail-closed applies to runtime inference failures. Invalid
    artifacts and incompatible model signatures always fail construction so a
    service does not silently start without a usable guard.
    """

    ATTACK_LABELS = frozenset(
        {"attack", "injection", "jailbreak", "malicious", "prompt_injection", "unsafe"}
    )
    BENIGN_LABELS = frozenset(
        {"benign", "clean", "harmless", "not_malicious", "safe"}
    )
    SUPPORTED_INPUTS = frozenset(
        {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    )

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        config_path: Path | None = None,
        *,
        threshold: float = 0.5,
        max_input_tokens: int = 6144,
        window_tokens: int | None = None,
        intra_op_threads: int = 2,
        failure_policy: GuardFailurePolicy | None = None,
        overlong_policy: GuardOverlongPolicy | None = None,
    ) -> None:
        """Load the tokenizer/config and validate the CPU ONNX session contract."""
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Guard threshold must be finite and between 0 and 1")
        if max_input_tokens < 1:
            raise ValueError("Guard maximum input-token budget must be positive")
        if intra_op_threads < 1:
            raise ValueError("Guard intra-op thread count must be positive")
        raw_failure_policy = (
            failure_policy
            if failure_policy is not None
            else os.environ.get("GUARD_FAILURE_POLICY", "closed")
        )
        raw_overlong_policy = (
            overlong_policy
            if overlong_policy is not None
            else os.environ.get("GUARD_OVERLONG_POLICY", "reject")
        )
        if raw_failure_policy not in {"closed", "open"}:
            raise ValueError("Guard failure policy must be 'closed' or 'open'")
        if raw_overlong_policy not in {"reject", "truncate"}:
            raise ValueError("Guard overlong policy must be 'reject' or 'truncate'")
        self.failure_policy: GuardFailurePolicy = cast(
            GuardFailurePolicy, raw_failure_policy
        )
        self.overlong_policy: GuardOverlongPolicy = cast(
            GuardOverlongPolicy, raw_overlong_policy
        )

        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.config_path = Path(config_path) if config_path else self.model_path.parent / "config.json"
        self.threshold = threshold
        self.max_input_tokens = max_input_tokens

        for path in (self.model_path, self.tokenizer_path, self.config_path):
            if not path.is_file():
                raise ValueError(f"Required guard artifact is missing: {path}")

        try:
            self.model_config = self._read_json(self.config_path)
            tokenizer_config_path = self.tokenizer_path.with_name("tokenizer_config.json")
            self.tokenizer_config = (
                self._read_json(tokenizer_config_path)
                if tokenizer_config_path.is_file()
                else {}
            )
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        except Exception as exc:
            raise GuardUnavailableError(
                "Prompt Guard tokenizer/config could not be loaded"
            ) from exc

        self.id_to_label, self.attack_index, self.benign_index = self._load_labels(
            self.model_config
        )
        self.attack_label = self.id_to_label[self.attack_index]
        self.benign_label = self.id_to_label[self.benign_index]

        position_limit = self.model_config.get("max_position_embeddings", 512)
        if not isinstance(position_limit, int) or position_limit < 3:
            raise ValueError(
                "Prompt Guard config must contain a valid max_position_embeddings"
            )
        if window_tokens is None:
            env_window = os.environ.get("GUARD_WINDOW_TOKENS")
            try:
                requested_window = int(env_window) if env_window else position_limit
            except ValueError as exc:
                raise ValueError("GUARD_WINDOW_TOKENS must be an integer") from exc
        else:
            requested_window = window_tokens
        if requested_window < 3:
            raise ValueError("Guard window must allow at least CLS, one token, and SEP")
        if requested_window > position_limit:
            raise ValueError(
                "GUARD_WINDOW_TOKENS exceeds config.json max_position_embeddings"
            )

        cls_token_id = self._special_token_id(
            id_field="cls_token_id",
            token_field="cls_token",
            candidates=("[CLS]", "<s>", "<|startoftext|>", "[BOS]"),
        )
        sep_token_id = self._special_token_id(
            id_field="sep_token_id",
            token_field="sep_token",
            candidates=("[SEP]", "</s>", "<|endoftext|>", "[EOS]"),
        )
        if cls_token_id is None or sep_token_id is None:
            raise ValueError(
                "Prompt Guard tokenizer must expose recognizable CLS and SEP token IDs"
            )
        self.cls_token_id: int = cls_token_id
        self.sep_token_id: int = sep_token_id

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise GuardUnavailableError(
                "Prompt Guard ONNX session could not be created"
            ) from exc

        providers = self.session.get_providers()
        if set(providers) != {"CPUExecutionProvider"}:
            raise ValueError(
                "Prompt Guard must use CPUExecutionProvider only; "
                f"actual providers: {providers!r}"
            )

        self.input_specs = self._validate_inputs(self.session.get_inputs())
        self.output_name, self.output_kind = self._select_output(
            self.session.get_outputs(), len(self.id_to_label)
        )

        static_sequence_lengths = [
            spec.shape[1]
            for spec in self.input_specs
            if isinstance(spec.shape[1], int) and spec.shape[1] > 0
        ]
        if len(set(static_sequence_lengths)) > 1:
            raise ValueError(
                "Prompt Guard ONNX inputs declare inconsistent fixed sequence lengths: "
                f"{static_sequence_lengths!r}"
            )
        contract_limit = min(static_sequence_lengths) if static_sequence_lengths else requested_window
        self.window_tokens = min(requested_window, position_limit, contract_limit)
        if self.window_tokens < 3:
            raise ValueError("ONNX input contract allows fewer than three sequence tokens")

        pad_id = self.model_config.get("pad_token_id")
        if not isinstance(pad_id, int):
            pad_id = self._special_token_id(
                id_field="pad_token_id",
                token_field="pad_token",
                candidates=("[PAD]", "<pad>"),
            )
        self.pad_token_id = pad_id if isinstance(pad_id, int) else 0

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read valid JSON from {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return payload

    @staticmethod
    def _normalize_label(value: object) -> str:
        return (
            str(value)
            .strip()
            .casefold()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @classmethod
    def _load_labels(
        cls, config: dict[str, Any]
    ) -> tuple[dict[int, str], int, int]:
        """Resolve class labels from model config; never assume class 1 is attack."""
        raw_labels = config.get("id2label")
        if isinstance(raw_labels, dict) and raw_labels:
            source_count = len(raw_labels)
            try:
                labels = {
                    int(index): cls._normalize_label(label)
                    for index, label in raw_labels.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError("Prompt Guard id2label contains invalid class IDs") from exc
        else:
            label_to_id = config.get("label2id")
            if not isinstance(label_to_id, dict) or not label_to_id:
                raise ValueError(
                    "Prompt Guard config must contain id2label or label2id"
                )
            source_count = len(label_to_id)
            try:
                labels = {
                    int(index): cls._normalize_label(label)
                    for label, index in label_to_id.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError("Prompt Guard label2id contains invalid class IDs") from exc

        if len(labels) != source_count:
            raise ValueError("Prompt Guard class mapping contains duplicate class IDs")
        if sorted(labels) != list(range(len(labels))):
            raise ValueError("Prompt Guard class IDs must be contiguous from zero")
        if len(labels) != 2:
            raise ValueError("Prompt Guard 2 must define exactly two classifier labels")

        attack_indices = [i for i, label in labels.items() if label in cls.ATTACK_LABELS]
        benign_indices = [i for i, label in labels.items() if label in cls.BENIGN_LABELS]
        if len(attack_indices) != 1 or len(benign_indices) != 1:
            raise ValueError(
                "Prompt Guard labels must identify exactly one benign class and "
                "one malicious/attack class; found "
                f"{labels!r}"
            )
        if attack_indices[0] == benign_indices[0]:
            raise ValueError("Prompt Guard benign and attack classes must differ")
        if config.get("num_labels", 2) != 2:
            raise ValueError("Prompt Guard config num_labels must be 2")
        return labels, attack_indices[0], benign_indices[0]

    def _special_token_id(
        self,
        *,
        id_field: str,
        token_field: str,
        candidates: tuple[str, ...],
    ) -> int | None:
        """Resolve a special token ID from model/tokenizer metadata or the vocab."""
        for source in (self.model_config, self.tokenizer_config):
            value = source.get(id_field)
            if isinstance(value, int):
                return value
            token_value = source.get(token_field)
            if isinstance(token_value, str):
                token_id = self.tokenizer.token_to_id(token_value)
                if token_id is not None:
                    return token_id
            elif isinstance(token_value, dict):
                content = token_value.get("content")
                if isinstance(content, str):
                    token_id = self.tokenizer.token_to_id(content)
                    if token_id is not None:
                        return token_id
        for candidate in candidates:
            token_id = self.tokenizer.token_to_id(candidate)
            if token_id is not None:
                return token_id
        return None

    @classmethod
    def _validate_inputs(cls, specs: list[Any]) -> list[Any]:
        if not specs:
            raise ValueError("Prompt Guard ONNX graph has no inputs")
        by_name: dict[str, Any] = {}
        for spec in specs:
            name = spec.name.casefold()
            if name not in cls.SUPPORTED_INPUTS:
                raise ValueError(f"Unsupported Prompt Guard ONNX input: {spec.name!r}")
            if name in by_name:
                raise ValueError(f"Duplicate ONNX input name after normalization: {name}")
            if len(spec.shape) != 2:
                raise ValueError(
                    f"Prompt Guard input {spec.name!r} must be rank 2; "
                    f"shape={spec.shape!r}"
                )
            batch_dim = spec.shape[0]
            if isinstance(batch_dim, int) and batch_dim not in {1}:
                raise ValueError(
                    f"Prompt Guard only supports batch size 1; input={spec.name!r}"
                )
            allowed_types = {"tensor(int32)", "tensor(int64)"}
            if name == "attention_mask":
                allowed_types.add("tensor(bool)")
            if spec.type not in allowed_types:
                raise ValueError(
                    f"Unsupported Prompt Guard input type for {spec.name!r}: "
                    f"{spec.type!r}"
                )
            by_name[name] = spec

        if not {"input_ids", "attention_mask"}.issubset(by_name):
            raise ValueError(
                "Prompt Guard ONNX graph must expose input_ids and attention_mask"
            )
        return specs

    @staticmethod
    def _select_output(specs: list[Any], class_count: int) -> tuple[str, str]:
        if not specs:
            raise ValueError("Prompt Guard ONNX graph has no outputs")
        candidates: list[Any] = []
        for spec in specs:
            if len(spec.shape) != 2:
                continue
            last_dim = spec.shape[1]
            if isinstance(last_dim, int) and last_dim != class_count:
                continue
            if spec.type not in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
                continue
            candidates.append(spec)
        if not candidates:
            raise ValueError(
                "Prompt Guard ONNX graph has no rank-2 floating-point classifier "
                f"output compatible with {class_count} labels"
            )
        named = [
            spec for spec in candidates
            if "logit" in spec.name.casefold() or "prob" in spec.name.casefold()
        ]
        if len(named) == 1:
            selected = named[0]
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            raise ValueError(
                "Prompt Guard ONNX output is ambiguous; candidate outputs are "
                f"{[spec.name for spec in candidates]!r}"
            )
        kind = "probabilities" if "prob" in selected.name.casefold() else "logits"
        return selected.name, kind

    @staticmethod
    def _numpy_dtype(onnx_type: str) -> Any:
        if onnx_type == "tensor(int32)":
            return np.int32
        if onnx_type == "tensor(int64)":
            return np.int64
        if onnx_type == "tensor(bool)":
            return np.bool_
        raise ValueError(f"Unsupported ONNX input dtype: {onnx_type}")

    @staticmethod
    def _softmax(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] != 2:
            raise GuardUnavailableError(
                "Prompt Guard output must have shape (1, 2); "
                f"found {values.shape!r}"
            )
        if not np.isfinite(values).all():
            raise GuardUnavailableError("Prompt Guard returned non-finite scores")
        shifted = values[0] - np.max(values[0])
        exp_values = np.exp(shifted)
        denominator = float(exp_values.sum())
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise GuardUnavailableError("Prompt Guard scores could not be normalized")
        return cast(np.ndarray[Any, Any], exp_values / denominator)

    @staticmethod
    def _validated_probabilities(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        probabilities = np.asarray(values, dtype=np.float64)
        if probabilities.shape != (1, 2):
            raise GuardUnavailableError(
                "Prompt Guard probability output must have shape (1, 2); "
                f"found {probabilities.shape!r}"
            )
        if not np.isfinite(probabilities).all():
            raise GuardUnavailableError("Prompt Guard returned non-finite probabilities")
        row = probabilities[0]
        if np.any(row < 0.0) or np.any(row > 1.0) or not math.isclose(
            float(row.sum()), 1.0, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise GuardUnavailableError("Prompt Guard returned an invalid probability vector")
        return cast(np.ndarray[Any, Any], row)

    def _prepare_feed(self, sequence_ids: list[int]) -> dict[str, np.ndarray[Any, Any]]:
        """Build input tensors from the observed metadata, including required padding."""
        target_lengths = [
            spec.shape[1]
            for spec in self.input_specs
            if isinstance(spec.shape[1], int) and spec.shape[1] > 0
        ]
        target_length = max(len(sequence_ids), min(target_lengths)) if target_lengths else len(sequence_ids)
        if target_lengths and len(sequence_ids) > min(target_lengths):
            raise GuardUnavailableError(
                "Prepared sequence exceeds the ONNX graph's fixed input length"
            )
        padding = target_length - len(sequence_ids)
        input_ids = sequence_ids + [self.pad_token_id] * padding
        attention_values = [1] * len(sequence_ids) + [0] * padding
        feed: dict[str, np.ndarray[Any, Any]] = {}

        for spec in self.input_specs:
            name = spec.name.casefold()
            dtype = self._numpy_dtype(spec.type)
            if name == "input_ids":
                values: list[int] | list[bool] = input_ids
            elif name == "attention_mask":
                values = attention_values
            elif name == "token_type_ids":
                values = [0] * target_length
            elif name == "position_ids":
                values = list(range(target_length))
            else:  # Guarded by _validate_inputs; protects future refactors.
                raise GuardUnavailableError("Unsupported Prompt Guard input signature")
            feed[spec.name] = np.asarray([values], dtype=dtype)
        return feed

    def _classify_window(self, content_ids: list[int]) -> float:
        sequence_ids = [self.cls_token_id, *content_ids, self.sep_token_id]
        feed = self._prepare_feed(sequence_ids)
        try:
            outputs = self.session.run([self.output_name], feed)
        except Exception as exc:
            raise GuardUnavailableError("Prompt Guard ONNX inference failed") from exc
        if len(outputs) != 1:
            raise GuardUnavailableError("Prompt Guard returned an unexpected output count")
        if self.output_kind == "probabilities":
            probabilities = self._validated_probabilities(np.asarray(outputs[0]))
        else:
            probabilities = self._softmax(np.asarray(outputs[0]))
        score = float(probabilities[self.attack_index])
        if not math.isfinite(score):
            raise GuardUnavailableError("Prompt Guard returned an invalid malicious score")
        return score

    def _failure_result(
        self,
        *,
        input_tokens: int,
        tokens_scanned: int,
        chunks_evaluated: int,
        truncated: bool,
    ) -> GuardResult:
        decision: GuardDecision = "allow" if self.failure_policy == "open" else "block"
        return GuardResult(
            decision=decision,
            label="guard_unavailable",
            score=None,
            input_tokens=input_tokens,
            tokens_scanned=tokens_scanned,
            chunks_evaluated=chunks_evaluated,
            truncated=truncated,
            reason="guard_inference_failed",
            error_code="guard_unavailable",
        )

    def classify(self, text: str) -> GuardResult:
        """Classify text and return the policy-adjusted typed decision.

        Runtime model/tokenizer failures become an explicit unavailable result;
        callers can audit ``reason``/``error_code`` and use ``blocked`` directly.
        Input-budget rejection is a distinct exception and is never fail-open.
        """
        if not isinstance(text, str):
            raise TypeError("Guard input must be a string")
        if not text.strip():
            return GuardResult(
                decision="allow",
                label=self.benign_label,
                score=0.0,
                input_tokens=0,
                tokens_scanned=0,
                chunks_evaluated=0,
                reason="empty_input",
            )

        try:
            encoding = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids = list(encoding.ids)
        except Exception:
            return self._failure_result(
                input_tokens=0,
                tokens_scanned=0,
                chunks_evaluated=0,
                truncated=False,
            )

        input_token_count = len(token_ids)
        truncated = False
        if input_token_count > self.max_input_tokens:
            if self.overlong_policy == "reject":
                raise GuardInputTooLongError(
                    f"Input has {input_token_count} tokens, exceeding the configured "
                    f"guard budget of {self.max_input_tokens}"
                )
            token_ids = token_ids[: self.max_input_tokens]
            truncated = True

        tokens_scanned = len(token_ids)
        if not token_ids:
            return GuardResult(
                decision="allow",
                label=self.benign_label,
                score=0.0,
                input_tokens=input_token_count,
                tokens_scanned=0,
                chunks_evaluated=0,
                truncated=truncated,
                reason="empty_tokenization",
            )

        content_window = self.window_tokens - 2  # reserve CLS and SEP
        if content_window < 1:
            raise GuardUnavailableError("Prompt Guard sequence window is too small")
        chunks = [
            token_ids[start : start + content_window]
            for start in range(0, len(token_ids), content_window)
        ]
        chunk_scores: list[float] = []
        try:
            for chunk in chunks:
                chunk_scores.append(self._classify_window(chunk))
        except Exception:
            # Do not return a partially scanned prompt as safe. Apply the policy
            # to the whole request if any chunk could not be evaluated.
            return self._failure_result(
                input_tokens=input_token_count,
                tokens_scanned=tokens_scanned,
                chunks_evaluated=len(chunk_scores),
                truncated=truncated,
            )

        # A prompt is as risky as its riskiest chunk. The full input must be
        # scanned; a failure in any chunk is handled above, not ignored.
        score = max(chunk_scores)
        blocked = score >= self.threshold
        return GuardResult(
            decision="block" if blocked else "allow",
            label=self.attack_label if blocked else self.benign_label,
            score=score,
            input_tokens=input_token_count,
            tokens_scanned=tokens_scanned,
            chunks_evaluated=len(chunk_scores),
            truncated=truncated,
            reason="threshold_exceeded" if blocked else "below_threshold",
        )
