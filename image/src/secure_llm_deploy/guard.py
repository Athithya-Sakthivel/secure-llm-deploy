"""CPU-only ONNX prompt-injection guard."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class GuardError(RuntimeError):
    """Base error for prompt-guard failures."""


class GuardUnavailableError(GuardError):
    """The guard could not produce a trustworthy classification."""


class GuardInputTooLongError(GuardError):
    """The input exceeded the configured token budget."""


class GuardResult:
    """Result returned by the prompt-injection classifier."""

    def __init__(
        self,
        *,
        blocked: bool,
        label: str,
        score: float,
        input_tokens: int,
    ) -> None:
        self.blocked = blocked
        self.label = label
        self.score = score
        self.input_tokens = input_tokens


class ONNXPromptGuard:
    """Run a local ONNX prompt-injection classifier on CPU only."""

    ATTACK_LABELS = {
        "attack",
        "injection",
        "jailbreak",
        "prompt_injection",
    }

    SUPPORTED_INPUT_NAMES = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    }

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        label_mapping_path: Path,
        *,
        threshold: float = 0.5,
        max_input_tokens: int = 8192,
        intra_op_threads: int = 4,
    ) -> None:
        """Load and validate the guard artifacts and CPU inference session."""
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Guard threshold must be between 0 and 1")

        if max_input_tokens < 1:
            raise ValueError("Guard maximum input tokens must be positive")

        if intra_op_threads < 1:
            raise ValueError("Guard intra-op thread count must be positive")

        required_files = (
            model_path,
            tokenizer_path,
            label_mapping_path,
        )
        for path in required_files:
            if not path.is_file():
                raise ValueError(
                    f"Required guard artifact is missing: {path.name}"
                )

        self.threshold = threshold
        self.max_input_tokens = max_input_tokens
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.id_to_label = self._load_labels(label_mapping_path)
        self.attack_index = self._find_attack_index(self.id_to_label)

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        providers = self.session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError(
                "The guard must use CPUExecutionProvider only"
            )

        self.input_specs = self.session.get_inputs()
        if not self.input_specs:
            raise ValueError("Guard ONNX model has no inputs")

        input_names = {item.name for item in self.input_specs}
        required_input_names = {"input_ids", "attention_mask"}

        if not required_input_names.issubset(input_names):
            raise ValueError("Unsupported ONNX input signature")

        input_types = {item.name: item.type for item in self.input_specs}

        if not input_types.keys() <= self.SUPPORTED_INPUT_NAMES:
            raise ValueError(
                "Guard ONNX model requires unsupported input fields"
            )

        for input_type in input_types.values():
            if input_type not in {"tensor(int32)", "tensor(int64)"}:
                raise ValueError(
                    "Guard ONNX input type must be int32 or int64"
                )

        self.input_types = input_types

        output_specs = self.session.get_outputs()
        if not output_specs:
            raise ValueError("Guard ONNX model has no outputs")

        output_names = [item.name for item in output_specs]
        self.logits_name = next(
            (
                name
                for name in output_names
                if "logit" in name.lower()
            ),
            output_names[0],
        )

        pad_token_id = self.tokenizer.token_to_id("[PAD]")
        if pad_token_id is None:
            pad_token_id = self.tokenizer.token_to_id("<pad>")
        if pad_token_id is None:
            pad_token_id = 0

        self.pad_token_id = pad_token_id

    @staticmethod
    def _normalize_label(label: object) -> str:
        """Normalize common label spelling variants."""
        return str(label).strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _load_labels(cls, mapping_path: Path) -> dict[int, str]:
        """Read and validate the classifier's class-to-label mapping."""
        try:
            with mapping_path.open("r", encoding="utf-8") as file:
                mapping: object = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Guard label mapping could not be loaded") from exc

        if not isinstance(mapping, dict):
            raise ValueError("Guard label mapping must be a JSON object")

        raw_labels = mapping.get("id2label") or mapping.get("idx_to_label")

        if not isinstance(raw_labels, dict) or not raw_labels:
            label_to_id = mapping.get("label2id") or mapping.get(
                "label_to_idx"
            )

            if not isinstance(label_to_id, dict) or not label_to_id:
                raise ValueError("Guard label mapping is invalid")

            raw_labels = {
                str(index): label
                for label, index in label_to_id.items()
            }

        try:
            id_to_label = {
                int(index): cls._normalize_label(label)
                for index, label in raw_labels.items()
            }
        except (TypeError, ValueError):
            raise ValueError(
                "Guard label mapping has invalid class indices"
            ) from None

        if len(id_to_label) != len(raw_labels):
            raise ValueError("Guard label mapping has duplicate class indices")

        if sorted(id_to_label) != list(range(len(id_to_label))):
            raise ValueError(
                "Guard class indices must be contiguous from zero"
            )

        if any(not label for label in id_to_label.values()):
            raise ValueError("Guard labels cannot be empty")

        return id_to_label

    @classmethod
    def _find_attack_index(
        cls,
        id_to_label: dict[int, str],
    ) -> int:
        """Require one unambiguous attack class."""
        attack_indices = [
            index
            for index, label in id_to_label.items()
            if label in cls.ATTACK_LABELS
        ]

        if len(attack_indices) != 1:
            raise ValueError(
                "Guard must define exactly one recognized attack label"
            )

        return attack_indices[0]

    @staticmethod
    def _numpy_type(onnx_type: str) -> Any:
        """Map supported ONNX integer tensor types to NumPy types."""
        if onnx_type == "tensor(int32)":
            return np.int32
        if onnx_type == "tensor(int64)":
            return np.int64

        raise ValueError("Unsupported ONNX input type")

    def _prepare_feed(
        self,
        token_ids: list[int],
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Prepare token tensors, respecting fixed ONNX input dimensions."""
        sequence_length = len(token_ids)
        target_length = sequence_length

        for spec in self.input_specs:
            if len(spec.shape) <= 1:
                continue

            sequence_dimension = spec.shape[1]
            if not isinstance(sequence_dimension, int):
                continue

            if sequence_dimension < sequence_length:
                raise GuardInputTooLongError(
                    "Input exceeds the ONNX model's fixed sequence length"
                )

            target_length = max(target_length, sequence_dimension)

        padding = target_length - sequence_length

        ids = np.asarray(
            [token_ids + [self.pad_token_id] * padding],
            dtype=np.int64,
        )
        mask = np.asarray(
            [[1] * sequence_length + [0] * padding],
            dtype=np.int64,
        )

        feed: dict[str, np.ndarray[Any, Any]] = {}

        for spec in self.input_specs:
            dtype = self._numpy_type(spec.type)

            if spec.name == "input_ids":
                value = ids
            elif spec.name == "attention_mask":
                value = mask
            elif spec.name == "token_type_ids":
                value = np.zeros_like(ids)
            elif spec.name == "position_ids":
                value = np.asarray(
                    [list(range(target_length))],
                    dtype=np.int64,
                )
            else:
                raise GuardUnavailableError(
                    "Unsupported ONNX input signature"
                )

            feed[spec.name] = value.astype(dtype, copy=False)

        return feed

    @staticmethod
    def _probabilities(
        values: np.ndarray[Any, Any],
        output_name: str,
    ) -> np.ndarray[Any, Any]:
        """Validate probability outputs or apply softmax to classifier logits."""
        scores = np.asarray(values, dtype=np.float64).reshape(-1)

        if scores.size < 2 or not np.all(np.isfinite(scores)):
            raise GuardUnavailableError(
                "Guard returned invalid classifier scores"
            )

        if "prob" in output_name.lower():
            total = float(np.sum(scores))

            if (
                np.any(scores < 0.0)
                or np.any(scores > 1.0)
                or not math.isclose(
                    total,
                    1.0,
                    rel_tol=1e-3,
                    abs_tol=1e-3,
                )
            ):
                raise GuardUnavailableError(
                    "Guard returned invalid probability scores"
                )

            return scores

        # Treat non-probability outputs as classifier logits.
        shifted = scores - np.max(scores)
        exponentials = np.exp(shifted)
        denominator = float(np.sum(exponentials))

        if not math.isfinite(denominator) or denominator <= 0.0:
            raise GuardUnavailableError(
                "Guard returned invalid classifier scores"
            )

        probabilities = exponentials / denominator

        if not np.all(np.isfinite(probabilities)):
            raise GuardUnavailableError(
                "Guard returned invalid classifier probabilities"
            )

        return np.asarray(probabilities, dtype=np.float64)

    def classify(self, text: str) -> GuardResult:
        """Classify text; failures raise an error so callers can fail closed."""
        if not isinstance(text, str):
            raise GuardUnavailableError("Guard input must be text")

        if not text.strip():
            return GuardResult(
                blocked=False,
                label="benign",
                score=0.0,
                input_tokens=0,
            )

        try:
            encoding = self.tokenizer.encode(
                text,
                add_special_tokens=True,
            )
            token_ids = encoding.ids

            if len(token_ids) > self.max_input_tokens:
                raise GuardInputTooLongError(
                    "Input exceeds the configured guard token limit"
                )

            if not token_ids:
                return GuardResult(
                    blocked=False,
                    label="benign",
                    score=0.0,
                    input_tokens=0,
                )

            feed = self._prepare_feed(token_ids)
            outputs = self.session.run([self.logits_name], feed)

            if not outputs:
                raise GuardUnavailableError(
                    "Guard returned no classifier output"
                )

            scores = self._probabilities(
                outputs[0],
                self.logits_name,
            )

            if scores.size != len(self.id_to_label):
                raise GuardUnavailableError(
                    "Guard score count does not match the label mapping"
                )

            score = float(scores[self.attack_index])
            if not math.isfinite(score):
                raise GuardUnavailableError(
                    "Guard returned a non-finite score"
                )

            blocked = score >= self.threshold

            return GuardResult(
                blocked=blocked,
                label="jailbreak" if blocked else "benign",
                score=score,
                input_tokens=len(token_ids),
            )

        except GuardError:
            raise
        except Exception:
            # Never include the prompt, tensor contents, or raw inference error.
            raise GuardUnavailableError("Guard inference failed") from None