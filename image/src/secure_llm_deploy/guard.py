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
    ATTACK_LABELS = {
        "jailbreak",
        "attack",
        "prompt_injection",
        "prompt injection",
        "injection",
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
        self.threshold = threshold
        self.max_input_tokens = max_input_tokens
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        with label_mapping_path.open("r", encoding="utf-8") as file:
            mapping: dict[str, Any] = json.load(file)

        raw_labels = mapping.get("id2label") or mapping.get("idx_to_label")
        if not isinstance(raw_labels, dict) or not raw_labels:
            label_to_id = mapping.get("label2id") or mapping.get("label_to_idx")
            if not isinstance(label_to_id, dict) or not label_to_id:
                raise ValueError("Guard label mapping is invalid")
            raw_labels = {str(index): label for label, index in label_to_id.items()}

        try:
            self.id_to_label = {
                int(index): str(label).strip().lower()
                for index, label in raw_labels.items()
            }
        except (TypeError, ValueError):
            raise ValueError("Guard label mapping has invalid class indices") from None

        if sorted(self.id_to_label) != list(range(len(self.id_to_label))):
            raise ValueError("Guard class indices must be contiguous from zero")

        attack_indices = [
            index
            for index, label in self.id_to_label.items()
            if label in self.ATTACK_LABELS
        ]
        if len(attack_indices) != 1:
            raise ValueError("Guard must define exactly one recognized attack label")

        self.attack_index = attack_indices[0]

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("The guard must use CPUExecutionProvider only")

        self.input_specs = self.session.get_inputs()
        input_names = {item.name for item in self.input_specs}

        if "input_ids" not in input_names or "attention_mask" not in input_names:
            raise ValueError("Unsupported ONNX input signature")

        output_names = [item.name for item in self.session.get_outputs()]
        logits_name = next(
            (name for name in output_names if "logit" in name.lower()),
            output_names[0] if output_names else None,
        )
        if logits_name is None:
            raise ValueError("Guard ONNX model has no outputs")

        self.logits_name = logits_name

        input_types = {item.name: item.type for item in self.input_specs}
        supported_names = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
        }
        if not input_types.keys() <= supported_names:
            raise ValueError("Guard ONNX model requires unsupported input fields")

        for input_type in input_types.values():
            if input_type not in {"tensor(int32)", "tensor(int64)"}:
                raise ValueError("Guard ONNX input type must be int32 or int64")

        self.input_types = input_types
        self.pad_token_id = (
            self.tokenizer.token_to_id("[PAD]")
            or self.tokenizer.token_to_id("<pad>")
            or 0
        )

    @staticmethod
    def _numpy_type(onnx_type: str) -> Any:
        if onnx_type == "tensor(int32)":
            return np.int32
        if onnx_type == "tensor(int64)":
            return np.int64
        raise ValueError("Unsupported ONNX input type")

    def _prepare_feed(self, token_ids: list[int]) -> dict[str, np.ndarray[Any, Any]]:
        sequence_length = len(token_ids)
        target_length = sequence_length

        for spec in self.input_specs:
            if len(spec.shape) > 1 and isinstance(spec.shape[1], int):
                fixed_length = spec.shape[1]
                if fixed_length < sequence_length:
                    raise GuardInputTooLongError(
                        "Input exceeds the ONNX model's fixed sequence length"
                    )
                target_length = max(target_length, fixed_length)

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
                raise GuardUnavailableError("Unsupported ONNX input signature")

            feed[spec.name] = value.astype(dtype, copy=False)

        return feed

    @staticmethod
    def _probabilities(
        values: np.ndarray[Any, Any],
        output_name: str,
    ) -> np.ndarray[Any, Any]:
        scores = np.asarray(values, dtype=np.float64).reshape(-1)

        if scores.size < 2 or not np.all(np.isfinite(scores)):
            raise GuardUnavailableError("Guard returned invalid classifier scores")

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

        # Treat explicitly named logits as logits.
        shifted = scores - np.max(scores)
        exponentials = np.exp(shifted)
        denominator = float(np.sum(exponentials))

        if not math.isfinite(denominator) or denominator <= 0.0:
            raise GuardUnavailableError(
                "Guard returned invalid classifier scores"
            )

        return np.asarray(
            exponentials / denominator,
            dtype=np.float64,
        )


    def classify(self, text: str) -> GuardResult:
        if not text.strip():
            return GuardResult(
                blocked=False,
                label="benign",
                score=0.0,
                input_tokens=0,
            )

        try:
            encoding = self.tokenizer.encode(text, add_special_tokens=True)
            token_ids = encoding.ids

            if len(token_ids) > self.max_input_tokens:
                raise GuardInputTooLongError(
                    "Input exceeds the configured guard token limit"
                )

            feed = self._prepare_feed(token_ids)
            outputs = self.session.run([self.logits_name], feed)
            scores = self._probabilities(outputs[0], self.logits_name)

            if scores.size != len(self.id_to_label):
                raise GuardUnavailableError(
                    "Guard score count does not match the label mapping"
                )

            score = float(scores[self.attack_index])
            if not math.isfinite(score):
                raise GuardUnavailableError("Guard returned a non-finite score")

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
            # Do not expose the input text, tensor values, or raw model exception.
            raise GuardUnavailableError("Guard inference failed") from None