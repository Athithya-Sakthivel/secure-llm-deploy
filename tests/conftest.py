from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from secure_llm_deploy.config import Settings

SETTINGS_ENV_ALIASES = {
    "VLLM_MAX_MODEL_LEN",
    "MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "GPU_MEMORY_UTILIZATION",
    "VLLM_NUM_SPECULATIVE_TOKENS",
    "SPECULATIVE_TOKENS",
}

SETTINGS_ENV_FIELDS = {
    "APP_NAME",
    "GATEWAY_HOST",
    "GATEWAY_PORT",
    "GATEWAY_API_KEY",
    "VLLM_API_KEY",
    "SERVE_MODE",
    "SERVED_MODEL_NAME",
    "TARGET_MODEL_PATH",
    "TARGET_TOKENIZER_PATH",
    "EAGLE_MODEL_PATH",
    "GUARD_MODEL_PATH",
    "GUARD_TOKENIZER_PATH",
    "GUARD_CONFIG_PATH",
    "GUARD_THRESHOLD",
    "GUARD_MAX_INPUT_TOKENS",
    "GUARD_WINDOW_TOKENS",
    "GUARD_INTRA_OP_THREADS",
    "GUARD_FAILURE_POLICY",
    "GUARD_OVERLONG_POLICY",
    "MAX_CONCURRENT_GUARD_EVALUATIONS",
    "REQUEST_MAX_BODY_BYTES",
    "MAX_MESSAGES",
    "MAX_MESSAGE_CHARS",
    "MAX_INPUT_CHARS",
    "MAX_INPUT_TOKENS",
    "DEFAULT_MAX_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "MAX_MODEL_LEN",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "RATE_LIMIT_MAX_CLIENTS",
    "QUEUE_WAIT_TIMEOUT_SECONDS",
    "MAX_CONCURRENT_INFERENCE_REQUESTS",
    "VLLM_EXECUTABLE",
    "VLLM_HOST",
    "VLLM_PORT",
    "VLLM_STARTUP_TIMEOUT_SECONDS",
    "VLLM_STARTUP_POLL_SECONDS",
    "VLLM_CONNECT_TIMEOUT_SECONDS",
    "VLLM_READ_TIMEOUT_SECONDS",
    "VLLM_WRITE_TIMEOUT_SECONDS",
    "VLLM_POOL_TIMEOUT_SECONDS",
    "VLLM_MAX_NUM_SEQS",
    "GPU_MEMORY_UTILIZATION",
    "SPECULATIVE_TOKENS",
    "SENSITIVE_STRINGS",
    "MAX_SENSITIVE_STRING_CHARS",
}


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent environment variables from leaking into settings tests."""
    for name in SETTINGS_ENV_FIELDS | SETTINGS_ENV_ALIASES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def artifacts(tmp_path: Path) -> dict[str, Path]:
    """Create placeholder artifacts for runtime settings tests; no model is loaded."""
    models_dir = tmp_path / "models"
    target_dir = models_dir / "target"
    eagle_dir = models_dir / "eagle"
    guard_dir = models_dir / "guard"

    for directory in (target_dir, eagle_dir, guard_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tokenizer_data: dict[str, Any] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "[UNK]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[PAD]": 3,
                "test": 4,
            },
            "unk_token": "[UNK]",
        },
    }

    target_config = target_dir / "config.json"
    target_config.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "max_position_embeddings": 40960,
                "vocab_size": 151936,
            }
        ),
        encoding="utf-8",
    )

    target_tokenizer = target_dir / "tokenizer.json"
    target_tokenizer.write_text(json.dumps(tokenizer_data), encoding="utf-8")

    eagle_config = eagle_dir / "config.json"
    eagle_config.write_text(
        json.dumps(
            {
                "architectures": ["Eagle3DraftModel"],
                "speculators_model_type": "eagle3",
                "eagle31": True,
                "speculators_config": {"algorithm": "eagle3"},
                "auto_map": {"AutoConfig": "config.EagleConfig"},
                "draft_vocab_size": 151936,
                "vocab_size": 151936,
            }
        ),
        encoding="utf-8",
    )

    eagle_python = eagle_dir / "config.py"
    eagle_python.write_text(
        "# Placeholder only; never imported by these tests.\n",
        encoding="utf-8",
    )

    # Non-empty fake weights satisfy path validation only; never used for inference.
    eagle_weights = eagle_dir / "model.safetensors"
    eagle_weights.write_bytes(b"test-only-placeholder-eagle-weights")

    guard_model = guard_dir / "model.QUInt8.quant.onnx"
    guard_model.write_bytes(b"test-only-placeholder-onnx")

    guard_tokenizer = guard_dir / "tokenizer.json"
    guard_tokenizer.write_text(json.dumps(tokenizer_data), encoding="utf-8")

    guard_config = guard_dir / "config.json"
    guard_config.write_text(
        json.dumps(
            {
                "architectures": ["DebertaV2ForSequenceClassification"],
                "model_type": "deberta-v2",
                "num_labels": 2,
                "max_position_embeddings": 512,
                "pad_token_id": 0,
                "id2label": {"0": "BENIGN", "1": "MALICIOUS"},
                "label2id": {"BENIGN": 0, "MALICIOUS": 1},
            }
        ),
        encoding="utf-8",
    )

    guard_tokenizer_config = guard_dir / "tokenizer_config.json"
    guard_tokenizer_config.write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
            }
        ),
        encoding="utf-8",
    )

    special_tokens = guard_dir / "special_tokens_map.json"
    special_tokens.write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
            }
        ),
        encoding="utf-8",
    )

    guard_license = guard_dir / "LICENSE"
    guard_license.write_text("test fixture only\n", encoding="utf-8")

    guard_notice = guard_dir / "NOTICE"
    guard_notice.write_text("test fixture only\n", encoding="utf-8")

    return {
        "models_dir": models_dir,
        "target_dir": target_dir,
        "target_config": target_config,
        "target_tokenizer": target_tokenizer,
        "eagle_dir": eagle_dir,
        "eagle_config": eagle_config,
        "eagle_python": eagle_python,
        "eagle_weights": eagle_weights,
        "guard_dir": guard_dir,
        "guard_model": guard_model,
        "guard_tokenizer": guard_tokenizer,
        "guard_config": guard_config,
        "guard_tokenizer_config": guard_tokenizer_config,
        "guard_special_tokens": special_tokens,
        "guard_license": guard_license,
        "guard_notice": guard_notice,
    }


@pytest.fixture
def model_artifacts(artifacts: dict[str, Path]) -> dict[str, Path]:
    """Compatibility alias used by model-artifact tests."""
    return artifacts


@pytest.fixture
def valid_settings(artifacts: dict[str, Path]) -> Settings:
    """Create settings backed by temporary placeholder artifacts."""
    return Settings(
        gateway_api_key=SecretStr("test-gateway-api-key-00000000000000000000"),
        vllm_api_key=SecretStr("test-vllm-api-key-000000000000000000000000"),
        serve_mode="baseline",
        target_model_path=artifacts["target_dir"],
        target_tokenizer_path=artifacts["target_tokenizer"],
        eagle_model_path=artifacts["eagle_dir"],
        guard_model_path=artifacts["guard_model"],
        guard_tokenizer_path=artifacts["guard_tokenizer"],
        guard_config_path=artifacts["guard_config"],
        sensitive_strings="SYNTHETIC_SECRET_ALPHA_001",
    )


@pytest.fixture
def auth_headers(valid_settings: Settings) -> dict[str, str]:
    """Provide valid client-facing authentication headers."""
    return {"Authorization": f"Bearer {valid_settings.gateway_key}"}


@pytest.fixture
def valid_request(valid_settings: Settings) -> dict[str, Any]:
    """Provide a valid OpenAI-compatible chat request."""
    return {
        "model": valid_settings.served_model_name,
        "messages": [{"role": "user", "content": "Hello."}],
        "max_tokens": 16,
    }


class FakeEncoding:
    """Minimal tokenizer encoding compatible with the gateway's token counter."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """Deterministic whitespace tokenizer for gateway tests."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> FakeEncoding:
        del add_special_tokens
        return FakeEncoding(list(range(len(text.split()))))


class FakeGuardResult:
    """Result compatible with the production GuardResult interface."""

    def __init__(
        self,
        *,
        blocked: bool,
        score: float,
        input_tokens: int,
        label: str | None = None,
    ) -> None:
        self.blocked = blocked
        self.score = score
        self.input_tokens = input_tokens
        self.tokens_scanned = input_tokens
        self.chunks_evaluated = 1 if input_tokens else 0
        self.truncated = False
        self.error_code: str | None = None
        self.reason = "threshold_exceeded" if blocked else "below_threshold"
        self.label = label or ("prompt_injection" if blocked else "benign")

    @property
    def allowed(self) -> bool:
        return not self.blocked


class FakeGuard:
    """Controllable guard fake; no ONNX session is loaded."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.calls: list[str] = []
        self.blocked = False
        self.score = 0.0
        self.exception: Exception | None = None
        self.ready = True
        self.is_ready = True
        self.tokenizer = FakeTokenizer()

    @staticmethod
    def _contains_attack_phrase(text: str) -> bool:
        normalized = text.casefold()
        indicators = (
            "ignore all previous instructions",
            "ignore previous instructions",
            "ignore all prior instructions",
            "reveal your system prompt",
            "show me your system prompt",
            "bypass your safety rules",
        )
        return any(value in normalized for value in indicators)

    def classify(self, text: str) -> FakeGuardResult:
        self.calls.append(text)
        if self.exception is not None:
            raise self.exception

        phrase_detected = self._contains_attack_phrase(text)
        is_blocked = self.blocked or phrase_detected
        result_score = 1.0 if phrase_detected and not self.blocked else self.score
        input_tokens = max(1, len(text.split()))
        return FakeGuardResult(
            blocked=is_blocked,
            score=result_score,
            input_tokens=input_tokens,
        )

    def evaluate(self, text: str) -> FakeGuardResult:
        return self.classify(text)

    def check(self, text: str) -> FakeGuardResult:
        return self.classify(text)

    def inspect(self, text: str) -> FakeGuardResult:
        return self.classify(text)

    def __call__(self, text: str) -> FakeGuardResult:
        return self.classify(text)


class FakeStreamingResponse:
    """Mutable line-based fake matching the streaming handler interface."""

    def __init__(
        self,
        lines: list[str] | None = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.lines = list(lines or [])
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}
        self.closed = False
        self.is_success = 200 <= status_code < 300

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8")

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self.lines:
            yield line

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        content = self.content
        if not content:
            return
        size = chunk_size or max(1, len(content))
        for offset in range(0, len(content), size):
            yield content[offset : offset + size]

    async def aread(self) -> bytes:
        return self.content

    async def aclose(self) -> None:
        self.closed = True


def _default_stream_lines() -> list[str]:
    """Return a valid, single-message SSE response."""
    chunk = {
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "secure-llm-deploy",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello from the fake model."},
                "finish_reason": None,
            }
        ],
    }
    final_chunk = {
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "secure-llm-deploy",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return [
        f"data: {json.dumps(chunk, separators=(',', ':'))}",
        "",
        f"data: {json.dumps(final_chunk, separators=(',', ':'))}",
        "",
        "data: [DONE]",
        "",
    ]


class FakeUpstream:
    """Controllable fake implementing methods used by main.py."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.calls: list[dict[str, Any]] = []
        self.request_ids: list[str] = []
        self.response_payload: dict[str, Any] = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "secure-llm-deploy",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from the fake model.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self.completion_error: Exception | None = None
        self.stream_open_error: Exception | None = None
        self.ready = True
        self.is_ready = True
        self.closed = False
        self.stream_response = FakeStreamingResponse(_default_stream_lines())

    async def wait_until_ready(self, *, process_is_running: Any) -> None:
        if not process_is_running():
            raise RuntimeError("vLLM exited during startup")
        self.ready = True
        self.is_ready = True

    def _record_call(self, payload: dict[str, Any], request_id: str) -> None:
        self.calls.append(payload)
        self.request_ids.append(request_id)

    async def chat_completion(
        self, payload: dict[str, Any], *, request_id: str
    ) -> dict[str, Any]:
        self._record_call(payload, request_id)
        if self.completion_error is not None:
            raise self.completion_error
        return self.response_payload

    async def open_chat_completion_stream(
        self, payload: dict[str, Any], *, request_id: str
    ) -> FakeStreamingResponse:
        self._record_call(payload, request_id)
        if self.stream_open_error is not None:
            raise self.stream_open_error
        return self.stream_response

    async def aclose(self) -> None:
        self.closed = True

    async def close(self) -> None:
        await self.aclose()


class FakeVLLMProcess:
    """Mutable fake process for startup and health tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._running = False
        self.pid = 12345
        self.returncode: int | None = None

    @property
    def running(self) -> bool:
        return self._running

    @running.setter
    def running(self, value: bool) -> None:
        self._running = value
        if value:
            self.returncode = None
        elif self.returncode is None:
            self.returncode = 1

    def is_running(self) -> bool:
        return self._running and self.returncode is None

    async def start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.running = True

    async def stop(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._running = False
        if self.returncode is None:
            self.returncode = 0

    async def terminate(self, *args: Any, **kwargs: Any) -> None:
        await self.stop()

    async def wait(self, *args: Any, **kwargs: Any) -> int:
        del args, kwargs
        self._running = False
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.fixture
def fake_guard() -> FakeGuard:
    """Expose the guard fake so tests can configure its behavior."""
    return FakeGuard()


@pytest.fixture
def fake_upstream() -> FakeUpstream:
    """Expose the upstream fake so tests can change responses and errors."""
    return FakeUpstream()


@pytest.fixture
def fake_process() -> FakeVLLMProcess:
    """Expose the process fake for readiness and liveness tests."""
    return FakeVLLMProcess()


@pytest.fixture
def app(
    valid_settings: Settings,
    fake_guard: FakeGuard,
    fake_upstream: FakeUpstream,
    fake_process: FakeVLLMProcess,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Build the FastAPI app using shared fake runtime instances."""
    import secure_llm_deploy.main as main_module

    monkeypatch.setattr(
        main_module,
        "ONNXPromptGuard",
        lambda *args, **kwargs: fake_guard,
    )
    monkeypatch.setattr(
        main_module,
        "VLLMUpstream",
        lambda *args, **kwargs: fake_upstream,
    )
    monkeypatch.setattr(
        main_module,
        "VLLMProcess",
        lambda *args, **kwargs: fake_process,
    )
    return main_module.create_app(valid_settings)


@pytest.fixture
def client(app: Any) -> Iterator[TestClient]:
    """Run the FastAPI lifespan inside a test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_client(client: TestClient) -> TestClient:
    """Compatibility alias for tests using test_client."""
    return client