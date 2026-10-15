"""Shared CPU-only fixtures for secure-llm-deploy tests."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import AsyncIterator, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from secure_llm_deploy.config import Settings
from secure_llm_deploy.guard import GuardResult
from secure_llm_deploy.upstream import UpstreamError

GATEWAY_TEST_KEY = "gateway-test-key-" + ("g" * 40)
VLLM_TEST_KEY = "vllm-test-key-" + ("v" * 40)


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent workstation or CI environment variables from changing tests."""
    setting_names = {
        field_name.casefold()
        for field_name in Settings.model_fields
    }

    for environment_name in tuple(os.environ):
        if environment_name.casefold() in setting_names:
            monkeypatch.delenv(environment_name, raising=False)


@pytest.fixture
def model_artifacts(tmp_path: Path) -> dict[str, Path]:
    """Create placeholder artifacts required by Settings.validate_runtime.

    These are deliberately not usable model files. The test runtime replaces
    model initialization before the application lifespan starts.
    """
    target_dir = tmp_path / "models" / "target-model"
    draft_dir = tmp_path / "models" / "draft-model"
    guard_dir = tmp_path / "models" / "vela-guard"
    guard_onnx_dir = guard_dir / "onnx"

    target_dir.mkdir(parents=True)
    draft_dir.mkdir(parents=True)
    guard_onnx_dir.mkdir(parents=True)

    target_config = target_dir / "config.json"
    target_tokenizer = target_dir / "tokenizer.json"
    draft_config = draft_dir / "config.json"
    guard_model = guard_onnx_dir / "model.onnx"
    guard_data = guard_onnx_dir / "model.onnx.data"
    guard_tokenizer = guard_dir / "tokenizer.json"
    guard_label_mapping = guard_dir / "label_mapping.json"

    target_config.write_text("{}", encoding="utf-8")
    target_tokenizer.write_text("{}", encoding="utf-8")
    draft_config.write_text("{}", encoding="utf-8")
    guard_model.write_bytes(b"placeholder-onnx-model")
    guard_data.write_bytes(b"placeholder-onnx-external-data")
    guard_tokenizer.write_text("{}", encoding="utf-8")
    guard_label_mapping.write_text(
        json.dumps(
            {
                "id2label": {
                    "0": "benign",
                    "1": "jailbreak",
                }
            }
        ),
        encoding="utf-8",
    )

    return {
        "target_dir": target_dir,
        "target_config": target_config,
        "target_tokenizer": target_tokenizer,
        "draft_dir": draft_dir,
        "draft_config": draft_config,
        "guard_dir": guard_dir,
        "guard_model": guard_model,
        "guard_data": guard_data,
        "guard_tokenizer": guard_tokenizer,
        "guard_label_mapping": guard_label_mapping,
    }


@pytest.fixture
def valid_settings(model_artifacts: dict[str, Path]) -> Settings:
    """Return valid settings pointing to isolated placeholder artifacts."""
    return Settings(
        gateway_api_key=SecretStr(GATEWAY_TEST_KEY),
        vllm_api_key=SecretStr(VLLM_TEST_KEY),
        serve_mode="baseline",
        served_model_name="secure-llm-deploy-test",
        target_model_path=model_artifacts["target_dir"],
        target_tokenizer_path=model_artifacts["target_tokenizer"],
        draft_model_path=model_artifacts["draft_dir"],
        guard_model_path=model_artifacts["guard_model"],
        guard_tokenizer_path=model_artifacts["guard_tokenizer"],
        guard_label_mapping_path=model_artifacts["guard_label_mapping"],
        sensitive_strings="SYNTHETIC_SECRET_ALPHA_001",
    )


class FakeGuard:
    """Deterministic replacement for ONNXPromptGuard."""

    def __init__(self) -> None:
        self.blocked = False
        self.score = 0.01
        self.input_tokens = 12
        self.exception: Exception | None = None
        self.calls: list[str] = []

    def classify(self, text: str) -> GuardResult:
        self.calls.append(text)

        if self.exception is not None:
            raise self.exception

        return GuardResult(
            blocked=self.blocked,
            label="jailbreak" if self.blocked else "benign",
            score=self.score,
            input_tokens=self.input_tokens,
        )


class FakeStreamingResponse:
    """Small async iterable compatible with the upstream streaming interface."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines if lines is not None else make_default_sse_lines()
        self.closed = False

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self.lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


def make_default_sse_lines(
    *,
    model: str = "secure-llm-deploy-test",
    content: str = "Hello from the fake model.",
) -> list[str]:
    """Build an ordinary OpenAI-compatible SSE sequence."""
    header = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": model,
    }

    events = [
        {
            **header,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        },
        {
            **header,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        },
        {
            **header,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    lines: list[str] = []
    for event in events:
        lines.extend(
            [
                f"data: {json.dumps(event, separators=(',', ':'))}",
                "",
            ]
        )

    lines.extend(["data: [DONE]", ""])
    return lines


class FakeUpstream:
    """In-memory replacement for VLLMUpstream."""

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[dict[str, Any]] = []
        self.request_ids: list[str] = []
        self.completion_error: UpstreamError | None = None
        self.stream_open_error: UpstreamError | None = None
        self.readiness_error: UpstreamError | None = None
        self.response_payload: dict[str, Any] = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "secure-llm-deploy-test",
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
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 6,
                "total_tokens": 18,
            },
        }
        self.stream_response = FakeStreamingResponse()

    async def wait_until_ready(
        self,
        *,
        process_is_running: Any,
    ) -> None:
        if self.readiness_error is not None:
            raise self.readiness_error

        if not process_is_running():
            raise UpstreamError("Fake process is not running")

    async def aclose(self) -> None:
        self.closed = True

    async def chat_completion(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> dict[str, Any]:
        self.calls.append(deepcopy(payload))
        self.request_ids.append(request_id)

        if self.completion_error is not None:
            raise self.completion_error

        return deepcopy(self.response_payload)

    async def open_chat_completion_stream(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> FakeStreamingResponse:
        self.calls.append(deepcopy(payload))
        self.request_ids.append(request_id)

        if self.stream_open_error is not None:
            raise self.stream_open_error

        return self.stream_response


class FakeVLLMProcess:
    """No-op process manager with an explicitly controlled running state."""

    def __init__(self) -> None:
        self.running = False
        self.start_calls = 0
        self.stop_calls = 0
        self.start_error: Exception | None = None

    async def start(self) -> None:
        self.start_calls += 1

        if self.start_error is not None:
            raise self.start_error

        self.running = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def is_running(self) -> bool:
        return self.running


@pytest.fixture
def fake_guard() -> FakeGuard:
    return FakeGuard()


@pytest.fixture
def fake_upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def fake_process() -> FakeVLLMProcess:
    return FakeVLLMProcess()


@pytest.fixture
def app(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
    fake_guard: FakeGuard,
    fake_upstream: FakeUpstream,
    fake_process: FakeVLLMProcess,
) -> FastAPI:
    """Create an app with its production dependencies replaced by fakes."""
    main_module = importlib.import_module("secure_llm_deploy.main")

    fake_upstream.response_payload["model"] = valid_settings.served_model_name
    fake_upstream.stream_response = FakeStreamingResponse(
        make_default_sse_lines(model=valid_settings.served_model_name)
    )

    def guard_factory(*args: Any, **kwargs: Any) -> FakeGuard:
        del args, kwargs
        return fake_guard

    def upstream_factory(*args: Any, **kwargs: Any) -> FakeUpstream:
        del args, kwargs
        return fake_upstream

    def process_factory(*args: Any, **kwargs: Any) -> FakeVLLMProcess:
        del args, kwargs
        return fake_process

    monkeypatch.setattr(main_module, "ONNXPromptGuard", guard_factory)
    monkeypatch.setattr(main_module, "VLLMUpstream", upstream_factory)
    monkeypatch.setattr(main_module, "VLLMProcess", process_factory)

    return cast(FastAPI, main_module.create_app(valid_settings))


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Start and stop the fake-backed FastAPI lifespan around each test."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {GATEWAY_TEST_KEY}"}


@pytest.fixture
def valid_request() -> dict[str, Any]:
    return {
        "model": "secure-llm-deploy-test",
        "messages": [
            {
                "role": "user",
                "content": "Explain what prompt injection is.",
            }
        ],
        "stream": False,
    }