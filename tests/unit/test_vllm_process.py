"""Unit tests for vLLM process command and lifecycle management."""

from __future__ import annotations

import json
import signal
from typing import Any

import pytest

import secure_llm_deploy.vllm_process as process_module
from secure_llm_deploy.config import Settings
from secure_llm_deploy.vllm_process import VLLMProcess


def test_baseline_command_uses_target_model_and_loopback(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    monkeypatch.setattr(
        process_module.shutil,
        "which",
        lambda _name: "/usr/local/bin/vllm",
    )

    process = VLLMProcess(valid_settings)
    command = process.build_command()

    assert command[0] == "/usr/local/bin/vllm"
    assert command[1] == "serve"
    assert command[2] == str(valid_settings.target_model_path)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == str(valid_settings.vllm_port)
    assert command[command.index("--served-model-name") + 1] == (
        valid_settings.served_model_name
    )
    assert "--speculative-config" not in command


def test_baseline_command_contains_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    monkeypatch.setattr(
        process_module.shutil,
        "which",
        lambda _name: "/usr/local/bin/vllm",
    )

    command = VLLMProcess(valid_settings).build_command()

    assert command[command.index("--max-model-len") + 1] == str(
        valid_settings.max_model_len
    )
    assert command[command.index("--max-num-seqs") + 1] == str(
        valid_settings.vllm_max_num_seqs
    )
    assert command[command.index("--gpu-memory-utilization") + 1] == str(
        valid_settings.gpu_memory_utilization
    )
    assert "--disable-uvicorn-access-log" in command
    assert "--no-enable-log-requests" in command


def test_speculative_command_contains_expected_json(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    settings = valid_settings.model_copy(update={"serve_mode": "speculative"})

    monkeypatch.setattr(
        process_module.shutil,
        "which",
        lambda _name: "/usr/local/bin/vllm",
    )

    command = VLLMProcess(settings).build_command()
    config_index = command.index("--speculative-config")
    speculative_config = json.loads(command[config_index + 1])

    assert speculative_config == {
        "method": "draft_model",
        "model": str(settings.draft_model_path),
        "num_speculative_tokens": settings.speculative_tokens,
    }


def test_command_never_contains_api_keys(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    monkeypatch.setattr(
        process_module.shutil,
        "which",
        lambda _name: "/usr/local/bin/vllm",
    )

    command = VLLMProcess(valid_settings).build_command()

    assert valid_settings.gateway_key not in command
    assert valid_settings.upstream_key not in command


def test_child_environment_removes_inherited_secret_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_API_KEY", "synthetic-gateway-secret")
    monkeypatch.setenv("SOME_PASSWORD", "synthetic-password")
    monkeypatch.setenv("HF_TOKEN", "synthetic-hf-token")
    monkeypatch.setenv("PRIVATE_KEY_DATA", "synthetic-private-key")

    environment = VLLMProcess._child_environment()

    assert "GATEWAY_API_KEY" not in environment
    assert "SOME_PASSWORD" not in environment
    assert "HF_TOKEN" not in environment
    assert "PRIVATE_KEY_DATA" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"


class FakeChildProcess:
    def __init__(self) -> None:
        self.pid = 12345
        self.returncode: int | None = None
        self.wait_calls = 0

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.mark.asyncio
async def test_start_uses_subprocess_exec_and_injects_internal_key(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    child = FakeChildProcess()
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        process_module.shutil,
        "which",
        lambda _name: "/usr/local/bin/vllm",
    )

    async def fake_create_subprocess_exec(
        *args: str,
        **kwargs: Any,
    ) -> FakeChildProcess:
        observed["args"] = list(args)
        observed["kwargs"] = kwargs
        return child

    monkeypatch.setattr(
        process_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    process = VLLMProcess(valid_settings)
    await process.start()

    assert process.is_running()
    assert observed["args"][1] == "serve"
    assert observed["kwargs"]["stdin"] is process_module.asyncio.subprocess.DEVNULL
    assert observed["kwargs"]["start_new_session"] is True

    child_environment = observed["kwargs"]["env"]
    assert child_environment["VLLM_API_KEY"] == valid_settings.upstream_key
    assert "GATEWAY_API_KEY" not in child_environment

    assert valid_settings.gateway_key not in observed["args"]
    assert valid_settings.upstream_key not in observed["args"]


@pytest.mark.asyncio
async def test_start_fails_safely_if_executable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    monkeypatch.setattr(process_module.shutil, "which", lambda _name: None)

    process = VLLMProcess(valid_settings)

    with pytest.raises(RuntimeError, match="vLLM executable was not found"):
        await process.start()


@pytest.mark.asyncio
async def test_start_wraps_subprocess_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    monkeypatch.setattr(
        process_module.shutil,
        "which",
        lambda _name: "/usr/local/bin/vllm",
    )

    async def fail_start(
        *_args: Any,
        **_kwargs: Any,
    ) -> FakeChildProcess:
        raise OSError("synthetic operating-system detail")

    monkeypatch.setattr(
        process_module.asyncio,
        "create_subprocess_exec",
        fail_start,
    )

    process = VLLMProcess(valid_settings)

    with pytest.raises(RuntimeError, match="Could not start the vLLM process"):
        await process.start()


@pytest.mark.asyncio
async def test_stop_sends_sigterm_and_waits_for_process(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    child = FakeChildProcess()
    child.returncode = None
    signals: list[int] = []

    process = VLLMProcess(valid_settings)
    process._process = child  # Arrange the process as if it had started.

    def fake_killpg(pid: int, sig: int) -> None:
        assert pid == child.pid
        signals.append(sig)
        child.returncode = -sig

    monkeypatch.setattr(process_module.os, "killpg", fake_killpg)

    await process.stop()

    assert signals == [signal.SIGTERM]
    assert child.wait_calls == 1
    assert not process.is_running()


@pytest.mark.asyncio
async def test_stop_is_idempotent_for_an_unstarted_process(
    valid_settings: Settings,
) -> None:
    process = VLLMProcess(valid_settings)

    await process.stop()

    assert not process.is_running()


@pytest.mark.asyncio
async def test_start_does_not_launch_a_second_process(
    monkeypatch: pytest.MonkeyPatch,
    valid_settings: Settings,
) -> None:
    process = VLLMProcess(valid_settings)
    process._process = FakeChildProcess()

    monkeypatch.setattr(
        process_module.shutil,
        "which",
        lambda _name: "/usr/local/bin/vllm",
    )

    with pytest.raises(RuntimeError, match="already running"):
        await process.start()