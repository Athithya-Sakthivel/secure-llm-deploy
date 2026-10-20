"""Manage the local vLLM server subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
from contextlib import suppress
from typing import Protocol

from .config import Settings

__all__ = ["VLLMProcess", "VLLMProcessError", "asyncio", "os", "shutil", "signal"]


class VLLMProcessError(RuntimeError):
    """Raised when the local vLLM process cannot be managed."""


class _ChildProcess(Protocol):
    """Minimal process interface used by test doubles."""

    pid: int
    returncode: int | None

    async def wait(self) -> int:
        """Wait for subprocess exit and return its status."""
        ...


class VLLMProcess:
    """Start and stop one vLLM process under the gateway's lifecycle."""

    _SHUTDOWN_TIMEOUT_SECONDS = 20.0

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._process: asyncio.subprocess.Process | _ChildProcess | None = None

    def build_command(self) -> list[str]:
        """Build the pinned local model's OpenAI-compatible vLLM command."""
        executable = shutil.which(self.settings.vllm_executable)
        if executable is None:
            raise VLLMProcessError(
                f"vLLM executable was not found: {self.settings.vllm_executable}"
            )

        command = [
            executable,
            "serve",
            str(self.settings.target_model_path),
            "--host",
            self.settings.vllm_host,
            "--port",
            str(self.settings.vllm_port),
            "--served-model-name",
            self.settings.served_model_name,
            "--dtype",
            "half",
            "--max-model-len",
            str(self.settings.max_model_len),
            "--max-num-seqs",
            str(self.settings.vllm_max_num_seqs),
            "--gpu-memory-utilization",
            str(self.settings.gpu_memory_utilization),
            "--disable-uvicorn-access-log",
            "--no-enable-log-requests",
        ]

        if self.settings.serve_mode == "speculative":
            # The EAGLE checkpoint is a speculator head, not a second full model.
            # Its revision is pinned by model.lock and its custom config.py was
            # downloaded into the local artifact directory at image build time.
            speculative_config = {
                "method": "eagle3",
                "model": str(self.settings.eagle_model_path),
                "num_speculative_tokens": self.settings.speculative_tokens,
                "draft_tensor_parallel_size": 1,
            }
            command.extend(
                [
                    "--trust-remote-code",
                    "--speculative-config",
                    json.dumps(speculative_config, separators=(",", ":")),
                ]
            )

        return command

    def _command(self) -> list[str]:
        """Compatibility alias for tests and earlier callers."""
        return self.build_command()

    @staticmethod
    def _child_environment() -> dict[str, str]:
        """Copy the environment but remove unrelated inherited credentials."""
        environment = dict(os.environ)
        sensitive_markers = (
            "API_KEY",
            "ACCESS_KEY",
            "TOKEN",
            "PASSWORD",
            "SECRET",
            "PRIVATE_KEY",
            "CREDENTIAL",
        )
        for name in tuple(environment):
            if any(marker in name.upper() for marker in sensitive_markers):
                environment.pop(name, None)

        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        return environment

    async def start(self) -> asyncio.subprocess.Process:
        """Start vLLM with offline model loading and internal API authentication."""
        if self.is_running():
            raise VLLMProcessError("vLLM process is already running")
        # Drop a previous process reference if that process has exited.
        self._process = None

        command = self.build_command()
        child_environment = self._child_environment()
        try:
            # vLLM 0.31 reads this variable for API authentication. Keeping the
            # secret out of argv avoids exposing it in process listings.
            child_environment["VLLM_API_KEY"] = self.settings.upstream_key
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                env=child_environment,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise VLLMProcessError("Could not start the vLLM process") from exc

        self._process = process
        return process

    def is_running(self) -> bool:
        """Return whether the managed child has not exited."""
        process = self._process
        return process is not None and process.returncode is None

    async def stop(self) -> None:
        """Stop vLLM gracefully, then kill its process group after a timeout."""
        process = self._process
        if process is None:
            return
        if process.returncode is not None:
            self._process = None
            return

        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)

        try:
            await asyncio.wait_for(
                process.wait(), timeout=self._SHUTDOWN_TIMEOUT_SECONDS
            )
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

        self._process = None
