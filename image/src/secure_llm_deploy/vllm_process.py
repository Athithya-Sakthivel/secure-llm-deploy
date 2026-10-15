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

__all__ = [
    "VLLMProcess",
    "VLLMProcessError",
    "asyncio",
    "os",
    "shutil",
    "signal",
]


class VLLMProcessError(RuntimeError):
    """Raised when the local vLLM process cannot be managed."""


class _ChildProcess(Protocol):
    """Small process interface, also convenient for unit-test fakes."""

    pid: int
    returncode: int | None

    async def wait(self) -> int:
        """Wait for the subprocess to finish and return its exit status."""
        ...


class VLLMProcess:
    """Start and stop vLLM using the application's validated settings."""

    _SHUTDOWN_TIMEOUT_SECONDS = 20.0

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._process: asyncio.subprocess.Process | _ChildProcess | None = None


    def build_command(self) -> list[str]:
        """Build the command used to launch the OpenAI-compatible vLLM API."""
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
            speculative_config = {
                "method": "draft_model",
                "model": str(self.settings.draft_model_path),
                "num_speculative_tokens": self.settings.speculative_tokens,
            }
            command.extend(
                [
                    "--speculative-config",
                    json.dumps(speculative_config, separators=(",", ":")),
                ]
            )

        return command


    def _command(self) -> list[str]:
        """Backward-compatible alias for :meth:`build_command`."""
        return self.build_command()


    @staticmethod
    def _child_environment() -> dict[str, str]:
        """Copy the environment while removing inherited secrets."""
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

        for name in list(environment):
            normalized_name = name.upper()
            if any(marker in normalized_name for marker in sensitive_markers):
                environment.pop(name, None)

        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"

        return environment



    async def start(self) -> asyncio.subprocess.Process:
        """Start vLLM without blocking on API readiness."""
        if self.is_running():
            raise VLLMProcessError("vLLM process is already running")

        command = self.build_command()
        child_environment = self._child_environment()
        child_environment["VLLM_API_KEY"] = self.settings.upstream_key

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                env=child_environment,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise VLLMProcessError(
                f"Could not start the vLLM process: {exc}"
            ) from exc

        self._process = process
        return process


    def is_running(self) -> bool:
        """Return whether the managed vLLM process has not exited."""
        process = self._process
        return process is not None and process.returncode is None

    async def stop(self) -> None:
        """Ask vLLM to exit, then force-kill it if graceful shutdown times out."""
        process = self._process
        if process is None:
            return

        if process.returncode is not None:
            self._process = None
            return

        try:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)

            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=self._SHUTDOWN_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        finally:
            self._process = None
