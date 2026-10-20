"""Authenticated HTTP client for the private, loopback-only vLLM API."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import httpx

from .config import Settings


class UpstreamError(RuntimeError):
    """vLLM is unavailable or returned an invalid response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class VLLMUpstream:
    """Small async client that never trusts environment proxy settings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = f"http://{settings.vllm_host}:{settings.vllm_port}"
        timeout = httpx.Timeout(
            connect=settings.vllm_connect_timeout_seconds,
            read=settings.vllm_read_timeout_seconds,
            write=settings.vllm_write_timeout_seconds,
            pool=settings.vllm_pool_timeout_seconds,
        )
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
            headers={"Authorization": f"Bearer {settings.upstream_key}"},
            limits=httpx.Limits(
                max_connections=32,
                max_keepalive_connections=16,
            ),
        )

    async def aclose(self) -> None:
        """Close pooled HTTP connections."""
        await self.client.aclose()

    async def wait_until_ready(
        self,
        *,
        process_is_running: Callable[[], bool],
    ) -> None:
        """Wait until authenticated /v1/models confirms the configured model."""
        deadline = time.monotonic() + self.settings.vllm_startup_timeout_seconds
        while time.monotonic() < deadline:
            try:
                if not process_is_running():
                    raise UpstreamError("vLLM exited during startup")
            except UpstreamError:
                raise
            except Exception:
                raise UpstreamError("Could not inspect vLLM process state") from None

            try:
                response = await asyncio.wait_for(
                    self.client.get("/v1/models"), timeout=5.0
                )
            except (httpx.HTTPError, TimeoutError):
                await asyncio.sleep(self.settings.vllm_startup_poll_seconds)
                continue

            if response.status_code in {401, 403}:
                raise UpstreamError(
                    "vLLM rejected the configured internal API key",
                    status_code=response.status_code,
                )
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise UpstreamError("vLLM returned invalid model-list JSON") from exc
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list):
                    raise UpstreamError("vLLM returned an invalid model list")
                model_ids = {
                    item.get("id") for item in data if isinstance(item, dict)
                }
                if self.settings.served_model_name not in model_ids:
                    raise UpstreamError(
                        "vLLM is ready but the configured model is not served"
                    )
                return
            if 400 <= response.status_code < 500:
                raise UpstreamError(
                    "vLLM rejected its readiness request",
                    status_code=response.status_code,
                )
            # 3xx and 5xx are not readiness; retry until the deadline.
            await asyncio.sleep(self.settings.vllm_startup_poll_seconds)

        raise UpstreamError("vLLM readiness timeout")

    @staticmethod
    def _raise_for_status(status_code: int) -> None:
        if 200 <= status_code < 300:
            return
        if status_code in {400, 413, 422}:
            raise UpstreamError(
                "vLLM rejected the request", status_code=status_code
            )
        if status_code in {401, 403}:
            raise UpstreamError(
                "vLLM authentication failed", status_code=status_code
            )
        raise UpstreamError("vLLM inference failed", status_code=status_code)

    async def chat_completion(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> dict[str, Any]:
        """Submit one non-streaming chat completion and validate JSON shape."""
        try:
            response = await self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-Request-ID": request_id},
            )
            self._raise_for_status(response.status_code)
            result = response.json()
        except UpstreamError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise UpstreamError("vLLM request failed") from exc

        if not isinstance(result, dict):
            raise UpstreamError("vLLM returned an invalid response")
        return result

    async def open_chat_completion_stream(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> httpx.Response:
        """Open a streaming response and validate status and media type."""
        try:
            request = self.client.build_request(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers={
                    "Accept": "text/event-stream",
                    "X-Request-ID": request_id,
                },
            )
            response = await self.client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise UpstreamError("vLLM streaming request failed") from exc

        try:
            self._raise_for_status(response.status_code)
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type.casefold():
                raise UpstreamError(
                    "vLLM returned an invalid streaming content type",
                    status_code=response.status_code,
                )
            return response
        except Exception:
            await response.aclose()
            raise
