"""Private local vLLM HTTP client."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .config import Settings


class UpstreamError(RuntimeError):
    """vLLM is unavailable or returned an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class VLLMUpstream:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = (
            f"http://{settings.vllm_host}:{settings.vllm_port}"
        )

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
            headers={"Authorization": f"Bearer {settings.upstream_key}"},
            limits=httpx.Limits(
                max_connections=32,
                max_keepalive_connections=16,
            ),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def wait_until_ready(
        self,
        *,
        process_is_running: Any,
    ) -> None:
        """Poll the local authenticated API until the expected model is served."""
        deadline = time.monotonic() + self.settings.vllm_startup_timeout_seconds

        while time.monotonic() < deadline:
            if not process_is_running():
                raise UpstreamError("vLLM exited during startup")

            try:
                response = await asyncio.wait_for(
                    self.client.get("/v1/models"),
                    timeout=5.0,
                )

                if response.status_code in {401, 403}:
                    raise UpstreamError(
                        "vLLM rejected the configured internal API key"
                    )

                if response.status_code == 200:
                    payload = response.json()
                    data = payload.get("data") if isinstance(payload, dict) else None

                    if not isinstance(data, list):
                        raise UpstreamError("vLLM returned an invalid model list")

                    model_ids = {
                        item.get("id")
                        for item in data
                        if isinstance(item, dict)
                    }

                    if self.settings.served_model_name not in model_ids:
                        raise UpstreamError(
                            "vLLM is ready but the configured model is not served"
                        )

                    return

            except UpstreamError:
                raise
            except (httpx.HTTPError, ValueError, TypeError, AttributeError, TimeoutError):
                pass

            await asyncio.sleep(self.settings.vllm_startup_poll_seconds)

        raise UpstreamError("vLLM readiness timeout")

    @staticmethod
    def _raise_for_status(status_code: int) -> None:
        if status_code < 400:
            return

        if status_code in {400, 413, 422}:
            raise UpstreamError(
                "vLLM rejected the request",
                status_code=status_code,
            )

        raise UpstreamError("vLLM inference failed", status_code=status_code)

    async def chat_completion(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> dict[str, Any]:
        try:
            response = await self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-Request-ID": request_id},
            )
            self._raise_for_status(response.status_code)

            result = response.json()
            if not isinstance(result, dict):
                raise UpstreamError("vLLM returned an invalid response")

            return result

        except UpstreamError:
            raise
        except (httpx.HTTPError, ValueError, TypeError):
            raise UpstreamError("vLLM request failed") from None

    async def open_chat_completion_stream(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> httpx.Response:
        """Open a stream and verify status/content-type before returning it."""
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

            try:
                self._raise_for_status(response.status_code)

                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type.lower():
                    raise UpstreamError(
                        "vLLM returned an invalid streaming content type"
                    )

                return response

            except Exception:
                await response.aclose()
                raise

        except UpstreamError:
            raise
        except httpx.HTTPError:
            raise UpstreamError("vLLM streaming request failed") from None