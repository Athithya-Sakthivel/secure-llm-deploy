"""Unit tests for the local vLLM HTTP client."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from secure_llm_deploy.config import Settings
from secure_llm_deploy.upstream import UpstreamError, VLLMUpstream

MockHandler = Callable[[httpx.Request], httpx.Response]


@asynccontextmanager
async def mocked_upstream(
    settings: Settings,
    handler: MockHandler,
) -> AsyncIterator[VLLMUpstream]:
    upstream = VLLMUpstream(settings)
    await upstream.client.aclose()

    upstream.client = httpx.AsyncClient(
        base_url=upstream.base_url,
        headers={
            "Authorization": f"Bearer {settings.upstream_key}",
        },
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )

    try:
        yield upstream
    finally:
        await upstream.aclose()


def completion_payload(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-upstream-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello."},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.asyncio
async def test_chat_completion_sends_internal_bearer_key(
    valid_settings: Settings,
) -> None:
    observed: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["authorization"] = request.headers.get("authorization", "")
        observed["request_id"] = request.headers.get("x-request-id", "")
        return httpx.Response(
            200,
            json=completion_payload(valid_settings.served_model_name),
        )

    async with mocked_upstream(valid_settings, handler) as upstream:
        result = await upstream.chat_completion(
            {"model": valid_settings.served_model_name, "messages": []},
            request_id="request-test-001",
        )

    assert result["object"] == "chat.completion"
    assert observed["path"] == "/v1/chat/completions"
    assert observed["authorization"] == f"Bearer {valid_settings.upstream_key}"
    assert observed["request_id"] == "request-test-001"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [
        (400, 400),
        (413, 413),
        (422, 422),
        (500, 500),
        (503, 503),
    ],
)
async def test_upstream_status_is_classified(
    valid_settings: Settings,
    status_code: int,
    expected_status: int,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "synthetic"})

    async with mocked_upstream(valid_settings, handler) as upstream:
        with pytest.raises(UpstreamError) as exc_info:
            await upstream.chat_completion(
                {"messages": []},
                request_id="request-test-002",
            )

    assert exc_info.value.status_code == expected_status


@pytest.mark.asyncio
async def test_invalid_json_response_is_rejected(
    valid_settings: Settings,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    async with mocked_upstream(valid_settings, handler) as upstream:
        with pytest.raises(UpstreamError, match="vLLM request failed"):
            await upstream.chat_completion(
                {"messages": []},
                request_id="request-test-003",
            )


@pytest.mark.asyncio
async def test_non_object_json_response_is_rejected(
    valid_settings: Settings,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "array"])

    async with mocked_upstream(valid_settings, handler) as upstream:
        with pytest.raises(UpstreamError, match="invalid response"):
            await upstream.chat_completion(
                {"messages": []},
                request_id="request-test-004",
            )


@pytest.mark.asyncio
async def test_connection_failure_is_sanitized(
    valid_settings: Settings,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic internal network detail")

    async with mocked_upstream(valid_settings, handler) as upstream:
        with pytest.raises(
            UpstreamError,
            match="vLLM request failed",
        ) as exc_info:
            await upstream.chat_completion(
                {"messages": []},
                request_id="request-test-005",
            )

    assert "synthetic internal network detail" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_readiness_accepts_expected_served_model(
    valid_settings: Settings,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": valid_settings.served_model_name},
                ],
            },
        )

    async with mocked_upstream(valid_settings, handler) as upstream:
        await upstream.wait_until_ready(process_is_running=lambda: True)


@pytest.mark.asyncio
async def test_readiness_rejects_missing_served_model(
    valid_settings: Settings,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "different-model"}]},
        )

    async with mocked_upstream(valid_settings, handler) as upstream:
        with pytest.raises(UpstreamError, match="configured model"):
            await upstream.wait_until_ready(process_is_running=lambda: True)


@pytest.mark.asyncio
async def test_readiness_fails_if_process_is_not_running(
    valid_settings: Settings,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("No request should be sent for a dead process")

    async with mocked_upstream(valid_settings, handler) as upstream:
        with pytest.raises(UpstreamError, match="exited during startup"):
            await upstream.wait_until_ready(process_is_running=lambda: False)


@pytest.mark.asyncio
async def test_readiness_rejects_invalid_internal_key(
    valid_settings: Settings,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    async with mocked_upstream(valid_settings, handler) as upstream:
        with pytest.raises(UpstreamError, match="rejected"):
            await upstream.wait_until_ready(process_is_running=lambda: True)


@pytest.mark.asyncio
async def test_stream_requires_event_stream_content_type(
    valid_settings: Settings,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"unexpected": "response"},
        )

    async with mocked_upstream(valid_settings, handler) as upstream:
        with pytest.raises(
            UpstreamError,
            match="invalid streaming content type",
        ):
            await upstream.open_chat_completion_stream(
                {"stream": True},
                request_id="request-test-006",
            )


@pytest.mark.asyncio
async def test_stream_returns_a_closeable_response(
    valid_settings: Settings,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    async with mocked_upstream(valid_settings, handler) as upstream:
        response = await upstream.open_chat_completion_stream(
            {"stream": True},
            request_id="request-test-007",
        )

        lines = [line async for line in response.aiter_lines()]
        await response.aclose()

    assert lines == ["data: [DONE]", ""]