"""API tests for streaming chat completions."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from secure_llm_deploy.upstream import UpstreamError


def make_chunk(
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
    model: str = "secure-llm-deploy-test",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def make_sse_lines(
    *payloads: dict[str, Any],
) -> list[str]:
    lines: list[str] = []

    for payload in payloads:
        lines.extend(
            [
                f"data: {json.dumps(payload, separators=(',', ':'))}",
                "",
            ]
        )

    lines.extend(["data: [DONE]", ""])
    return lines


def streaming_request(
    valid_request: dict[str, Any],
) -> dict[str, Any]:
    return {**valid_request, "stream": True}


def test_streaming_returns_event_stream(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=streaming_request(valid_request),
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    assert "Hello from the fake model." in response.text
    assert len(fake_upstream.calls) == 1


def test_streaming_requires_authentication(
    client: TestClient,
    valid_request: dict[str, Any],
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=streaming_request(valid_request),
    )

    assert response.status_code == 401


def test_streaming_open_failure_returns_503_before_headers(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    fake_upstream.stream_open_error = UpstreamError(
        "synthetic stream connection failure"
    )

    response = client.post(
        "/v1/chat/completions",
        json=streaming_request(valid_request),
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
    assert "synthetic stream connection failure" not in response.text


def test_streaming_output_pii_is_redacted(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    model = "secure-llm-deploy-test"
    fake_upstream.stream_response.lines = make_sse_lines(
        make_chunk({"role": "assistant"}, model=model),
        make_chunk(
            {"content": "Contact bob@example.com for support."},
            model=model,
        ),
        make_chunk({}, finish_reason="stop", model=model),
    )

    response = client.post(
        "/v1/chat/completions",
        json=streaming_request(valid_request),
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert "bob@example.com" not in response.text
    assert "[REDACTED_EMAIL]" in response.text
    assert "data: [DONE]" in response.text


def test_streaming_configured_secret_split_across_chunks_is_redacted(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    secret = "SYNTHETIC_SECRET_ALPHA_001"
    model = "secure-llm-deploy-test"

    prefix = "ordinary safe content " * 30
    suffix = " more safe content" * 40
    combined_text = prefix + secret + suffix

    secret_start = len(prefix)
    first_cut = secret_start + 8
    second_cut = secret_start + 17

    fake_upstream.stream_response.lines = make_sse_lines(
        make_chunk({"role": "assistant"}, model=model),
        make_chunk({"content": combined_text[:first_cut]}, model=model),
        make_chunk(
            {"content": combined_text[first_cut:second_cut]},
            model=model,
        ),
        make_chunk({"content": combined_text[second_cut:]}, model=model),
        make_chunk({}, finish_reason="stop", model=model),
    )

    response = client.post(
        "/v1/chat/completions",
        json=streaming_request(valid_request),
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert secret not in response.text
    assert "[REDACTED_SECRET]" in response.text
    assert "data: [DONE]" in response.text


def test_streaming_malformed_event_emits_safe_sse_error(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    fake_upstream.stream_response.lines = [
        "data: {invalid-json",
        "",
    ]

    response = client.post(
        "/v1/chat/completions",
        json=streaming_request(valid_request),
        headers=auth_headers,
    )

    # The HTTP stream has already started, so its status cannot change to 503.
    assert response.status_code == 200
    assert "upstream_stream_error" in response.text
    assert "invalid-json" not in response.text


def test_stream_is_closed_after_success(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=streaming_request(valid_request),
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert fake_upstream.stream_response.closed


def test_stream_is_closed_after_filter_failure(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    fake_upstream.stream_response.lines = [
        'data: {"error":{"message":"synthetic upstream detail"}}',
        "",
    ]

    response = client.post(
        "/v1/chat/completions",
        json=streaming_request(valid_request),
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert fake_upstream.stream_response.closed
    assert "synthetic upstream detail" not in response.text
    assert "upstream_stream_error" in response.text