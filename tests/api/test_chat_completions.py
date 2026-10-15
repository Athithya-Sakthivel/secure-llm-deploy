"""API tests for non-streaming chat completions."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from secure_llm_deploy.guard import GuardUnavailableError
from secure_llm_deploy.upstream import UpstreamError


def test_valid_chat_completion_returns_sanitized_response(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == (
        "Hello from the fake model."
    )

    assert len(fake_upstream.calls) == 1
    assert response.headers["x-request-id"]


def test_wrong_model_name_is_rejected(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
    fake_guard: Any,
) -> None:
    valid_request["model"] = "unknown-model"

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert fake_guard.calls == []
    assert fake_upstream.calls == []


def test_blocked_prompt_returns_403_without_inference(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_guard: Any,
    fake_upstream: Any,
) -> None:
    fake_guard.blocked = True
    fake_guard.score = 0.99

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "prompt_blocked"
    assert len(fake_guard.calls) == 1
    assert fake_upstream.calls == []


def test_guard_failure_fails_closed_with_503(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_guard: Any,
    fake_upstream: Any,
) -> None:
    fake_guard.exception = GuardUnavailableError(
        "synthetic guard failure"
    )

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
    assert fake_upstream.calls == []


def test_input_pii_is_redacted_before_upstream(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_guard: Any,
    fake_upstream: Any,
    valid_request: dict[str, Any],
) -> None:
    original_content = "Send the report to alice@example.com."
    valid_request["messages"] = [
        {"role": "user", "content": original_content}
    ]

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert len(fake_guard.calls) == 1
    assert original_content in fake_guard.calls[0]

    forwarded_content = fake_upstream.calls[0]["messages"][0]["content"]
    assert "alice@example.com" not in forwarded_content
    assert "[REDACTED_EMAIL]" in forwarded_content


def test_output_pii_is_redacted_before_returning_to_client(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    fake_upstream.response_payload["choices"][0]["message"]["content"] = (
        "Contact bob@example.com."
    )

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.text

    assert "bob@example.com" not in body
    assert "[REDACTED_EMAIL]" in body


def test_configured_secret_is_redacted_from_output(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    secret = "SYNTHETIC_SECRET_ALPHA_001"
    fake_upstream.response_payload["choices"][0]["message"]["content"] = (
        f"Sensitive value: {secret}"
    )

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert secret not in response.text
    assert "[REDACTED_SECRET]" in response.text


def test_message_above_configured_character_limit_is_rejected(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_guard: Any,
    fake_upstream: Any,
) -> None:
    valid_request["messages"] = [
        {"role": "user", "content": "x" * 30001}
    ]

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 413
    assert fake_guard.calls == []
    assert fake_upstream.calls == []


def test_output_budget_above_runtime_maximum_is_rejected(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    valid_request["max_tokens"] = 1025

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert fake_upstream.calls == []


def test_tool_call_fields_are_rejected(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    valid_request["tools"] = []

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert fake_upstream.calls == []


def test_malformed_json_returns_422(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"{not-valid-json",
        headers={
            **auth_headers,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 422


def test_upstream_connection_failure_returns_503(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    fake_upstream.completion_error = UpstreamError(
        "synthetic unavailable upstream"
    )

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
    assert "synthetic unavailable upstream" not in response.text


def test_invalid_upstream_response_fails_safely(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    fake_upstream.response_payload["choices"] = []

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


def test_upstream_receives_request_id(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_request: dict[str, Any],
    fake_upstream: Any,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert fake_upstream.request_ids
    assert fake_upstream.request_ids[0] == response.headers["x-request-id"]