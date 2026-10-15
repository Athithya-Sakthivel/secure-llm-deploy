"""API tests for authentication, request limits and rate limiting."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_models_requires_authentication(client: TestClient) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert response.headers["www-authenticate"] == "Bearer"


def test_chat_completions_requires_authentication(
    client: TestClient,
    valid_request: dict[str, object],
) -> None:
    response = client.post("/v1/chat/completions", json=valid_request)

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_invalid_token_is_rejected(
    client: TestClient,
    valid_request: dict[str, object],
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers={"Authorization": "Bearer invalid-test-token"},
    )

    assert response.status_code == 401


def test_malformed_bearer_header_is_rejected(
    client: TestClient,
    valid_request: dict[str, object],
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers={"Authorization": "Basic synthetic-test-value"},
    )

    assert response.status_code == 401


def test_unknown_route_is_authenticated_before_not_found(
    client: TestClient,
) -> None:
    response = client.get("/internal/debug")

    assert response.status_code == 401


def test_unknown_route_with_valid_key_returns_not_found(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get("/internal/debug", headers=auth_headers)

    assert response.status_code == 404


def test_duplicate_authorization_headers_are_rejected(
    client: TestClient,
    valid_request: dict[str, object],
    auth_headers: dict[str, str],
) -> None:
    headers = [
        ("Authorization", auth_headers["Authorization"]),
        ("Authorization", "Bearer another-synthetic-key"),
    ]

    response = client.post(
        "/v1/chat/completions",
        json=valid_request,
        headers=headers,
    )

    assert response.status_code == 401


def test_missing_auth_is_checked_before_invalid_json(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"{not-valid-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 401


def test_oversized_body_is_rejected_with_valid_auth(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"x" * (65536 + 1),
        headers={
            **auth_headers,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_unauthenticated_oversized_body_is_rejected_before_body_limit(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"x" * (65536 + 1),
        headers={"Content-Type": "application/json"},
    )

    # Authentication is deliberately checked before request-body consumption.
    assert response.status_code == 401


def test_spoofed_forwarded_ip_does_not_bypass_rate_limit(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    statuses: list[int] = []

    for index in range(6):
        response = client.get(
            "/v1/models",
            headers={
                **auth_headers,
                "X-Forwarded-For": f"198.51.100.{index + 1}",
                "X-Real-IP": f"203.0.113.{index + 1}",
            },
        )
        statuses.append(response.status_code)

    assert statuses[:5] == [200, 200, 200, 200, 200]
    assert statuses[5] == 429


def test_rate_limit_returns_retry_after(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    for _ in range(5):
        assert client.get("/v1/models", headers=auth_headers).status_code == 200

    response = client.get("/v1/models", headers=auth_headers)

    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1
    assert response.json()["error"]["code"] == "rate_limit_exceeded"


def test_response_includes_request_id(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get("/v1/models", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["x-request-id"]