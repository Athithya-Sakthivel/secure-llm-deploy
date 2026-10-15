"""API tests for liveness and readiness endpoints."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.conftest import FakeVLLMProcess


def test_liveness_is_public(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_readiness_is_public(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_health_endpoints_do_not_require_bearer_token(
    client: TestClient,
) -> None:
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 200


def test_readiness_is_false_when_guard_is_not_ready(
    client: TestClient,
    app: FastAPI,
) -> None:
    app.state.guard_ready = False

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readiness_is_false_when_vllm_is_not_ready(
    client: TestClient,
    app: FastAPI,
) -> None:
    app.state.vllm_ready = False

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readiness_is_false_when_process_has_exited(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    fake_process.running = False
    app.state.vllm_ready = True
    app.state.guard_ready = True

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_liveness_is_false_when_process_has_exited(
    client: TestClient,
    fake_process: FakeVLLMProcess,
) -> None:
    fake_process.running = False

    response = client.get("/health/live")

    assert response.status_code == 503
    assert response.json() == {"status": "not_live"}


def test_health_response_does_not_expose_model_paths(
    client: TestClient,
    app: FastAPI,
) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.text

    assert str(app.state.settings.target_model_path) not in body
    assert str(app.state.settings.guard_model_path) not in body
    assert app.state.settings.gateway_key not in body
    assert app.state.settings.upstream_key not in body