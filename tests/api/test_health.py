
"""API tests for public liveness and dependency-aware readiness endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.conftest import FakeVLLMProcess


def _mark_ready(
    app: FastAPI,
    process: FakeVLLMProcess,
) -> None:
    """Set the application state to represent healthy dependencies."""
    app.state.guard = object()
    app.state.upstream = object()
    app.state.vllm_process = process
    app.state.guard_ready = True
    app.state.vllm_ready = True


def test_liveness_is_public(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_readiness_is_public(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_health_endpoints_do_not_require_bearer_token(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)

    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "live"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


def test_readiness_is_false_when_guard_is_not_ready(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)
    app.state.guard_ready = False

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readiness_is_false_when_guard_object_is_missing(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)
    app.state.guard = None

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readiness_is_false_when_vllm_is_not_ready(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)
    app.state.vllm_ready = False

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readiness_is_false_when_upstream_is_missing(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)
    app.state.upstream = None

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readiness_is_false_when_process_has_exited(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)
    fake_process.running = False

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_liveness_stays_live_when_vllm_process_has_exited(
    client: TestClient,
    fake_process: FakeVLLMProcess,
) -> None:
    fake_process.running = False

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_readiness_fails_closed_when_process_state_check_raises(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)

    class BrokenProcess:
        def is_running(self) -> bool:
            raise RuntimeError("synthetic process-state failure")

    app.state.vllm_process = BrokenProcess()

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_health_response_does_not_expose_model_paths_or_keys(
    client: TestClient,
    app: FastAPI,
    fake_process: FakeVLLMProcess,
) -> None:
    _mark_ready(app, fake_process)

    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.text

    assert str(app.state.settings.target_model_path) not in body
    assert str(app.state.settings.guard_model_path) not in body
    assert app.state.settings.gateway_key not in body
    assert app.state.settings.upstream_key not in body
