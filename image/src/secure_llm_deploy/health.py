"""Kubernetes liveness and readiness responses."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


def liveness_response(request: Request) -> JSONResponse:
    process = getattr(request.app.state, "vllm_process", None)

    if process is not None and not process.is_running():
        return JSONResponse(
            status_code=503,
            content={"status": "not_live"},
        )

    return JSONResponse(
        status_code=200,
        content={"status": "live"},
    )


def readiness_response(request: Request) -> JSONResponse:
    guard_ready = bool(getattr(request.app.state, "guard_ready", False))
    vllm_ready = bool(getattr(request.app.state, "vllm_ready", False))
    process = getattr(request.app.state, "vllm_process", None)

    if process is not None and not process.is_running():
        vllm_ready = False

    if not guard_ready or not vllm_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready"},
    )