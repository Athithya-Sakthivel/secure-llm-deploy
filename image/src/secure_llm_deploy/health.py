"""Gateway liveness and readiness responses."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


def liveness_response(request: Request) -> JSONResponse:
    """Liveness reports process health, independent of downstream dependencies."""
    del request
    return JSONResponse(status_code=200, content={"status": "live"})


def readiness_response(request: Request) -> JSONResponse:
    """Readiness requires both guard initialization and a live vLLM process."""
    state = request.app.state
    guard_ready = bool(
        getattr(state, "guard_ready", False)
        and getattr(state, "guard", None) is not None
    )
    process = getattr(state, "vllm_process", None)
    try:
        process_running = bool(process is not None and process.is_running())
    except Exception:
        process_running = False
    inference_ready = bool(
        getattr(state, "vllm_ready", False)
        and process_running
        and getattr(state, "upstream", None) is not None
    )
    if not guard_ready or not inference_ready:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(status_code=200, content={"status": "ready"})
