
"""Secure LLM gateway application.

Security properties:
- Bearer authentication is required except for explicit health endpoints.
- Request bodies are bounded before FastAPI parses JSON.
- Per-client rate limits use the direct ASGI peer address.
- Prompt guard failures fail closed.
- PII redaction applies to input and output, including SSE chunks.
- Audit logs contain metadata only, never prompts or completions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .audit import AuditLogger
from .auth import authenticate_bearer_token
from .config import Settings
from .guard import (
    GuardInputTooLongError,
    GuardResult,
    GuardUnavailableError,
    ONNXPromptGuard,
)
from .health import liveness_response, readiness_response
from .output_safety import OutputSafety, OutputSafetyError, OutputStreamError
from .pii import PiiRedactor
from .rate_limit import SlidingWindowRateLimiter
from .schemas import ChatCompletionRequest, ChatMessage
from .upstream import UpstreamError, VLLMUpstream
from .vllm_process import VLLMProcess

logger = logging.getLogger("secure_llm_deploy.main")

PUBLIC_HEALTH_ROUTES = {
    ("GET", "/health/live"),
    ("GET", "/health/ready"),
}

ROUTE_LABELS = {
    ("GET", "/health/live"): "health_live",
    ("GET", "/health/ready"): "health_ready",
    ("GET", "/v1/models"): "models",
    ("POST", "/v1/chat/completions"): "chat_completions",
}


def configure_logging() -> None:
    """Configure useful INFO-level console logs without enabling debug output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=False,
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def endpoint_label(method: str, path: str) -> str:
    """Return a stable route name instead of logging arbitrary URL paths."""
    return ROUTE_LABELS.get((method, path), "other")


def error_payload(
    message: str,
    *,
    error_type: str,
    code: str,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


async def send_json(
    send: Send,
    *,
    status_code: int,
    payload: dict[str, Any],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    response_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if headers:
        response_headers.extend(headers)

    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": response_headers,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        }
    )


class GatewaySecurityMiddleware:
    """Default-deny authentication and bounded request handling."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: Settings,
        limiter: SlidingWindowRateLimiter,
        audit: AuditLogger,
    ) -> None:
        self.app = app
        self.settings = settings
        self.limiter = limiter
        self.audit = audit

    @staticmethod
    def _header_values(scope: Scope, name: bytes) -> list[bytes]:
        return [
            value
            for key, value in scope.get("headers", [])
            if key.lower() == name
        ]

    @staticmethod
    def _client_identity(scope: Scope) -> str:
        client = scope.get("client")
        if client is None or not client[0]:
            return "unknown"
        return str(client[0])

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "/"))
        endpoint = endpoint_label(method, path)
        request_id = uuid.uuid4().hex
        started = time.perf_counter()

        scope.setdefault("state", {})["request_id"] = request_id

        status_code: int | None = None
        response_started = False
        response_completed = False
        logged = False

        def log_completion() -> None:
            nonlocal logged

            if logged:
                return

            logged = True
            final_status = status_code if status_code is not None else 500
            duration_ms = (time.perf_counter() - started) * 1000

            self.audit.log_http_request(
                request_id=request_id,
                method=method,
                endpoint=endpoint,
                status_code=final_status,
                duration_ms=duration_ms,
            )

            logger.info(
                "http_request_complete request_id=%s method=%s endpoint=%s "
                "status_code=%d duration_ms=%.2f",
                request_id,
                method,
                endpoint,
                final_status,
                duration_ms,
            )

        async def wrapped_send(message: Message) -> None:
            nonlocal status_code, response_started, response_completed

            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_started = True

                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))

                await send({**message, "headers": headers})
                return

            await send(message)

            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_completed = True
                log_completion()

        async def reject(
            *,
            code: int,
            message: str,
            error_type: str,
            error_code: str,
            headers: list[tuple[bytes, bytes]] | None = None,
        ) -> None:
            logger.info(
                "request_rejected request_id=%s endpoint=%s status_code=%d "
                "reason=%s",
                request_id,
                endpoint,
                code,
                error_code,
            )
            await send_json(
                wrapped_send,
                status_code=code,
                payload=error_payload(
                    message,
                    error_type=error_type,
                    code=error_code,
                ),
                headers=headers,
            )

        try:
            is_public_health = (method, path) in PUBLIC_HEALTH_ROUTES

            if not is_public_health:
                auth_headers = self._header_values(scope, b"authorization")
                authorization = (
                    auth_headers[0].decode("latin-1")
                    if len(auth_headers) == 1
                    else None
                )

                if not authenticate_bearer_token(
                    authorization,
                    self.settings.gateway_api_key,
                ):
                    await reject(
                        code=401,
                        message="Missing or invalid bearer token",
                        error_type="authentication_error",
                        error_code="invalid_api_key",
                        headers=[(b"www-authenticate", b"Bearer")],
                    )
                    return

                allowed, retry_after = await self.limiter.allow(
                    self._client_identity(scope)
                )
                if not allowed:
                    await reject(
                        code=429,
                        message="Rate limit exceeded",
                        error_type="rate_limit_error",
                        error_code="rate_limit_exceeded",
                        headers=[
                            (b"retry-after", str(retry_after).encode("ascii"))
                        ],
                    )
                    return

            # Reject ambiguous Content-Length headers.
            content_lengths = self._header_values(scope, b"content-length")
            if len(content_lengths) > 1:
                await reject(
                    code=400,
                    message="Invalid Content-Length",
                    error_type="invalid_request_error",
                    error_code="invalid_content_length",
                )
                return

            if content_lengths:
                try:
                    content_length = int(content_lengths[0])
                except ValueError:
                    await reject(
                        code=400,
                        message="Invalid Content-Length",
                        error_type="invalid_request_error",
                        error_code="invalid_content_length",
                    )
                    return

                if (
                    content_length < 0
                    or content_length > self.settings.request_max_body_bytes
                ):
                    await reject(
                        code=413,
                        message="Request body too large",
                        error_type="invalid_request_error",
                        error_code="request_too_large",
                    )
                    return

            # Buffer request chunks to enforce a hard limit even when the
            # client omits Content-Length or uses chunked transfer encoding.
            body_messages: list[Message] = []
            body_size = 0

            while True:
                incoming = await receive()

                if incoming["type"] == "http.disconnect":
                    logger.info(
                        "client_disconnected request_id=%s endpoint=%s phase=request_body",
                        request_id,
                        endpoint,
                    )
                    return

                if incoming["type"] != "http.request":
                    continue

                body_size += len(incoming.get("body", b""))
                if body_size > self.settings.request_max_body_bytes:
                    await reject(
                        code=413,
                        message="Request body too large",
                        error_type="invalid_request_error",
                        error_code="request_too_large",
                    )
                    return

                body_messages.append(incoming)
                if not incoming.get("more_body", False):
                    break

            message_index = 0

            async def replay_receive() -> Message:
                nonlocal message_index

                if message_index < len(body_messages):
                    message = body_messages[message_index]
                    message_index += 1
                    return message

                return await receive()

            await self.app(scope, replay_receive, wrapped_send)

            # Some ASGI apps may return without emitting a final body event.
            if not response_completed:
                log_completion()

        except Exception:
            logger.exception(
                "middleware_failure request_id=%s endpoint=%s",
                request_id,
                endpoint,
            )

            if response_started:
                # Headers may already be sent; do not attempt a second response.
                raise

            await reject(
                code=500,
                message="Internal server error",
                error_type="server_error",
                error_code="internal_error",
            )

        finally:
            if not logged:
                log_completion()


def input_token_upper_bound(messages: Sequence[ChatMessage]) -> int:
    """Compute a conservative character/byte budget before inference.

    This is not exact tokenizer accounting. The vLLM model's own context
    validation remains authoritative.
    """
    total = 32

    for message in messages:
        total += len(message.content.encode("utf-8"))
        total += len(message.role.encode("utf-8"))
        total += 32

    return total


async def acquire_slot(
    semaphore: asyncio.Semaphore,
    *,
    wait_seconds: float,
) -> None:
    try:
        async with asyncio.timeout(wait_seconds):
            await semaphore.acquire()
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Service is at concurrency capacity",
        ) from None


async def run_guard(
    guard: ONNXPromptGuard,
    semaphore: asyncio.Semaphore,
    text: str,
    *,
    wait_timeout: float,
) -> GuardResult:
    await acquire_slot(semaphore, wait_seconds=wait_timeout)
    task = asyncio.create_task(asyncio.to_thread(guard.classify, text))

    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        # Thread work cannot be force-cancelled. Keep the semaphore slot until
        # it finishes, preventing an unbounded number of abandoned guard jobs.
        try:
            await asyncio.shield(task)
        finally:
            semaphore.release()
        raise
    except Exception:
        semaphore.release()
        raise
    else:
        semaphore.release()
        return result


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    runtime_settings = settings or Settings()
    audit = AuditLogger()

    limiter = SlidingWindowRateLimiter(
        max_requests=runtime_settings.rate_limit_requests,
        window_seconds=runtime_settings.rate_limit_window_seconds,
        max_clients=runtime_settings.rate_limit_max_clients,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        guard: ONNXPromptGuard | None = None
        upstream: VLLMUpstream | None = None
        process: VLLMProcess | None = None

        app.state.guard_ready = False
        app.state.vllm_ready = False

        logger.info(
            "application_starting app=%s mode=%s model=%s",
            runtime_settings.app_name,
            runtime_settings.serve_mode,
            runtime_settings.served_model_name,
        )

        try:
            runtime_settings.validate_runtime()

            logger.info("guard_loading provider=CPUExecutionProvider")
            guard = await asyncio.to_thread(
                ONNXPromptGuard,
                runtime_settings.guard_model_path,
                runtime_settings.guard_tokenizer_path,
                runtime_settings.guard_label_mapping_path,
                threshold=runtime_settings.guard_threshold,
                max_input_tokens=runtime_settings.guard_max_input_tokens,
                intra_op_threads=runtime_settings.guard_intra_op_threads,
            )
            logger.info("guard_ready provider=CPUExecutionProvider")

            redactor = PiiRedactor(runtime_settings.sensitive_values)
            output_safety = OutputSafety(redactor)
            upstream = VLLMUpstream(runtime_settings)
            process = VLLMProcess(runtime_settings)

            app.state.guard = guard
            app.state.upstream = upstream
            app.state.vllm_process = process
            app.state.redactor = redactor
            app.state.output_safety = output_safety
            app.state.guard_semaphore = asyncio.Semaphore(
                runtime_settings.max_concurrent_guard_evaluations
            )
            app.state.inference_semaphore = asyncio.Semaphore(
                runtime_settings.max_concurrent_inference_requests
            )

            app.state.guard_ready = True

            logger.info(
                "vllm_starting host=%s port=%d mode=%s",
                runtime_settings.vllm_host,
                runtime_settings.vllm_port,
                runtime_settings.serve_mode,
            )
            await process.start()

            await upstream.wait_until_ready(
                process_is_running=process.is_running,
            )
            app.state.vllm_ready = True

            logger.info(
                "application_ready mode=%s model=%s",
                runtime_settings.serve_mode,
                runtime_settings.served_model_name,
            )
            yield

        except Exception:
            audit.log_startup_failure()
            logger.exception("application_startup_failed")
            raise

        finally:
            app.state.guard_ready = False
            app.state.vllm_ready = False

            logger.info("application_shutting_down")

            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                if process is not None:
                    await process.stop()

            logger.info("application_stopped")

    app = FastAPI(
        title=runtime_settings.app_name,
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    app.state.settings = runtime_settings
    app.state.audit = audit
    app.state.guard_ready = False
    app.state.vllm_ready = False
    app.state.guard = None
    app.state.upstream = None
    app.state.vllm_process = None
    app.state.redactor = None
    app.state.output_safety = None
    app.state.guard_semaphore = None
    app.state.inference_semaphore = None

    app.add_middleware(
        GatewaySecurityMiddleware,
        settings=runtime_settings,
        limiter=limiter,
        audit=audit,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        logger.info(
            "request_validation_failed request_id=%s endpoint=%s",
            getattr(request.state, "request_id", "unknown"),
            endpoint_label(request.method, request.url.path),
        )
        return JSONResponse(
            status_code=422,
            content=error_payload(
                "Request validation failed",
                error_type="invalid_request_error",
                code="invalid_request",
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        safe_errors = {
            400: ("Invalid request", "invalid_request_error", "bad_request"),
            401: (
                "Missing or invalid bearer token",
                "authentication_error",
                "invalid_api_key",
            ),
            403: (
                "Prompt blocked by the security guard",
                "permission_error",
                "prompt_blocked",
            ),
            413: (
                "Request exceeds configured limits",
                "invalid_request_error",
                "request_too_large",
            ),
            422: (
                "Request validation failed",
                "invalid_request_error",
                "invalid_request",
            ),
            429: (
                "Rate limit exceeded",
                "rate_limit_error",
                "rate_limit_exceeded",
            ),
            503: (
                "Inference service is unavailable",
                "server_error",
                "service_unavailable",
            ),
        }

        message, error_type, code = safe_errors.get(
            exc.status_code,
            ("Request failed", "server_error", "request_failed"),
        )

        logger.info(
            "http_exception request_id=%s endpoint=%s status_code=%d code=%s",
            getattr(request.state, "request_id", "unknown"),
            endpoint_label(request.method, request.url.path),
            exc.status_code,
            code,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(
                message,
                error_type=error_type,
                code=code,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        _exc: Exception,
    ) -> JSONResponse:
        logger.error(
            "unhandled_application_error request_id=%s endpoint=%s",
            getattr(request.state, "request_id", "unknown"),
            endpoint_label(request.method, request.url.path),
        )
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "Internal server error",
                error_type="server_error",
                code="internal_error",
            ),
        )

    @app.get("/health/live")
    async def live(request: Request) -> JSONResponse:
        return liveness_response(request)

    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        return readiness_response(request)

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        process: VLLMProcess | None = request.app.state.vllm_process

        if (
            not request.app.state.vllm_ready
            or process is None
            or not process.is_running()
        ):
            raise HTTPException(status_code=503, detail="Service unavailable")

        return {
            "object": "list",
            "data": [
                {
                    "id": runtime_settings.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "secure-llm-deploy",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
    ) -> Response:
        request_id: str = request.state.request_id
        guard: ONNXPromptGuard | None = request.app.state.guard
        upstream: VLLMUpstream | None = request.app.state.upstream
        process: VLLMProcess | None = request.app.state.vllm_process
        redactor: PiiRedactor | None = request.app.state.redactor
        output_safety: OutputSafety | None = request.app.state.output_safety
        guard_semaphore: asyncio.Semaphore | None = (
            request.app.state.guard_semaphore
        )
        inference_semaphore: asyncio.Semaphore | None = (
            request.app.state.inference_semaphore
        )

        if (
            not request.app.state.guard_ready
            or not request.app.state.vllm_ready
            or guard is None
            or upstream is None
            or process is None
            or redactor is None
            or output_safety is None
            or guard_semaphore is None
            or inference_semaphore is None
            or not process.is_running()
        ):
            raise HTTPException(status_code=503, detail="Service unavailable")

        if body.model != runtime_settings.served_model_name:
            raise HTTPException(status_code=400, detail="Unknown model")

        if len(body.messages) > runtime_settings.max_messages:
            raise HTTPException(status_code=413, detail="Too many messages")

        if any(
            len(message.content) > runtime_settings.max_message_chars
            for message in body.messages
        ):
            raise HTTPException(
                status_code=413,
                detail="Message exceeds the configured character limit",
            )

        if (
            sum(len(message.content) for message in body.messages)
            > runtime_settings.max_input_chars
        ):
            raise HTTPException(
                status_code=413,
                detail="Input exceeds the configured character limit",
            )

        max_tokens = (
            body.max_tokens
            if body.max_tokens is not None
            else runtime_settings.default_max_tokens
        )
        if max_tokens > runtime_settings.max_output_tokens:
            raise HTTPException(
                status_code=422,
                detail="Requested output exceeds the configured token budget",
            )

        input_bound = input_token_upper_bound(body.messages)
        if input_bound > runtime_settings.max_input_tokens:
            raise HTTPException(
                status_code=413,
                detail="Input exceeds the configured token budget",
            )

        guard_text = "\n".join(
            f"<{message.role}>\n{message.content}"
            for message in body.messages
        )

        try:
            guard_result = await run_guard(
                guard,
                guard_semaphore,
                guard_text,
                wait_timeout=runtime_settings.queue_wait_timeout_seconds,
            )
        except GuardInputTooLongError:
            audit.log_security_decision(
                request_id=request_id,
                guard_decision="rejected_oversize",
                status_code=413,
                input_token_upper_bound=input_bound,
                error_category="guard_input_too_long",
            )
            logger.info(
                "guard_rejected_oversize request_id=%s input_bound=%d",
                request_id,
                input_bound,
            )
            raise HTTPException(
                status_code=413,
                detail="Input exceeds the guard token limit",
            ) from None
        except GuardUnavailableError:
            audit.log_security_decision(
                request_id=request_id,
                guard_decision="unavailable",
                status_code=503,
                input_token_upper_bound=input_bound,
                error_category="guard_unavailable",
            )
            logger.error("guard_unavailable request_id=%s", request_id)
            raise HTTPException(
                status_code=503,
                detail="Security guard is unavailable",
            ) from None
        except HTTPException:
            raise

        logger.info(
            "guard_evaluated request_id=%s decision=%s score=%.4f tokens=%d",
            request_id,
            "block" if guard_result.blocked else "allow",
            guard_result.score,
            guard_result.input_tokens,
        )

        if guard_result.blocked:
            audit.log_security_decision(
                request_id=request_id,
                guard_decision="block",
                guard_score=guard_result.score,
                guard_input_tokens=guard_result.input_tokens,
                input_token_upper_bound=input_bound,
                status_code=403,
            )
            raise HTTPException(
                status_code=403,
                detail="Prompt blocked by the security guard",
            )

        safe_messages: list[ChatMessage] = []
        redaction_count = 0

        for message in body.messages:
            safe_content, count = redactor.redact_text_with_count(
                message.content
            )
            redaction_count += count
            safe_messages.append(
                message.model_copy(update={"content": safe_content})
            )

        safe_body = body.model_copy(
            update={
                "messages": safe_messages,
                "max_tokens": max_tokens,
            }
        )

        safe_input_bound = input_token_upper_bound(safe_body.messages)
        if (
            safe_input_bound > runtime_settings.max_input_tokens
            or safe_input_bound + max_tokens > runtime_settings.max_model_len
        ):
            raise HTTPException(
                status_code=413,
                detail="Redacted input exceeds the configured context budget",
            )

        payload = safe_body.model_dump(mode="json", exclude_none=True)

        logger.info(
            "inference_started request_id=%s stream=%s input_bound=%d "
            "max_output_tokens=%d pii_redactions=%d",
            request_id,
            body.stream,
            safe_input_bound,
            max_tokens,
            redaction_count,
        )

        await acquire_slot(
            inference_semaphore,
            wait_seconds=runtime_settings.queue_wait_timeout_seconds,
        )

        if body.stream:
            try:
                stream_response = await upstream.open_chat_completion_stream(
                    payload,
                    request_id=request_id,
                )
            except UpstreamError as exc:
                inference_semaphore.release()
                logger.error(
                    "upstream_stream_open_failed request_id=%s status=%s",
                    request_id,
                    exc.status_code or "unavailable",
                )
                audit.log_security_decision(
                    request_id=request_id,
                    guard_decision="allow",
                    guard_score=guard_result.score,
                    guard_input_tokens=guard_result.input_tokens,
                    input_token_upper_bound=safe_input_bound,
                    pii_redactions=redaction_count,
                    status_code=503,
                    error_category="upstream_unavailable",
                )
                raise HTTPException(
                    status_code=503,
                    detail="Inference service is unavailable",
                ) from None

            released = False

            def release_inference_slot() -> None:
                nonlocal released
                if not released:
                    released = True
                    inference_semaphore.release()

            async def cleanup_stream() -> None:
                try:
                    await stream_response.aclose()
                finally:
                    release_inference_slot()

            async def stream_body() -> AsyncIterator[bytes]:
                outcome = "complete"

                try:
                    async for event in output_safety.filter_sse(
                        stream_response.aiter_lines()
                    ):
                        yield event
                except OutputStreamError:
                    outcome = "failed"
                    logger.error(
                        "inference_stream_failed request_id=%s",
                        request_id,
                    )
                    yield output_safety.error_event()
                except asyncio.CancelledError:
                    outcome = "client_disconnected"
                    logger.info(
                        "inference_stream_cancelled request_id=%s",
                        request_id,
                    )
                    raise
                finally:
                    audit.log_security_decision(
                        request_id=request_id,
                        guard_decision="allow",
                        guard_score=guard_result.score,
                        guard_input_tokens=guard_result.input_tokens,
                        input_token_upper_bound=safe_input_bound,
                        pii_redactions=redaction_count,
                        status_code=200,
                        error_category=(
                            None if outcome == "complete" else outcome
                        ),
                    )
                    logger.info(
                        "inference_stream_finished request_id=%s outcome=%s",
                        request_id,
                        outcome,
                    )
                    await cleanup_stream()

            return StreamingResponse(
                stream_body(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
                background=BackgroundTask(cleanup_stream),
            )

        try:
            raw_response = await upstream.chat_completion(
                payload,
                request_id=request_id,
            )
            safe_response = output_safety.sanitize_completion(raw_response)

        except UpstreamError as exc:
            logger.error(
                "upstream_inference_failed request_id=%s status=%s",
                request_id,
                exc.status_code or "unavailable",
            )
            audit.log_security_decision(
                request_id=request_id,
                guard_decision="allow",
                guard_score=guard_result.score,
                guard_input_tokens=guard_result.input_tokens,
                input_token_upper_bound=safe_input_bound,
                pii_redactions=redaction_count,
                status_code=503,
                error_category="upstream_error",
            )
            raise HTTPException(
                status_code=503,
                detail="Inference service is unavailable",
            ) from None

        except OutputSafetyError:
            logger.error(
                "unsafe_upstream_response request_id=%s",
                request_id,
            )
            audit.log_security_decision(
                request_id=request_id,
                guard_decision="allow",
                guard_score=guard_result.score,
                guard_input_tokens=guard_result.input_tokens,
                input_token_upper_bound=safe_input_bound,
                pii_redactions=redaction_count,
                status_code=503,
                error_category="unsafe_upstream_output",
            )
            raise HTTPException(
                status_code=503,
                detail="Inference response could not be safely validated",
            ) from None

        finally:
            inference_semaphore.release()

        usage = safe_response.get("usage")
        output_tokens = (
            usage.get("completion_tokens")
            if isinstance(usage, dict)
            else None
        )

        audit.log_security_decision(
            request_id=request_id,
            guard_decision="allow",
            guard_score=guard_result.score,
            guard_input_tokens=guard_result.input_tokens,
            input_token_upper_bound=safe_input_bound,
            output_tokens=output_tokens,
            pii_redactions=redaction_count,
            status_code=200,
        )

        logger.info(
            "inference_finished request_id=%s status_code=200 output_tokens=%s "
            "pii_redactions=%d",
            request_id,
            output_tokens if output_tokens is not None else "unknown",
            redaction_count,
        )

        return JSONResponse(content=safe_response)

    return app


configure_logging()
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=app.state.settings.gateway_host,
        port=app.state.settings.gateway_port,
        workers=1,
        access_log=False,
        log_level="info",
    )
