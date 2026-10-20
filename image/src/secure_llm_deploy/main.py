"""Secure LLM gateway application.

The gateway owns authentication, request bounds, rate limiting, prompt guard
policy, PII redaction, output safety, audit metadata and the vLLM lifecycle.
It never logs prompts, completions, authorization headers or model tensor data.
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

from . import __version__
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
from .output_safety import OutputSafety, OutputSafetyError
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
    """Configure console logs while keeping framework access logs quiet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=False,
    )
    for name in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def endpoint_label(method: str, path: str) -> str:
    """Use stable endpoint labels rather than arbitrary URL paths in logs."""
    return ROUTE_LABELS.get((method, path), "other")


def error_payload(message: str, *, error_type: str, code: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": code}}


async def send_json(
    send: Send,
    *,
    status_code: int,
    payload: dict[str, Any],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    """Send a small JSON response directly from ASGI middleware."""
    body = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    response_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if headers:
        response_headers.extend(headers)
    await send(
        {"type": "http.response.start", "status": status_code, "headers": response_headers}
    )
    await send(
        {"type": "http.response.body", "body": body, "more_body": False}
    )


class GatewaySecurityMiddleware:
    """Default-deny authentication and bounded ASGI request-body handling."""

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
        return [value for key, value in scope.get("headers", []) if key.lower() == name]

    @staticmethod
    def _client_identity(scope: Scope) -> str:
        client = scope.get("client")
        if not client or not client[0]:
            return "unknown"
        # Deliberately do not trust caller-supplied X-Forwarded-For headers.
        return str(client[0])

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "/"))
        endpoint = endpoint_label(method, path)
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        state = scope.get("state")
        if not isinstance(state, dict):
            state = {}
            scope["state"] = state
        state["request_id"] = request_id

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
                "request_rejected request_id=%s endpoint=%s status_code=%d reason=%s",
                request_id,
                endpoint,
                code,
                error_code,
            )
            await send_json(
                wrapped_send,
                status_code=code,
                payload=error_payload(message, error_type=error_type, code=error_code),
                headers=headers,
            )

        try:
            if (method, path) not in PUBLIC_HEALTH_ROUTES:
                auth_headers = self._header_values(scope, b"authorization")
                authorization = (
                    auth_headers[0].decode("latin-1") if len(auth_headers) == 1 else None
                )
                if not authenticate_bearer_token(
                    authorization, self.settings.gateway_api_key
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
                        headers=[(b"retry-after", str(retry_after).encode("ascii"))],
                    )
                    return

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
                except (ValueError, TypeError):
                    await reject(
                        code=400,
                        message="Invalid Content-Length",
                        error_type="invalid_request_error",
                        error_code="invalid_content_length",
                    )
                    return
                if content_length < 0 or content_length > self.settings.request_max_body_bytes:
                    await reject(
                        code=413,
                        message="Request body too large",
                        error_type="invalid_request_error",
                        error_code="request_too_large",
                    )
                    return

            # Buffer up to the configured limit before FastAPI's JSON parser.
            # This also enforces the limit when Content-Length is absent/understated.
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
            if not response_completed:
                log_completion()
        except Exception:
            logger.exception(
                "middleware_failure request_id=%s endpoint=%s",
                request_id,
                endpoint,
            )
            if response_started:
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
    """Return a conservative UTF-8 byte count for audit metadata, not tokenization.

    The actual guard tokenizer count is authoritative for gateway budgets. This
    byte count is used only as conservative request-size metadata before inference.
    """
    text = "\n".join(f"<{message.role}>\n{message.content}" for message in messages)
    return len(text.encode("utf-8"))


def _format_guard_text(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(f"<{message.role}>\n{message.content}" for message in messages)


async def acquire_slot(semaphore: asyncio.Semaphore, *, wait_seconds: float) -> None:
    """Acquire a bounded worker slot or return a safe 503 response."""
    try:
        async with asyncio.timeout(wait_seconds):
            await semaphore.acquire()
    except TimeoutError:
        raise HTTPException(
            status_code=503, detail="Service is at concurrency capacity"
        ) from None


async def run_guard(
    guard: ONNXPromptGuard,
    semaphore: asyncio.Semaphore,
    text: str,
    *,
    wait_timeout: float,
) -> GuardResult:
    """Run synchronous tokenizer/ONNX work off-loop with cancellation-safe slots."""
    await acquire_slot(semaphore, wait_seconds=wait_timeout)
    try:
        task = asyncio.create_task(asyncio.to_thread(guard.classify, text))
    except Exception:
        semaphore.release()
        raise

    release_in_callback = False
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Threads cannot be force-cancelled. Retain capacity until the task ends.
        release_in_callback = True
        task.add_done_callback(lambda _task: semaphore.release())
        raise
    finally:
        if not release_in_callback:
            semaphore.release()


async def count_guard_tokens(
    guard: ONNXPromptGuard,
    semaphore: asyncio.Semaphore,
    text: str,
    *,
    wait_timeout: float,
) -> int:
    """Count tokens using the same tokenizer that scored the prompt."""
    await acquire_slot(semaphore, wait_seconds=wait_timeout)
    try:
        encoding = await asyncio.to_thread(
            guard.tokenizer.encode, text, add_special_tokens=False
        )
        return len(encoding.ids)
    except Exception:
        raise HTTPException(
            status_code=503, detail="Unable to validate the inference token budget"
        ) from None
    finally:
        semaphore.release()


def _safe_upstream_status(exc: UpstreamError) -> tuple[int, str]:
    """Map upstream errors to a non-sensitive downstream status/message."""
    if exc.status_code == 413:
        return 413, "Inference request exceeds the model's request limit"
    if exc.status_code in {400, 422}:
        return 422, "Inference request was rejected"
    return 503, "Inference service is unavailable"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the gateway application and bind all dependencies to its lifespan."""
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
                runtime_settings.guard_config_path,
                threshold=runtime_settings.guard_threshold,
                max_input_tokens=runtime_settings.guard_max_input_tokens,
                window_tokens=runtime_settings.guard_window_tokens,
                intra_op_threads=runtime_settings.guard_intra_op_threads,
                failure_policy=runtime_settings.guard_failure_policy,
                overlong_policy=runtime_settings.guard_overlong_policy,
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
            await upstream.wait_until_ready(process_is_running=process.is_running)
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
        version=__version__,
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
        request: Request, _exc: RequestValidationError
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
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
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
            429: ("Rate limit exceeded", "rate_limit_error", "rate_limit_exceeded"),
            503: (
                "Inference/security service is unavailable",
                "server_error",
                "service_unavailable",
            ),
        }
        message, error_type, code = safe_errors.get(
            exc.status_code, ("Request failed", "server_error", "request_failed")
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
            content=error_payload(message, error_type=error_type, code=code),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, _exc: Exception
    ) -> JSONResponse:
        logger.error(
            "unhandled_application_error request_id=%s endpoint=%s",
            getattr(request.state, "request_id", "unknown"),
            endpoint_label(request.method, request.url.path),
        )
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "Internal server error", error_type="server_error", code="internal_error"
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
        body: ChatCompletionRequest, request: Request
    ) -> Response:
        request_id: str = request.state.request_id
        guard: ONNXPromptGuard | None = request.app.state.guard
        upstream: VLLMUpstream | None = request.app.state.upstream
        process: VLLMProcess | None = request.app.state.vllm_process
        redactor: PiiRedactor | None = request.app.state.redactor
        output_safety: OutputSafety | None = request.app.state.output_safety
        guard_semaphore: asyncio.Semaphore | None = request.app.state.guard_semaphore
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
                status_code=413, detail="Message exceeds the configured character limit"
            )
        if sum(len(message.content) for message in body.messages) > runtime_settings.max_input_chars:
            raise HTTPException(
                status_code=413, detail="Input exceeds the configured character limit"
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

        guard_text = _format_guard_text(body.messages)
        input_bound = input_token_upper_bound(body.messages)
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
            logger.info("guard_rejected_oversize request_id=%s", request_id)
            raise HTTPException(
                status_code=413, detail="Input exceeds the guard token limit"
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
                status_code=503, detail="Security guard is unavailable"
            ) from None

        # A partial scan must never authorize the complete, unscanned prompt.
        if guard_result.truncated:
            audit.log_security_decision(
                request_id=request_id,
                guard_decision="rejected_truncated_scan",
                guard_score=guard_result.score,
                guard_input_tokens=guard_result.tokens_scanned,
                input_token_upper_bound=input_bound,
                status_code=413,
                error_category="guard_scan_truncated",
            )
            raise HTTPException(
                status_code=413,
                detail="Input exceeds the complete prompt-guard scan budget",
            )
        if guard_result.input_tokens > runtime_settings.max_input_tokens:
            audit.log_security_decision(
                request_id=request_id,
                guard_decision="rejected_oversize",
                guard_score=guard_result.score,
                guard_input_tokens=guard_result.input_tokens,
                input_token_upper_bound=input_bound,
                status_code=413,
                error_category="input_token_budget_exceeded",
            )
            raise HTTPException(
                status_code=413,
                detail="Input exceeds the configured token budget",
            )

        # Fail-closed inference errors are represented by a block result with no
        # score; return 503, not a false assertion that the prompt was malicious.
        # Fail-open is honoured only when tokenization still produced a usable
        # count, so an uncountable prompt cannot bypass context limits.
        if guard_result.error_code is not None:
            if guard_result.input_tokens <= 0 and guard_text.strip():
                audit.log_security_decision(
                    request_id=request_id,
                    guard_decision="unavailable_unbounded_input",
                    status_code=503,
                    input_token_upper_bound=input_bound,
                    error_category="guard_tokenization_unavailable",
                )
                raise HTTPException(
                    status_code=503,
                    detail="Security guard could not validate the input budget",
                )
            audit.log_security_decision(
                request_id=request_id,
                guard_decision=(
                    "unavailable_block" if guard_result.blocked else "unavailable_allow"
                ),
                guard_score=None,
                guard_input_tokens=guard_result.input_tokens,
                input_token_upper_bound=input_bound,
                status_code=503 if guard_result.blocked else 200,
                error_category=guard_result.error_code,
            )
            logger.warning(
                "guard_inference_unavailable request_id=%s policy=%s",
                request_id,
                runtime_settings.guard_failure_policy,
            )
            if guard_result.blocked:
                raise HTTPException(
                    status_code=503, detail="Security guard is unavailable"
                )
        elif guard_result.blocked:
            audit.log_security_decision(
                request_id=request_id,
                guard_decision="block",
                guard_score=guard_result.score,
                guard_input_tokens=guard_result.input_tokens,
                input_token_upper_bound=input_bound,
                status_code=403,
            )
            logger.info(
                "guard_blocked request_id=%s score=%s tokens=%d",
                request_id,
                f"{guard_result.score:.4f}" if guard_result.score is not None else "unknown",
                guard_result.input_tokens,
            )
            raise HTTPException(
                status_code=403, detail="Prompt blocked by the security guard"
            )

        score_text = (
            f"{guard_result.score:.4f}" if guard_result.score is not None else "unavailable"
        )
        logger.info(
            "guard_evaluated request_id=%s decision=%s score=%s tokens=%d chunks=%d",
            request_id,
            "block" if guard_result.blocked else "allow",
            score_text,
            guard_result.input_tokens,
            guard_result.chunks_evaluated,
        )

        safe_messages: list[ChatMessage] = []
        redaction_count = 0
        for message in body.messages:
            safe_content, count = redactor.redact_text_with_count(message.content)
            redaction_count += count
            safe_messages.append(message.model_copy(update={"content": safe_content}))
        safe_body = body.model_copy(
            update={"messages": safe_messages, "max_tokens": max_tokens}
        )
        safe_guard_text = _format_guard_text(safe_body.messages)
        safe_input_tokens = await count_guard_tokens(
            guard,
            guard_semaphore,
            safe_guard_text,
            wait_timeout=runtime_settings.queue_wait_timeout_seconds,
        )
        safe_input_bound = len(safe_guard_text.encode("utf-8"))
        if safe_input_tokens > runtime_settings.max_input_tokens:
            raise HTTPException(
                status_code=413,
                detail="Redacted input exceeds the configured token budget",
            )
        if safe_input_tokens + max_tokens > runtime_settings.max_model_len:
            raise HTTPException(
                status_code=413,
                detail="Input and output exceed the configured model context length",
            )

        payload = safe_body.model_dump(mode="json", exclude_none=True)
        logger.info(
            "inference_started request_id=%s stream=%s input_tokens=%d "
            "max_output_tokens=%d pii_redactions=%d",
            request_id,
            body.stream,
            safe_input_tokens,
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
                    payload, request_id=request_id
                )
            except UpstreamError as exc:
                inference_semaphore.release()
                status, upstream_message = _safe_upstream_status(exc)
                audit.log_security_decision(
                    request_id=request_id,
                    guard_decision="allow",
                    guard_score=guard_result.score,
                    guard_input_tokens=guard_result.input_tokens,
                    input_token_upper_bound=safe_input_bound,
                    pii_redactions=redaction_count,
                    status_code=status,
                    error_category="upstream_unavailable",
                )
                raise HTTPException(status_code=status, detail=upstream_message) from None
            except Exception:
                inference_semaphore.release()
                raise HTTPException(
                    status_code=503, detail="Inference service is unavailable"
                ) from None

            released = False
            closed = False

            async def cleanup_stream() -> None:
                nonlocal released, closed
                if closed:
                    return
                closed = True
                try:
                    await stream_response.aclose()
                except Exception:
                    logger.warning("upstream_stream_close_failed request_id=%s", request_id)
                finally:
                    if not released:
                        released = True
                        inference_semaphore.release()

            async def stream_body() -> AsyncIterator[bytes | str]:
                outcome = "complete"
                try:
                    async for event in output_safety.filter_sse(
                        stream_response.aiter_lines()
                    ):
                        yield event
                except asyncio.CancelledError:
                    outcome = "client_disconnected"
                    logger.info("inference_stream_cancelled request_id=%s", request_id)
                    raise
                except Exception:
                    outcome = "failed"
                    logger.error("inference_stream_failed request_id=%s", request_id)
                    yield output_safety.error_event()
                finally:
                    audit.log_security_decision(
                        request_id=request_id,
                        guard_decision=(
                            "unavailable_allow" if guard_result.error_code else "allow"
                        ),
                        guard_score=guard_result.score,
                        guard_input_tokens=guard_result.input_tokens,
                        input_token_upper_bound=safe_input_bound,
                        pii_redactions=redaction_count,
                        status_code=200,
                        error_category=(None if outcome == "complete" else outcome),
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
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                background=BackgroundTask(cleanup_stream),
            )

        try:
            raw_response = await upstream.chat_completion(payload, request_id=request_id)
            safe_response = output_safety.sanitize_completion(raw_response)
        except UpstreamError as exc:
            status, upstream_message = _safe_upstream_status(exc)
            logger.error(
                "upstream_inference_failed request_id=%s status=%s",
                request_id,
                exc.status_code or "unavailable",
            )
            audit.log_security_decision(
                request_id=request_id,
                guard_decision=(
                    "unavailable_allow" if guard_result.error_code else "allow"
                ),
                guard_score=guard_result.score,
                guard_input_tokens=guard_result.input_tokens,
                input_token_upper_bound=safe_input_bound,
                pii_redactions=redaction_count,
                status_code=status,
                error_category="upstream_error",
            )
            raise HTTPException(status_code=status, detail=upstream_message) from None
        except OutputSafetyError:
            logger.error("unsafe_upstream_response request_id=%s", request_id)
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
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        audit.log_security_decision(
            request_id=request_id,
            guard_decision=(
                "unavailable_allow" if guard_result.error_code else "allow"
            ),
            guard_score=guard_result.score,
            guard_input_tokens=guard_result.input_tokens,
            input_token_upper_bound=safe_input_bound,
            output_tokens=output_tokens,
            pii_redactions=redaction_count,
            status_code=200,
            error_category=guard_result.error_code,
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
