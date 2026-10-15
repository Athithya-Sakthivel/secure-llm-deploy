"""Structured metadata-only security audit events."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


class AuditLogger:
    def __init__(self) -> None:
        self.logger = logging.getLogger("secure_llm_deploy.audit")

    def _emit(self, event: dict[str, Any]) -> None:
        event["timestamp"] = datetime.now(UTC).isoformat(timespec="milliseconds")
        self.logger.info(
            json.dumps(
                event,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )

    def log_http_request(
        self,
        *,
        request_id: str,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        error_category: str | None = None

        if status_code == 400:
            error_category = "bad_request"
        elif status_code == 401:
            error_category = "authentication"
        elif status_code == 403:
            error_category = "guard_blocked"
        elif status_code == 413:
            error_category = "request_too_large"
        elif status_code == 422:
            error_category = "request_validation"
        elif status_code == 429:
            error_category = "rate_limit"
        elif status_code >= 500:
            error_category = "server_error"

        self._emit(
            {
                "event": "http_request",
                "request_id": request_id,
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": round(max(0.0, duration_ms), 2),
                "error_category": error_category,
            }
        )

    def log_security_decision(
        self,
        *,
        request_id: str,
        guard_decision: str,
        status_code: int,
        guard_score: float | None = None,
        guard_input_tokens: int | None = None,
        input_token_upper_bound: int | None = None,
        output_tokens: int | None = None,
        pii_redactions: int | None = None,
        error_category: str | None = None,
    ) -> None:
        self._emit(
            {
                "event": "inference_security_decision",
                "request_id": request_id,
                "guard_decision": guard_decision,
                "status_code": status_code,
                "guard_score": (
                    round(guard_score, 6)
                    if guard_score is not None
                    else None
                ),
                "guard_input_tokens": guard_input_tokens,
                "input_token_upper_bound": input_token_upper_bound,
                "output_tokens": output_tokens,
                "pii_redactions": pii_redactions,
                "error_category": error_category,
            }
        )

    def log_startup_failure(self) -> None:
        self._emit(
            {
                "event": "startup_failure",
                "error_category": "service_initialization",
            }
        )