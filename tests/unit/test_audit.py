"""Unit tests for metadata-only security audit logging."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from secure_llm_deploy.audit import AuditLogger


@pytest.fixture
def audit() -> AuditLogger:
    return AuditLogger()


def audit_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for record in caplog.records:
        if record.name != "secure_llm_deploy.audit":
            continue

        payload = json.loads(record.getMessage())
        assert isinstance(payload, dict)
        events.append(payload)

    return events


def test_http_request_event_contains_expected_metadata(
    audit: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="secure_llm_deploy.audit",
    ):
        audit.log_http_request(
            request_id="synthetic-request-001",
            method="POST",
            endpoint="chat_completions",
            status_code=200,
            duration_ms=123.456,
        )

    events = audit_events(caplog)
    assert len(events) == 1

    event = events[0]
    assert event["event"] == "http_request"
    assert event["request_id"] == "synthetic-request-001"
    assert event["method"] == "POST"
    assert event["endpoint"] == "chat_completions"
    assert event["status_code"] == 200
    assert event["duration_ms"] == pytest.approx(123.46)
    assert event["error_category"] is None
    assert "timestamp" in event


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    [
        (400, "bad_request"),
        (401, "authentication"),
        (403, "guard_blocked"),
        (413, "request_too_large"),
        (422, "request_validation"),
        (429, "rate_limit"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
def test_http_status_maps_to_error_category(
    audit: AuditLogger,
    caplog: pytest.LogCaptureFixture,
    status_code: int,
    expected_category: str,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="secure_llm_deploy.audit",
    ):
        audit.log_http_request(
            request_id="synthetic-request-002",
            method="POST",
            endpoint="chat_completions",
            status_code=status_code,
            duration_ms=1.0,
        )

    event = audit_events(caplog)[0]
    assert event["error_category"] == expected_category


def test_security_decision_event_contains_metadata_only(
    audit: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="secure_llm_deploy.audit",
    ):
        audit.log_security_decision(
            request_id="synthetic-request-003",
            guard_decision="allow",
            status_code=200,
            guard_score=0.012345678,
            guard_input_tokens=42,
            input_token_upper_bound=60,
            output_tokens=19,
            pii_redactions=2,
        )

    event = audit_events(caplog)[0]

    assert event["event"] == "inference_security_decision"
    assert event["guard_decision"] == "allow"
    assert event["status_code"] == 200
    assert event["guard_score"] == pytest.approx(0.012346)
    assert event["guard_input_tokens"] == 42
    assert event["input_token_upper_bound"] == 60
    assert event["output_tokens"] == 19
    assert event["pii_redactions"] == 2
    assert event["error_category"] is None


def test_missing_token_counts_remain_null(
    audit: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="secure_llm_deploy.audit",
    ):
        audit.log_security_decision(
            request_id="synthetic-request-004",
            guard_decision="block",
            status_code=403,
        )

    event = audit_events(caplog)[0]

    assert event["guard_input_tokens"] is None
    assert event["input_token_upper_bound"] is None
    assert event["output_tokens"] is None
    assert event["pii_redactions"] is None


def test_startup_failure_event_is_generic(
    audit: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="secure_llm_deploy.audit",
    ):
        audit.log_startup_failure()

    event = audit_events(caplog)[0]

    assert event["event"] == "startup_failure"
    assert event["error_category"] == "service_initialization"
    assert "traceback" not in event
    assert "exception" not in event


def test_audit_event_never_contains_prompt_or_credentials(
    audit: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt_canary = "PROMPT_CANARY_DO_NOT_LOG_001"
    response_canary = "RESPONSE_CANARY_DO_NOT_LOG_002"
    api_key_canary = "API_KEY_CANARY_DO_NOT_LOG_003"

    with caplog.at_level(
        logging.INFO,
        logger="secure_llm_deploy.audit",
    ):
        audit.log_http_request(
            request_id="synthetic-request-005",
            method="POST",
            endpoint="chat_completions",
            status_code=200,
            duration_ms=10,
        )
        audit.log_security_decision(
            request_id="synthetic-request-005",
            guard_decision="allow",
            status_code=200,
        )

    rendered_logs = caplog.text

    assert prompt_canary not in rendered_logs
    assert response_canary not in rendered_logs
    assert api_key_canary not in rendered_logs


def test_negative_duration_is_clamped_to_zero(
    audit: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="secure_llm_deploy.audit",
    ):
        audit.log_http_request(
            request_id="synthetic-request-006",
            method="GET",
            endpoint="health_ready",
            status_code=200,
            duration_ms=-10,
        )

    event = audit_events(caplog)[0]
    assert event["duration_ms"] == 0