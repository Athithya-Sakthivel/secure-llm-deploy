"""Unit tests for non-streaming and SSE output safety."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from secure_llm_deploy.output_safety import (
    OutputSafety,
    OutputSafetyError,
    OutputStreamError,
)
from secure_llm_deploy.pii import PiiRedactor


@pytest.fixture
def safety() -> OutputSafety:
    return OutputSafety(
        PiiRedactor(("SYNTHETIC_SECRET_ALPHA_001",))
    )


def valid_completion(content: str = "Hello.") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "secure-llm-deploy-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def chunk(
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "secure-llm-deploy-test",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def sse_lines(*payloads: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    for payload in payloads:
        lines.extend(
            [
                f"data: {json.dumps(payload, separators=(',', ':'))}",
                "",
            ]
        )

    lines.extend(["data: [DONE]", ""])
    return lines


async def async_lines(lines: list[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def collect_stream(
    safety: OutputSafety,
    lines: list[str],
) -> list[bytes]:
    return [
        event
        async for event in safety.filter_sse(async_lines(lines))
    ]


def decode_events(events: list[bytes]) -> list[dict[str, Any]]:
    """Decode generated JSON SSE events, excluding [DONE] and error events."""
    decoded: list[dict[str, Any]] = []

    for event in events:
        for line in event.decode("utf-8").splitlines():
            if not line.startswith("data:"):
                continue

            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                continue

            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict) and "choices" in payload:
                decoded.append(payload)

    return decoded


def extract_text(
    events: list[dict[str, Any]],
    field: str,
) -> str:
    text_parts: list[str] = []

    for event in events:
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            continue

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            value = delta.get(field)
            if isinstance(value, str):
                text_parts.append(value)

    return "".join(text_parts)


def test_non_streaming_pii_is_redacted(
    safety: OutputSafety,
) -> None:
    payload = valid_completion(
        "Contact alice@example.com. Reference SYNTHETIC_SECRET_ALPHA_001."
    )

    result = safety.sanitize_completion(payload)
    content = result["choices"][0]["message"]["content"]

    assert "alice@example.com" not in content
    assert "SYNTHETIC_SECRET_ALPHA_001" not in content
    assert "[REDACTED_EMAIL]" in content
    assert "[REDACTED_SECRET]" in content


def test_non_streaming_response_preserves_valid_usage(
    safety: OutputSafety,
) -> None:
    result = safety.sanitize_completion(valid_completion())

    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_unrecognized_upstream_fields_are_not_forwarded(
    safety: OutputSafety,
) -> None:
    payload = valid_completion()
    payload["internal_metadata"] = "SYNTHETIC_SECRET_ALPHA_001"
    payload["choices"][0]["internal_field"] = "must not escape"

    result = safety.sanitize_completion(payload)

    assert "internal_metadata" not in result
    assert "internal_field" not in result["choices"][0]


def test_unrecognized_message_fields_are_not_forwarded(
    safety: OutputSafety,
) -> None:
    payload = valid_completion()
    payload["choices"][0]["message"]["vendor_metadata"] = "internal"

    result = safety.sanitize_completion(payload)

    assert "vendor_metadata" not in result["choices"][0]["message"]


def test_tool_call_output_is_rejected(
    safety: OutputSafety,
) -> None:
    payload = valid_completion()
    payload["choices"][0]["message"]["tool_calls"] = []

    with pytest.raises(OutputSafetyError, match="Tool-call output"):
        safety.sanitize_completion(payload)


def test_function_call_output_is_rejected(
    safety: OutputSafety,
) -> None:
    payload = valid_completion()
    payload["choices"][0]["message"]["function_call"] = {}

    with pytest.raises(OutputSafetyError, match="Tool-call output"):
        safety.sanitize_completion(payload)


@pytest.mark.parametrize(
    "field",
    ["id", "object", "created", "model", "choices"],
)
def test_missing_required_top_level_field_is_rejected(
    safety: OutputSafety,
    field: str,
) -> None:
    payload = valid_completion()
    del payload[field]

    with pytest.raises(OutputSafetyError):
        safety.sanitize_completion(payload)


def test_more_than_one_choice_is_rejected(
    safety: OutputSafety,
) -> None:
    payload = valid_completion()
    payload["choices"].append(payload["choices"][0])

    with pytest.raises(OutputSafetyError):
        safety.sanitize_completion(payload)


def test_invalid_usage_counts_are_rejected(
    safety: OutputSafety,
) -> None:
    payload = valid_completion()
    payload["usage"]["prompt_tokens"] = -1

    with pytest.raises(OutputSafetyError, match="Invalid usage"):
        safety.sanitize_completion(payload)


def test_boolean_usage_counts_are_rejected(
    safety: OutputSafety,
) -> None:
    payload = valid_completion()
    payload["usage"]["completion_tokens"] = True

    with pytest.raises(OutputSafetyError, match="Invalid usage"):
        safety.sanitize_completion(payload)

@pytest.mark.asyncio
async def test_sse_preserves_normal_completion_text(
    safety: OutputSafety,
) -> None:
    lines = sse_lines(
        chunk({"role": "assistant"}),
        chunk({"content": "Hello "}),
        chunk({"content": "world."}),
        chunk({}, finish_reason="stop"),
    )

    raw_events = await collect_stream(safety, lines)
    decoded = decode_events(raw_events)

    assert extract_text(decoded, "content") == "Hello world."
    assert raw_events[-1] == b"data: [DONE]\n\n"

@pytest.mark.asyncio
async def test_sensitive_string_split_across_chunks_is_redacted(
    safety: OutputSafety,
) -> None:
    secret = "SYNTHETIC_SECRET_ALPHA_001"
    text = ("ordinary-prefix " * 30) + secret + (" ordinary-suffix" * 50)

    # Split within the secret and use enough trailing text to force an earlier
    # prefix to be emitted. This exercises incremental rather than final-only
    # redaction.
    secret_start = text.index(secret)
    first_cut = secret_start + 7
    second_cut = secret_start + 16

    lines = sse_lines(
        chunk({"role": "assistant"}),
        chunk({"content": text[:first_cut]}),
        chunk({"content": text[first_cut:second_cut]}),
        chunk({"content": text[second_cut:]}),
        chunk({}, finish_reason="stop"),
    )

    raw_events = await collect_stream(safety, lines)
    decoded = decode_events(raw_events)
    full_output = b"".join(raw_events).decode("utf-8")

    assert secret not in full_output
    assert "[REDACTED_SECRET]" in full_output
    assert extract_text(decoded, "content") == (
        ("ordinary-prefix " * 30)
        + "[REDACTED_SECRET]"
        + (" ordinary-suffix" * 50)
    )


@pytest.mark.asyncio
async def test_stream_content_and_refusal_use_separate_buffers(
    safety: OutputSafety,
) -> None:
    content = "CONTENT_MARKER_" + ("C" * 600)
    refusal = "REFUSAL_MARKER_" + ("R" * 600)

    lines = sse_lines(
        chunk({"role": "assistant"}),
        chunk({"content": content, "refusal": refusal}),
        chunk({}, finish_reason="stop"),
    )

    raw_events = await collect_stream(safety, lines)
    decoded = decode_events(raw_events)

    assert extract_text(decoded, "content") == content
    assert extract_text(decoded, "refusal") == refusal
    assert "CONTENT_MARKER_" not in extract_text(decoded, "refusal")
    assert "REFUSAL_MARKER_" not in extract_text(decoded, "content")


@pytest.mark.asyncio
async def test_valid_usage_chunk_is_preserved(
    safety: OutputSafety,
) -> None:
    usage_chunk = {
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "secure-llm-deploy-test",
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    events = await collect_stream(
        safety,
        sse_lines(
            chunk({"role": "assistant"}),
            chunk({"content": "Done"}),
            chunk({}, finish_reason="stop"),
            usage_chunk,
        )[:-2] + ["data: [DONE]", ""],
    )

    usage_payloads = [
        payload
        for payload in decode_events(events)
        if payload.get("choices") == []
    ]

    assert usage_payloads
    assert usage_payloads[0]["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_malformed_sse_json_fails_safely(
    safety: OutputSafety,
) -> None:
    with pytest.raises(OutputStreamError, match="Invalid JSON"):
        await collect_stream(
            safety,
            ["data: {not-valid-json", ""],
        )


@pytest.mark.asyncio
async def test_upstream_error_event_fails_safely(
    safety: OutputSafety,
) -> None:
    with pytest.raises(OutputStreamError):
        await collect_stream(
            safety,
            [
                'data: {"error":{"message":"internal upstream detail"}}',
                "",
            ],
        )


@pytest.mark.asyncio
async def test_stream_without_done_marker_is_rejected(
    safety: OutputSafety,
) -> None:
    with pytest.raises(
        OutputStreamError,
        match="ended before \\[DONE\\]",
    ):
        await collect_stream(
            safety,
            sse_lines(
                chunk({"role": "assistant"}),
                chunk({"content": "partial output"}),
            )[:-2],
        )


@pytest.mark.asyncio
async def test_done_before_any_chunk_is_rejected(
    safety: OutputSafety,
) -> None:
    with pytest.raises(OutputStreamError):
        await collect_stream(
            safety,
            ["data: [DONE]", ""],
        )


def test_error_event_does_not_include_exception_details() -> None:
    raw = OutputSafety.error_event().decode("utf-8")

    assert "Inference stream failed" in raw
    assert "Traceback" not in raw
    assert "SYNTHETIC_SECRET_ALPHA_001" not in raw