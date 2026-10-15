"""Non-streaming and SSE response sanitation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from .pii import PiiRedactor


class OutputSafetyError(RuntimeError):
    """The output could not be validated or safely filtered."""


class OutputStreamError(RuntimeError):
    """The upstream stream was malformed, incomplete, or unsafe to forward."""


class StreamingTextRedactor:
    """Keep a bounded suffix until cross-chunk matches can be resolved."""

    def __init__(self, redactor: PiiRedactor) -> None:
        self.redactor = redactor
        self.pending = ""
        self.holdback = redactor.max_match_width

    def feed(self, text: str) -> str:
        self.pending += text
        planned_cut = len(self.pending) - self.holdback

        if planned_cut <= 1:
            return ""

        safe_end = planned_cut - 1

        for span in self.redactor.find_spans(self.pending):
            if span.start < planned_cut <= span.end:
                safe_end = min(safe_end, max(0, span.start - 1))

        if safe_end <= 0:
            return ""

        prefix = self.pending[:safe_end]
        self.pending = self.pending[safe_end:]
        return self.redactor.redact_text(prefix)

    def flush(self) -> str:
        tail = self.redactor.redact_text(self.pending)
        self.pending = ""
        return tail


class OutputSafety:
    def __init__(self, redactor: PiiRedactor) -> None:
        self.redactor = redactor

    @classmethod
    def _safe_usage(cls, value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            raise OutputSafetyError("Invalid usage object")

        fields = ("prompt_tokens", "completion_tokens", "total_tokens")
        usage: dict[str, int] = {}

        for field in fields:
            count = value.get(field)

            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise OutputSafetyError("Invalid usage values")

            usage[field] = count

        return usage

    def sanitize_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct a safe response; discard unknown upstream fields."""
        if "error" in payload:
            raise OutputSafetyError("Upstream returned an error object")

        completion_id = payload.get("id")
        object_type = payload.get("object")
        created = payload.get("created")
        model = payload.get("model")
        choices = payload.get("choices")

        if (
            not isinstance(completion_id, str)
            or not isinstance(object_type, str)
            or not isinstance(created, int)
            or isinstance(created, bool)
            or not isinstance(model, str)
            or not isinstance(choices, list)
            or len(choices) != 1
        ):
            raise OutputSafetyError("Invalid completion response")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise OutputSafetyError("Invalid completion choice")

        index = choice.get("index", 0)
        finish_reason = choice.get("finish_reason")
        message = choice.get("message")

        if (
            index != 0
            or not isinstance(message, dict)
            or message.get("role") != "assistant"
            or (finish_reason is not None and not isinstance(finish_reason, str))
        ):
            raise OutputSafetyError("Invalid completion message")

        if "tool_calls" in message or "function_call" in message:
            raise OutputSafetyError("Tool-call output is not supported")

        safe_message: dict[str, Any] = {"role": "assistant"}

        for field in ("content", "refusal"):
            value = message.get(field)
            if value is not None and not isinstance(value, str):
                raise OutputSafetyError("Invalid message text field")
            if isinstance(value, str):
                safe_message[field] = self.redactor.redact_text(value)

        safe_choice = {
            "index": 0,
            "message": safe_message,
            "finish_reason": finish_reason,
        }

        result: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [safe_choice],
        }

        usage = payload.get("usage")
        if usage is not None:
            result["usage"] = self._safe_usage(usage)

        fingerprint = payload.get("system_fingerprint")
        if isinstance(fingerprint, str):
            result["system_fingerprint"] = fingerprint

        return result

    @staticmethod
    def _event(payload: dict[str, Any]) -> bytes:
        data = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return f"data: {data}\n\n".encode()

    @staticmethod
    def _done_event() -> bytes:
        return b"data: [DONE]\n\n"

    @staticmethod
    def _usage_chunk(
        header: dict[str, Any],
        usage: dict[str, int],
    ) -> dict[str, Any]:
        return {
            **header,
            "choices": [],
            "usage": usage,
        }

    @staticmethod
    def _text_chunk(
        header: dict[str, Any],
        field: str,
        text: str,
    ) -> dict[str, Any]:
        return {
            **header,
            "choices": [
                {
                    "index": 0,
                    "delta": {field: text},
                    "finish_reason": None,
                }
            ],
        }

    @staticmethod
    def _finish_chunk(
        header: dict[str, Any],
        finish_reason: str,
    ) -> dict[str, Any]:
        return {
            **header,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _safe_header(self, payload: dict[str, Any]) -> dict[str, Any]:
        completion_id = payload.get("id")
        created = payload.get("created")
        model = payload.get("model")

        if (
            not isinstance(completion_id, str)
            or not isinstance(created, int)
            or isinstance(created, bool)
            or not isinstance(model, str)
        ):
            raise OutputStreamError("Invalid stream metadata")

        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }

    async def filter_sse(self, lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Parse upstream SSE and emit only sanitized chat-completion chunks."""
        scanners = {
            "content": StreamingTextRedactor(self.redactor),
            "refusal": StreamingTextRedactor(self.redactor),
        }
        header: dict[str, Any] | None = None
        event_lines: list[str] = []
        seen_done = False

        async def consume_data(data: str) -> list[bytes]:
            nonlocal header, seen_done

            if data == "[DONE]":
                if header is None:
                    raise OutputStreamError("Stream ended before its first chunk")

                events: list[bytes] = []
                for field, scanner in scanners.items():
                    tail = scanner.flush()
                    if tail:
                        events.append(self._event(self._text_chunk(header, field, tail)))

                events.append(self._done_event())
                seen_done = True
                return events

            try:
                raw_payload = json.loads(data)
            except json.JSONDecodeError:
                raise OutputStreamError("Invalid JSON in upstream stream") from None

            if not isinstance(raw_payload, dict) or "error" in raw_payload:
                raise OutputStreamError("Invalid upstream stream event")

            header = self._safe_header(raw_payload)
            choices = raw_payload.get("choices")

            if not isinstance(choices, list):
                raise OutputStreamError("Invalid upstream stream choices")

            if not choices:
                usage = raw_payload.get("usage")
                if usage is None:
                    return []
                return [self._event(self._usage_chunk(header, self._safe_usage(usage)))]

            if len(choices) != 1 or not isinstance(choices[0], dict):
                raise OutputStreamError("Only one streaming choice is supported")

            choice = choices[0]
            if choice.get("index", 0) != 0:
                raise OutputStreamError("Invalid stream choice index")

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise OutputStreamError("Invalid stream delta")

            if "tool_calls" in delta or "function_call" in delta:
                raise OutputStreamError("Tool-call streaming is not supported")

            safe_delta: dict[str, str] = {}
            role = delta.get("role")
            if role is not None:
                if role != "assistant":
                    raise OutputStreamError("Invalid streamed message role")
                safe_delta["role"] = role

            for field in ("content", "refusal"):
                value = delta.get(field)
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise OutputStreamError("Invalid streamed text field")
                safe_text = scanners[field].feed(value)
                if safe_text:
                    safe_delta[field] = safe_text

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise OutputStreamError("Invalid stream finish reason")

            events = []

            if finish_reason is None:
                if safe_delta:
                    events.append(
                        self._event(
                            {
                                **header,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": safe_delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    )
                return events

            if safe_delta:
                events.append(
                    self._event(
                        {
                            **header,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": safe_delta,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                )

            for field, scanner in scanners.items():
                tail = scanner.flush()
                if tail:
                    events.append(self._event(self._text_chunk(header, field, tail)))

            events.append(self._event(self._finish_chunk(header, finish_reason)))
            return events

        try:
            async for line in lines:
                if seen_done:
                    if line.strip():
                        raise OutputStreamError("Data received after stream completion")
                    continue

                if line == "":
                    if not event_lines:
                        continue

                    data_lines = [
                        item[5:].lstrip()
                        for item in event_lines
                        if item.startswith("data:")
                    ]
                    event_lines.clear()

                    if not data_lines:
                        continue

                    data = "\n".join(data_lines)
                    for event in await consume_data(data):
                        yield event
                elif line.startswith(":"):
                    # Ignore server-sent event comments / keep-alive lines.
                    continue
                elif line.startswith(("data:", "event:", "id:", "retry:")):
                    # Only the data field is used to build a sanitized event.
                    event_lines.append(line)
                else:
                    raise OutputStreamError("Unsupported SSE field")

            if not seen_done:
                # Do not flush potentially incomplete buffered text on a broken
                # stream. It is safer to lose the buffered suffix than emit it.
                raise OutputStreamError("Upstream stream ended before [DONE]")

        except OutputStreamError:
            raise
        except Exception:
            # Includes upstream read failures. Never expose the raw exception.
            raise OutputStreamError("Upstream stream failed") from None

    @staticmethod
    def error_event() -> bytes:
        return OutputSafety._event(
            {
                "error": {
                    "message": "Inference stream failed",
                    "type": "server_error",
                    "code": "upstream_stream_error",
                }
            }
        )