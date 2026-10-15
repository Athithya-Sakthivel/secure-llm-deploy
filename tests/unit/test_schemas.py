"""Unit tests for the supported OpenAI-compatible request schemas."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from secure_llm_deploy.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    StreamOptions,
)


def valid_payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid chat request with optional top-level overrides."""
    payload: dict[str, Any] = {
        "model": "secure-llm-deploy-test",
        "messages": [
            {
                "role": "user",
                "content": "Explain prompt injection.",
            }
        ],
        "stream": False,
    }
    payload.update(overrides)
    return payload


def test_valid_minimal_request_is_accepted() -> None:
    request = ChatCompletionRequest.model_validate(valid_payload())

    assert request.model == "secure-llm-deploy-test"
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.stream is False
    assert request.n == 1


def test_valid_request_accepts_supported_generation_parameters() -> None:
    request = ChatCompletionRequest.model_validate(
        valid_payload(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Explain input validation."},
            ],
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            stop=["END", "STOP"],
            presence_penalty=0.1,
            frequency_penalty=0.2,
            seed=123,
        )
    )

    assert request.temperature == pytest.approx(0.2)
    assert request.top_p == pytest.approx(0.9)
    assert request.max_tokens == 256
    assert request.stop == ["END", "STOP"]
    assert request.presence_penalty == pytest.approx(0.1)
    assert request.frequency_penalty == pytest.approx(0.2)
    assert request.seed == 123


def test_valid_stream_options_require_streaming() -> None:
    request = ChatCompletionRequest.model_validate(
        valid_payload(
            stream=True,
            stream_options={"include_usage": True},
        )
    )

    assert request.stream is True
    assert isinstance(request.stream_options, StreamOptions)
    assert request.stream_options.include_usage is True


@pytest.mark.parametrize(
    "role",
    ["system", "developer", "user", "assistant"],
)
def test_supported_message_roles_are_accepted(role: str) -> None:
    message = ChatMessage.model_validate(
        {"role": role, "content": "Test content."}
    )

    assert message.role == role


@pytest.mark.parametrize(
    "role",
    ["tool", "function", "unknown", "", None],
)
def test_unsupported_message_roles_are_rejected(role: Any) -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(
            {"role": role, "content": "Test content."}
        )


@pytest.mark.parametrize(
    "content",
    ["", " ", "\t", "\n", "  \n  "],
)
def test_blank_message_content_is_rejected(content: str) -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(
            {"role": "user", "content": content}
        )


def test_message_without_content_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate({"role": "user"})


def test_message_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": "Hello.",
                "tool_calls": [],
            }
        )


def test_request_requires_at_least_one_message() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(messages=[])
        )


def test_request_requires_at_least_one_user_message() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(
                messages=[
                    {
                        "role": "assistant",
                        "content": "Previous assistant response.",
                    }
                ]
            )
        )


def test_request_rejects_more_than_32_messages() -> None:
    messages = [
        {"role": "user", "content": f"Message {index}"}
        for index in range(33)
    ]

    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(messages=messages)
        )


def test_request_rejects_message_over_schema_character_limit() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(
                messages=[
                    {
                        "role": "user",
                        "content": "x" * 60001,
                    }
                ]
            )
        )


@pytest.mark.parametrize(
    "max_tokens",
    [0, -1, 4097],
)
def test_invalid_max_tokens_are_rejected(max_tokens: int) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(max_tokens=max_tokens)
        )


@pytest.mark.parametrize(
    "temperature",
    [-0.01, 2.01],
)
def test_temperature_outside_allowed_range_is_rejected(
    temperature: float,
) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(temperature=temperature)
        )


@pytest.mark.parametrize(
    "top_p",
    [0.0, -0.1, 1.01],
)
def test_top_p_outside_allowed_range_is_rejected(top_p: float) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(top_p=top_p)
        )


@pytest.mark.parametrize(
    "field",
    ["presence_penalty", "frequency_penalty"],
)
@pytest.mark.parametrize(
    "value",
    [-2.1, 2.1],
)
def test_penalties_outside_allowed_range_are_rejected(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(**{field: value})
        )


def test_only_one_generation_choice_is_supported() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(n=2)
        )


def test_seed_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(seed=-1)
        )


def test_unknown_top_level_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(
                tools=[],
                tool_choice="auto",
            )
        )


def test_stream_options_without_streaming_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(
                stream=False,
                stream_options={"include_usage": True},
            )
        )


def test_empty_stop_sequence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(stop="")
        )


def test_stop_sequence_longer_than_128_characters_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(stop="x" * 129)
        )


def test_more_than_four_stop_sequences_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(stop=["A", "B", "C", "D", "E"])
        )


def test_empty_stop_item_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            valid_payload(stop=["END", ""])
        )


def test_unknown_stream_options_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        StreamOptions.model_validate(
            {
                "include_usage": True,
                "unsupported_option": True,
            }
        )