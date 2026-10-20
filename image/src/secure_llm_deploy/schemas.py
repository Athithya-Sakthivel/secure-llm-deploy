"""Strict request schemas for the supported OpenAI-compatible API subset."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictSchema(BaseModel):
    """Reject unknown request fields instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictSchema):
    role: Literal["system", "developer", "user", "assistant"]
    content: str = Field(min_length=1, max_length=60000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content cannot be blank")
        return value


class StreamOptions(StrictSchema):
    include_usage: bool = False


class ChatCompletionRequest(StrictSchema):
    model: str = Field(min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2147483647)

    n: Literal[1] = 1
    stream: bool = False
    stream_options: StreamOptions | None = None

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Model name cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> ChatCompletionRequest:
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("At least one user message is required")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")

        stop_sequences = [self.stop] if isinstance(self.stop, str) else self.stop
        if stop_sequences is not None:
            if len(stop_sequences) > 4:
                raise ValueError("At most four stop sequences are supported")
            if any(not value or len(value) > 128 for value in stop_sequences):
                raise ValueError("Each stop sequence must contain 1-128 characters")
        return self
