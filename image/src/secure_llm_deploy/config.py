"""Validated environment configuration for Secure LLM Deploy."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings; expensive model-contract checks run at image build."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
        validate_default=True,
        populate_by_name=True,
    )

    app_name: str = "secure-llm-deploy"
    gateway_host: str = "0.0.0.0"
    gateway_port: int = Field(default=8000, ge=1, le=65535)
    gateway_api_key: SecretStr | None = Field(default=None, repr=False)
    vllm_api_key: SecretStr | None = Field(default=None, repr=False)

    serve_mode: Literal["baseline", "speculative"] = "baseline"
    served_model_name: str = "secure-llm-deploy"
    target_model_path: Path = Path("/opt/models/target")
    target_tokenizer_path: Path = Path("/opt/models/target/tokenizer.json")
    eagle_model_path: Path = Path("/opt/models/eagle")

    guard_model_path: Path = Path("/opt/models/guard/model.QUInt8.quant.onnx")
    guard_tokenizer_path: Path = Path("/opt/models/guard/tokenizer.json")
    guard_config_path: Path = Path("/opt/models/guard/config.json")
    guard_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    guard_max_input_tokens: int = Field(default=6144, ge=128, le=32768)
    guard_window_tokens: int = Field(default=512, ge=16, le=512)
    guard_intra_op_threads: int = Field(default=2, ge=1, le=64)
    guard_failure_policy: Literal["closed", "open"] = "closed"
    guard_overlong_policy: Literal["reject", "truncate"] = "reject"
    max_concurrent_guard_evaluations: int = Field(default=2, ge=1, le=32)

    request_max_body_bytes: int = Field(default=65536, ge=1024, le=1048576)
    max_messages: int = Field(default=32, ge=1, le=32)
    max_message_chars: int = Field(default=30000, ge=1, le=60000)
    max_input_chars: int = Field(default=30000, ge=1, le=60000)

    max_input_tokens: int = Field(default=3072, ge=128)
    default_max_tokens: int = Field(default=512, ge=1)
    max_output_tokens: int = Field(default=1024, ge=1)
    max_model_len: int = Field(
        default=4096,
        ge=256,
        validation_alias=AliasChoices("VLLM_MAX_MODEL_LEN", "MAX_MODEL_LEN"),
    )

    rate_limit_requests: int = Field(default=5, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    rate_limit_max_clients: int = Field(default=10000, ge=100)
    queue_wait_timeout_seconds: float = Field(default=5.0, gt=0)
    max_concurrent_inference_requests: int = Field(default=2, ge=1, le=32)

    vllm_executable: str = "vllm"
    vllm_host: Literal["127.0.0.1"] = "127.0.0.1"
    vllm_port: int = Field(default=8001, ge=1, le=65535)
    vllm_startup_timeout_seconds: int = Field(default=600, ge=10)
    vllm_startup_poll_seconds: float = Field(default=2.0, gt=0)
    vllm_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    vllm_read_timeout_seconds: float = Field(default=120.0, gt=0)
    vllm_write_timeout_seconds: float = Field(default=30.0, gt=0)
    vllm_pool_timeout_seconds: float = Field(default=5.0, gt=0)
    vllm_max_num_seqs: int = Field(default=2, ge=1)
    gpu_memory_utilization: float = Field(
        default=0.80,
        gt=0.1,
        le=0.99,
        validation_alias=AliasChoices(
            "VLLM_GPU_MEMORY_UTILIZATION", "GPU_MEMORY_UTILIZATION"
        ),
    )
    speculative_tokens: int = Field(
        default=2,
        ge=1,
        le=8,
        validation_alias=AliasChoices(
            "VLLM_NUM_SPECULATIVE_TOKENS", "SPECULATIVE_TOKENS"
        ),
    )

    sensitive_strings: str = Field(default="", repr=False)
    max_sensitive_string_chars: int = Field(default=1024, ge=8, le=4096)

    @model_validator(mode="after")
    def validate_combinations(self) -> Settings:
        """Catch invalid cross-field values at construction time."""
        self._validate_combinations()
        return self

    def _validate_combinations(self) -> None:
        """Shared by Pydantic validation and validate_runtime()."""
        if self.max_input_tokens + self.max_output_tokens > self.max_model_len:
            raise ValueError(
                "MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS exceeds MAX_MODEL_LEN"
            )
        if self.default_max_tokens > self.max_output_tokens:
            raise ValueError("DEFAULT_MAX_TOKENS cannot exceed MAX_OUTPUT_TOKENS")
        if self.max_input_tokens >= self.guard_max_input_tokens:
            raise ValueError(
                "MAX_INPUT_TOKENS must be lower than GUARD_MAX_INPUT_TOKENS"
            )
        if self.guard_window_tokens > self.guard_max_input_tokens:
            raise ValueError(
                "GUARD_WINDOW_TOKENS cannot exceed GUARD_MAX_INPUT_TOKENS"
            )
        if self.vllm_port == self.gateway_port:
            raise ValueError("Gateway and vLLM ports must be different")
        if self.max_message_chars > self.max_input_chars:
            raise ValueError("MAX_MESSAGE_CHARS cannot exceed MAX_INPUT_CHARS")
        if not self.vllm_executable.strip():
            raise ValueError("VLLM_EXECUTABLE cannot be empty")
        if not self.served_model_name.strip():
            raise ValueError("SERVED_MODEL_NAME cannot be empty")

    @property
    def gateway_key(self) -> str:
        """Return the client-facing API key, or fail if not configured."""
        if self.gateway_api_key is None:
            raise ValueError("GATEWAY_API_KEY is required")
        return self.gateway_api_key.get_secret_value()

    @property
    def upstream_key(self) -> str:
        """Return the internal vLLM API key, or fail if not configured."""
        if self.vllm_api_key is None:
            raise ValueError("VLLM_API_KEY is required")
        return self.vllm_api_key.get_secret_value()

    @property
    def sensitive_values(self) -> tuple[str, ...]:
        """Return configured nonempty strings without rewriting their content."""
        return tuple(line for line in self.sensitive_strings.splitlines() if line.strip())

    def validate_runtime(self) -> None:
        """Validate credentials, budgets and minimum required local artifacts.

        Full weight-index, EAGLE-schema and ONNX-graph validation belongs to the
        image-build validator and ONNXPromptGuard. Keeping those expensive checks
        in one place avoids duplicating the build-time validator here.
        """
        # model_copy(update=...) does not validate updated values, so repeat this
        # inexpensive check here before tests or runtime startup depend on it.
        self._validate_combinations()

        if self.gateway_api_key is None or self.vllm_api_key is None:
            raise ValueError(
                "GATEWAY_API_KEY and VLLM_API_KEY must both be configured"
            )
        gateway_key = self.gateway_key
        upstream_key = self.upstream_key
        if len(gateway_key) < 32 or len(upstream_key) < 32:
            raise ValueError("Both API keys must contain at least 32 characters")
        if hmac.compare_digest(gateway_key, upstream_key):
            raise ValueError("Gateway and upstream API keys must be different")

        values = self.sensitive_values
        if len(values) > 100:
            raise ValueError("At most 100 sensitive strings may be configured")
        for value in values:
            if not 8 <= len(value) <= self.max_sensitive_string_chars:
                raise ValueError(
                    "Configured sensitive strings must be between 8 and "
                    "MAX_SENSITIVE_STRING_CHARS characters"
                )

        required_files = (
            self.target_model_path / "config.json",
            self.target_tokenizer_path,
            self.guard_model_path,
            self.guard_tokenizer_path,
            self.guard_config_path,
        )
        for path in required_files:
            if not path.is_file():
                raise ValueError(f"Required model artifact is missing: {path}")
        if not self.target_model_path.is_dir():
            raise ValueError("TARGET_MODEL_PATH must be a directory")

        if self.serve_mode == "speculative":
            # Keep order aligned with useful validation errors: config then weights.
            eagle_files = (
                self.eagle_model_path / "config.json",
                self.eagle_model_path / "model.safetensors",
                self.eagle_model_path / "config.py",
            )
            for path in eagle_files:
                if not path.is_file():
                    raise ValueError(f"Required EAGLE-3 artifact is missing: {path}")
