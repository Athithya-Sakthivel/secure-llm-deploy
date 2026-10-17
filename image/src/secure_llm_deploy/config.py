"""Validated environment configuration for Secure LLM Deploy."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with explicit defaults and fail-closed validation."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
        validate_default=True,
        populate_by_name=True,
    )

    # Public FastAPI gateway.
    app_name: str = "secure-llm-deploy"
    gateway_host: str = "0.0.0.0"
    gateway_port: int = Field(default=8000, ge=1, le=65535)

    # Client-facing and internal API credentials are deliberately distinct.
    gateway_api_key: SecretStr | None = Field(default=None, repr=False)
    vllm_api_key: SecretStr | None = Field(default=None, repr=False)

    # Serving configuration.
    serve_mode: Literal["baseline", "speculative"] = "baseline"
    served_model_name: str = "secure-llm-deploy"

    target_model_path: Path = Path("/opt/models/target")
    target_tokenizer_path: Path = Path("/opt/models/target/tokenizer.json")

    # EAGLE-3 is the only enabled speculative method. An EAGLE checkpoint is
    # an auxiliary speculator, not a standalone draft language model, and it
    # does not need its own tokenizer.json.
    eagle_model_path: Path = Path("/opt/models/eagle")

    # Local prompt-injection guard. ONNX inference uses CPUExecutionProvider.
    guard_model_path: Path = Path("/opt/models/guard/onnx/model.onnx")
    guard_tokenizer_path: Path = Path("/opt/models/guard/tokenizer.json")
    guard_label_mapping_path: Path = Path("/opt/models/guard/label_mapping.json")
    guard_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    guard_max_input_tokens: int = Field(default=8192, ge=128, le=32768)
    guard_intra_op_threads: int = Field(default=4, ge=1, le=64)
    max_concurrent_guard_evaluations: int = Field(default=2, ge=1, le=32)

    # HTTP request limits.
    request_max_body_bytes: int = Field(default=65536, ge=1024, le=1048576)
    max_messages: int = Field(default=32, ge=1, le=32)
    max_message_chars: int = Field(default=30000, ge=1, le=60000)
    max_input_chars: int = Field(default=30000, ge=1, le=60000)

    # Conservative initial profile for a 16 GB NVIDIA T4. These are starting
    # limits, not a guarantee that every vLLM build will fit in GPU memory.
    max_input_tokens: int = Field(default=3072, ge=128)
    default_max_tokens: int = Field(default=512, ge=1)
    max_output_tokens: int = Field(default=1024, ge=1)
    max_model_len: int = Field(
        default=4096,
        ge=256,
        validation_alias=AliasChoices("VLLM_MAX_MODEL_LEN", "MAX_MODEL_LEN"),
    )

    # Per-client rate limiting and inference concurrency.
    rate_limit_requests: int = Field(default=5, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    rate_limit_max_clients: int = Field(default=10000, ge=100)
    queue_wait_timeout_seconds: float = Field(default=5.0, gt=0)
    max_concurrent_inference_requests: int = Field(default=2, ge=1, le=32)

    # Internal vLLM process.
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

    # EAGLE-3 speculative decoding. Do not expose draft_model as an option.
    speculative_tokens: int = Field(
        default=2,
        ge=1,
        le=8,
        validation_alias=AliasChoices(
            "VLLM_NUM_SPECULATIVE_TOKENS", "SPECULATIVE_TOKENS"
        ),
    )

    # Newline-separated exact values to redact. Never configure real credentials.
    sensitive_strings: str = Field(default="", repr=False)
    max_sensitive_string_chars: int = Field(default=1024, ge=8, le=4096)

    @property
    def gateway_key(self) -> str:
        """Return the configured client-facing key."""
        if self.gateway_api_key is None:
            raise ValueError("GATEWAY_API_KEY is required")
        return self.gateway_api_key.get_secret_value()

    @property
    def upstream_key(self) -> str:
        """Return the configured internal vLLM key."""
        if self.vllm_api_key is None:
            raise ValueError("VLLM_API_KEY is required")
        return self.vllm_api_key.get_secret_value()

    @property
    def sensitive_values(self) -> tuple[str, ...]:
        """Return nonempty configured sensitive strings."""
        return tuple(
            line for line in self.sensitive_strings.splitlines() if line.strip()
        )

    def validate_runtime(self) -> None:
        """Validate credentials, budgets, and the artifacts needed at runtime.

        Full repository pinning and weight-manifest validation are build-time
        responsibilities. This method checks the application/runtime contract.
        """
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
        if self.vllm_port == self.gateway_port:
            raise ValueError("Gateway and vLLM ports must be different")
        if len(self.sensitive_values) > 100:
            raise ValueError("At most 100 sensitive strings may be configured")
        for value in self.sensitive_values:
            if not 8 <= len(value) <= self.max_sensitive_string_chars:
                raise ValueError(
                    "Configured sensitive strings must be between 8 and "
                    "MAX_SENSITIVE_STRING_CHARS characters"
                )

        # These assets are needed even when speculative decoding is disabled.
        required_files = (
            self.target_model_path / "config.json",
            self.target_tokenizer_path,
            self.guard_model_path,
            self.guard_tokenizer_path,
            self.guard_label_mapping_path,
        )
        for path in required_files:
            if not path.is_file():
                raise ValueError(f"Required model artifact is missing: {path}")

        if not self.target_model_path.is_dir():
            raise ValueError("TARGET_MODEL_PATH must be a directory")

        # Vela's ONNX graph uses an external tensor-data sidecar.
        if self.guard_model_path.name == "model.onnx":
            external_data = self.guard_model_path.with_name("model.onnx.data")
            if not external_data.is_file():
                raise ValueError(f"Vela ONNX external data file is missing: {external_data}")

        # EAGLE is optional in baseline mode, but must be a complete EAGLE-3
        # checkpoint when speculative decoding is enabled. No draft tokenizer.
        if self.serve_mode == "speculative":
            required_eagle_files = (
                self.eagle_model_path / "config.json",
                self.eagle_model_path / "model.safetensors",
            )
            for path in required_eagle_files:
                if not path.is_file():
                    raise ValueError(f"Required EAGLE-3 artifact is missing: {path}")

        if not self.vllm_executable.strip():
            raise ValueError("VLLM_EXECUTABLE cannot be empty")
