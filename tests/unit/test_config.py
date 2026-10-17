
"""Unit tests for environment settings and runtime configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from secure_llm_deploy.config import Settings


def test_default_settings_use_baseline_mode() -> None:
    settings = Settings(_env_file=None)

    assert settings.serve_mode == "baseline"
    assert settings.served_model_name == "secure-llm-deploy"
    assert settings.gateway_host == "0.0.0.0"
    assert settings.gateway_port == 8000

    assert settings.target_model_path == Path("/opt/models/target")
    assert settings.target_tokenizer_path == Path(
        "/opt/models/target/tokenizer.json"
    )
    assert settings.eagle_model_path == Path("/opt/models/eagle")
    assert settings.guard_model_path == Path(
        "/opt/models/guard/onnx/model.onnx"
    )
    assert settings.guard_tokenizer_path == Path(
        "/opt/models/guard/tokenizer.json"
    )
    assert settings.guard_label_mapping_path == Path(
        "/opt/models/guard/label_mapping.json"
    )

    assert settings.vllm_host == "127.0.0.1"
    assert settings.vllm_port == 8001

    assert settings.max_input_tokens == 3072
    assert settings.default_max_tokens == 512
    assert settings.max_output_tokens == 1024
    assert settings.max_model_len == 4096

    assert settings.vllm_max_num_seqs == 2
    assert settings.gpu_memory_utilization == pytest.approx(0.80)
    assert settings.speculative_tokens == 2

    assert settings.rate_limit_requests == 5
    assert settings.rate_limit_window_seconds == 60

    assert settings.gateway_api_key is None
    assert settings.vllm_api_key is None


@pytest.mark.parametrize(
    ("environment_name", "field_name", "raw_value", "expected"),
    [
        ("VLLM_MAX_MODEL_LEN", "max_model_len", "6144", 6144),
        ("MAX_MODEL_LEN", "max_model_len", "6144", 6144),
        (
            "VLLM_GPU_MEMORY_UTILIZATION",
            "gpu_memory_utilization",
            "0.75",
            0.75,
        ),
        ("GPU_MEMORY_UTILIZATION", "gpu_memory_utilization", "0.75", 0.75),
        ("VLLM_NUM_SPECULATIVE_TOKENS", "speculative_tokens", "6", 6),
        ("SPECULATIVE_TOKENS", "speculative_tokens", "6", 6),
    ],
)
def test_runtime_environment_aliases(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    field_name: str,
    raw_value: str,
    expected: int | float,
) -> None:
    monkeypatch.setenv(environment_name, raw_value)
    settings = Settings(_env_file=None)

    assert getattr(settings, field_name) == pytest.approx(expected)


@pytest.mark.parametrize(
    "field_name",
    ["gateway_api_key", "vllm_api_key"],
)
def test_validate_runtime_requires_both_api_keys(
    valid_settings: Settings,
    field_name: str,
) -> None:
    settings = valid_settings.model_copy(update={field_name: None})

    with pytest.raises(
        ValueError,
        match="GATEWAY_API_KEY and VLLM_API_KEY must both be configured",
    ):
        settings.validate_runtime()


def test_validate_runtime_rejects_identical_api_keys(
    valid_settings: Settings,
) -> None:
    settings = valid_settings.model_copy(
        update={"vllm_api_key": valid_settings.gateway_api_key}
    )

    with pytest.raises(
        ValueError,
        match="Gateway and upstream API keys must be different",
    ):
        settings.validate_runtime()


@pytest.mark.parametrize(
    "field_name",
    ["gateway_api_key", "vllm_api_key"],
)
def test_validate_runtime_rejects_short_api_keys(
    valid_settings: Settings,
    field_name: str,
) -> None:
    settings = valid_settings.model_copy(
        update={field_name: SecretStr("too-short")}
    )

    with pytest.raises(
        ValueError,
        match="Both API keys must contain at least 32 characters",
    ):
        settings.validate_runtime()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("gateway_port", 0),
        ("gateway_port", 65536),
        ("vllm_port", 0),
        ("vllm_port", 65536),
        ("rate_limit_requests", 0),
        ("rate_limit_window_seconds", 0),
        ("request_max_body_bytes", 0),
        ("guard_threshold", -0.01),
        ("guard_threshold", 1.01),
        ("max_messages", 0),
        ("max_messages", 33),
        ("max_output_tokens", 0),
        ("max_model_len", 255),
        ("gpu_memory_utilization", 1.0),
        ("speculative_tokens", 0),
        ("speculative_tokens", 9),
        ("vllm_max_num_seqs", 0),
        ("serve_mode", "unknown"),
        ("vllm_host", "0.0.0.0"),
    ],
)
def test_pydantic_rejects_invalid_field_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field_name: value})


def test_validate_runtime_rejects_context_budget_overflow(
    valid_settings: Settings,
) -> None:
    settings = valid_settings.model_copy(
        update={"max_output_tokens": 2048}
    )

    with pytest.raises(
        ValueError,
        match=r"MAX_INPUT_TOKENS \+ MAX_OUTPUT_TOKENS exceeds MAX_MODEL_LEN",
    ):
        settings.validate_runtime()


def test_validate_runtime_rejects_default_output_above_maximum(
    valid_settings: Settings,
) -> None:
    settings = valid_settings.model_copy(
        update={"max_output_tokens": 256}
    )

    with pytest.raises(
        ValueError,
        match="DEFAULT_MAX_TOKENS cannot exceed MAX_OUTPUT_TOKENS",
    ):
        settings.validate_runtime()


def test_validate_runtime_rejects_input_budget_equal_to_guard_budget(
    valid_settings: Settings,
) -> None:
    settings = valid_settings.model_copy(
        update={
            "max_model_len": 16384,
            "max_input_tokens": 8192,
            "guard_max_input_tokens": 8192,
        }
    )

    with pytest.raises(
        ValueError,
        match="MAX_INPUT_TOKENS must be lower than GUARD_MAX_INPUT_TOKENS",
    ):
        settings.validate_runtime()


def test_validate_runtime_rejects_equal_gateway_and_vllm_ports(
    valid_settings: Settings,
) -> None:
    settings = valid_settings.model_copy(
        update={"vllm_port": valid_settings.gateway_port}
    )

    with pytest.raises(
        ValueError,
        match="Gateway and vLLM ports must be different",
    ):
        settings.validate_runtime()


def test_validate_runtime_rejects_blank_vllm_executable(
    valid_settings: Settings,
) -> None:
    settings = valid_settings.model_copy(update={"vllm_executable": " "})

    with pytest.raises(
        ValueError,
        match="VLLM_EXECUTABLE cannot be empty",
    ):
        settings.validate_runtime()


def test_validate_runtime_does_not_require_real_model_weights(
    valid_settings: Settings,
) -> None:
    """Baseline configuration tests do not load real model weights."""
    valid_settings.validate_runtime()


def test_validate_runtime_rejects_missing_target_tokenizer(
    valid_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = valid_settings.model_copy(
        update={
            "target_tokenizer_path": tmp_path / "missing-tokenizer.json"
        }
    )

    with pytest.raises(
        ValueError,
        match="Required model artifact is missing",
    ):
        settings.validate_runtime()


def test_validate_runtime_rejects_missing_target_config(
    valid_settings: Settings,
    model_artifacts: dict[str, Path],
) -> None:
    model_artifacts["target_config"].unlink()

    with pytest.raises(
        ValueError,
        match=r"Required model artifact is missing: .*config\.json",
    ):
        valid_settings.validate_runtime()


def test_validate_runtime_rejects_missing_guard_artifact(
    valid_settings: Settings,
    model_artifacts: dict[str, Path],
) -> None:
    model_artifacts["guard_model"].unlink()

    with pytest.raises(
        ValueError,
        match=r"Required model artifact is missing: .*model\.onnx",
    ):
        valid_settings.validate_runtime()


def test_validate_runtime_rejects_missing_onnx_external_data(
    valid_settings: Settings,
    model_artifacts: dict[str, Path],
) -> None:
    model_artifacts["guard_data"].unlink()

    with pytest.raises(
        ValueError,
        match="Vela ONNX external data file is missing",
    ):
        valid_settings.validate_runtime()


def test_validate_runtime_rejects_missing_eagle_config_in_speculative_mode(
    valid_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = valid_settings.model_copy(
        update={
            "serve_mode": "speculative",
            "eagle_model_path": tmp_path / "missing-eagle-model",
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Required EAGLE-3 artifact is missing: .*config\.json",
    ):
        settings.validate_runtime()


def test_validate_runtime_rejects_missing_eagle_weights_in_speculative_mode(
    valid_settings: Settings,
    model_artifacts: dict[str, Path],
) -> None:
    model_artifacts["eagle_weights"].unlink()
    settings = valid_settings.model_copy(
        update={"serve_mode": "speculative"}
    )

    with pytest.raises(
        ValueError,
        match=r"Required EAGLE-3 artifact is missing: .*model\.safetensors",
    ):
        settings.validate_runtime()


def test_speculative_mode_does_not_require_eagle_tokenizer(
    valid_settings: Settings,
) -> None:
    """EAGLE-3 is an auxiliary speculator, not a standalone draft LM."""
    settings = valid_settings.model_copy(
        update={"serve_mode": "speculative"}
    )

    settings.validate_runtime()
    assert not (settings.eagle_model_path / "tokenizer.json").exists()


def test_baseline_mode_does_not_require_eagle_artifacts(
    valid_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = valid_settings.model_copy(
        update={
            "serve_mode": "baseline",
            "eagle_model_path": tmp_path / "eagle-not-installed",
        }
    )

    settings.validate_runtime()


def test_sensitive_values_ignore_blank_lines(
    valid_settings: Settings,
) -> None:
    settings = valid_settings.model_copy(
        update={
            "sensitive_strings": (
                "\n"
                "SYNTHETIC_SECRET_ALPHA_001\n"
                "\n"
                "SYNTHETIC_SECRET_BETA_002\n"
            )
        }
    )

    assert settings.sensitive_values == (
        "SYNTHETIC_SECRET_ALPHA_001",
        "SYNTHETIC_SECRET_BETA_002",
    )


@pytest.mark.parametrize(
    "sensitive_strings",
    [
        "tiny",
        "x" * 1025,
    ],
)
def test_validate_runtime_rejects_invalid_sensitive_string_lengths(
    valid_settings: Settings,
    sensitive_strings: str,
) -> None:
    settings = valid_settings.model_copy(
        update={"sensitive_strings": sensitive_strings}
    )

    with pytest.raises(
        ValueError,
        match="Configured sensitive strings must be between",
    ):
        settings.validate_runtime()


def test_validate_runtime_rejects_too_many_sensitive_strings(
    valid_settings: Settings,
) -> None:
    values = [
        f"SYNTHETIC_SECRET_{index:03d}"
        for index in range(101)
    ]
    settings = valid_settings.model_copy(
        update={"sensitive_strings": "\n".join(values)}
    )

    with pytest.raises(
        ValueError,
        match="At most 100 sensitive strings may be configured",
    ):
        settings.validate_runtime()
