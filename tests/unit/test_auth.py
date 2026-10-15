"""Unit tests for Bearer-token authentication."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from secure_llm_deploy.auth import authenticate_bearer_token

TEST_KEY = "synthetic-gateway-key-" + ("k" * 40)


@pytest.fixture
def configured_secret() -> SecretStr:
    return SecretStr(TEST_KEY)


def test_valid_bearer_token_is_accepted(
    configured_secret: SecretStr,
) -> None:
    assert authenticate_bearer_token(
        f"Bearer {TEST_KEY}",
        configured_secret,
    )


@pytest.mark.parametrize("scheme", ["bearer", "BEARER", "Bearer"])
def test_bearer_scheme_is_case_insensitive(
    configured_secret: SecretStr,
    scheme: str,
) -> None:
    assert authenticate_bearer_token(
        f"{scheme} {TEST_KEY}",
        configured_secret,
    )


def test_wrong_token_is_rejected(
    configured_secret: SecretStr,
) -> None:
    assert not authenticate_bearer_token(
        "Bearer " + ("x" * len(TEST_KEY)),
        configured_secret,
    )


def test_token_with_extra_fields_is_rejected(
    configured_secret: SecretStr,
) -> None:
    assert not authenticate_bearer_token(
        f"Bearer {TEST_KEY} extra",
        configured_secret,
    )


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "",
        "Bearer",
        "Bearer ",
        "Basic some-token",
        "Token some-token",
        "\tBearer some-token",
        "Bearer\tsome-token",
        "Bearer some-token extra",
    ],
)
def test_malformed_or_missing_authorization_is_rejected(
    configured_secret: SecretStr,
    authorization: str | None,
) -> None:
    assert not authenticate_bearer_token(
        authorization,
        configured_secret,
    )


def test_missing_configured_secret_is_rejected() -> None:
    assert not authenticate_bearer_token(
        f"Bearer {TEST_KEY}",
        None,
    )


def test_empty_configured_secret_does_not_authenticate() -> None:
    # The helper compares the values it receives. Runtime configuration
    # independently rejects empty/short secrets before the app serves traffic.
    assert not authenticate_bearer_token(
        "Bearer some-nonempty-token",
        SecretStr(""),
    )


def test_token_comparison_is_case_sensitive(
    configured_secret: SecretStr,
) -> None:
    changed_key = TEST_KEY[:-1] + TEST_KEY[-1].upper()
    if changed_key == TEST_KEY:
        changed_key = TEST_KEY[:-1] + "z"

    assert not authenticate_bearer_token(
        f"Bearer {changed_key}",
        configured_secret,
    )


def test_whitespace_inside_token_is_rejected(
    configured_secret: SecretStr,
) -> None:
    malformed = TEST_KEY[:10] + " " + TEST_KEY[10:]

    assert not authenticate_bearer_token(
        f"Bearer {malformed}",
        configured_secret,
    )