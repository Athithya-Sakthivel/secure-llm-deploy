"""Unit tests for deterministic PII and sensitive-string redaction."""

from __future__ import annotations

import pytest

from secure_llm_deploy.pii import PiiRedactor, luhn_valid


@pytest.fixture
def redactor() -> PiiRedactor:
    return PiiRedactor()


def test_redacts_email_address(
    redactor: PiiRedactor,
) -> None:
    result, count = redactor.redact_text_with_count(
        "Contact alice.smith+test@example.co.uk for help."
    )

    assert result == "Contact [REDACTED_EMAIL] for help."
    assert count == 1


def test_redacts_multiple_distinct_pii_values(
    redactor: PiiRedactor,
) -> None:
    text = (
        "Email alice@example.com, call +1 (415) 555-0136, "
        "and check host 192.168.1.10."
    )

    result, count = redactor.redact_text_with_count(text)

    assert "[REDACTED_EMAIL]" in result
    assert "[REDACTED_PHONE]" in result
    assert "[REDACTED_IP]" in result
    assert "alice@example.com" not in result
    assert "192.168.1.10" not in result
    assert count >= 3


@pytest.mark.parametrize(
    "phone",
    [
        "415-555-0136",
        "(415) 555-0136",
        "+1 415 555 0136",
        "020 7946 0958",
    ],
)
def test_redacts_supported_phone_formats(
    redactor: PiiRedactor,
    phone: str,
) -> None:
    result = redactor.redact_text(f"Call {phone} for support.")

    assert "[REDACTED_PHONE]" in result
    assert phone not in result


@pytest.mark.parametrize(
    "address",
    [
        "192.168.1.10",
        "8.8.8.8",
        "10.0.0.1",
    ],
)
def test_redacts_valid_ipv4_addresses(
    redactor: PiiRedactor,
    address: str,
) -> None:
    result = redactor.redact_text(f"Server address: {address}")

    assert "[REDACTED_IP]" in result
    assert address not in result


@pytest.mark.parametrize(
    "address",
    [
        "2001:db8::1",
        "2001:4860:4860::8888",
        "::1",
    ],
)
def test_redacts_valid_ipv6_addresses(
    redactor: PiiRedactor,
    address: str,
) -> None:
    result = redactor.redact_text(f"Server address: {address}")

    assert "[REDACTED_IP]" in result
    assert address not in result


def test_luhn_accepts_a_standard_test_card_number() -> None:
    assert luhn_valid("4111 1111 1111 1111")


def test_luhn_rejects_wrong_checksum() -> None:
    assert not luhn_valid("4111 1111 1111 1112")


@pytest.mark.parametrize(
    "number",
    [
        "4111111111111111",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
    ],
)
def test_redacts_luhn_valid_card_like_numbers(
    redactor: PiiRedactor,
    number: str,
) -> None:
    result = redactor.redact_text(f"Payment card: {number}")

    assert "[REDACTED_CARD]" in result
    assert number not in result


def test_invalid_card_checksum_is_not_classified_as_a_card(
    redactor: PiiRedactor,
) -> None:
    number = "4111 1111 1111 1112"

    spans = redactor.find_spans(f"Reference {number}")

    assert all(span.replacement != "[REDACTED_CARD]" for span in spans)


def test_short_numeric_references_are_not_redacted(
    redactor: PiiRedactor,
) -> None:
    text = "Order 12345, HTTP port 8080, rack 42."

    result, count = redactor.redact_text_with_count(text)

    assert result == text
    assert count == 0


@pytest.mark.parametrize(
    "invalid_address",
    [
        "999.999.999.999",
        "256.10.10.10",
        "192.168.1",
    ],
)
def test_invalid_ipv4_is_not_classified_as_an_ip(
    redactor: PiiRedactor,
    invalid_address: str,
) -> None:
    spans = redactor.find_spans(f"Address {invalid_address}")

    assert all(span.replacement != "[REDACTED_IP]" for span in spans)


def test_configured_sensitive_string_is_redacted() -> None:
    secret = "SYNTHETIC_SECRET_ALPHA_001"
    redactor = PiiRedactor((secret,))

    result, count = redactor.redact_text_with_count(
        f"Configured value: {secret}."
    )

    assert result == "Configured value: [REDACTED_SECRET]."
    assert count == 1
    assert secret not in result


def test_repeated_sensitive_string_occurrences_are_redacted() -> None:
    secret = "SYNTHETIC_SECRET_BETA_002"
    redactor = PiiRedactor((secret,))

    result, count = redactor.redact_text_with_count(
        f"{secret} and then {secret}"
    )

    assert result == "[REDACTED_SECRET] and then [REDACTED_SECRET]"
    assert count == 2


def test_configured_secret_takes_precedence_over_email_match() -> None:
    secret = "alice@example.com"
    redactor = PiiRedactor((secret,))

    result = redactor.redact_text(
        f"Sensitive value: {secret}."
    )

    assert result == "Sensitive value: [REDACTED_SECRET]."
    assert "[REDACTED_EMAIL]" not in result


def test_overlapping_spans_are_resolved_without_duplicate_replacements() -> None:
    secret = "alice@example.com"
    redactor = PiiRedactor((secret,))

    spans = redactor.find_spans(f"{secret}")

    assert len(spans) == 1
    assert spans[0].replacement == "[REDACTED_SECRET]"


def test_unrelated_text_is_preserved(
    redactor: PiiRedactor,
) -> None:
    text = "A prompt-injection test with no contact details."

    assert redactor.redact_text(text) == text


def test_empty_text_is_unchanged(
    redactor: PiiRedactor,
) -> None:
    result, count = redactor.redact_text_with_count("")

    assert result == ""
    assert count == 0


def test_redaction_is_idempotent(
    redactor: PiiRedactor,
) -> None:
    text = "Contact alice@example.com"
    once = redactor.redact_text(text)
    twice = redactor.redact_text(once)

    assert twice == once


def test_long_input_is_processed_without_losing_the_sensitive_value(
    redactor: PiiRedactor,
) -> None:
    text = ("ordinary text " * 5000) + "alice@example.com" + (
        " trailing text" * 5000
    )

    result = redactor.redact_text(text)

    assert "[REDACTED_EMAIL]" in result
    assert "alice@example.com" not in result
    assert result.startswith("ordinary text ")
    assert result.endswith(" trailing text")


def test_embedded_ipv4_like_text_should_not_be_redacted_as_a_standalone_ip(
    redactor: PiiRedactor,
) -> None:
    # Regression test: IPv4 matches should respect identifier boundaries.
    text = "host=x192.168.1.10y"

    assert redactor.redact_text(text) == text