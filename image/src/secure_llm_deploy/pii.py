"""Local regex-based PII and configured-sensitive-string redaction."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

EMAIL_RE = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?){1,4}"
)

CARD_RE = re.compile(r"(?:\d[ -]?){12,18}\d")

IPV4_RE = re.compile(
    r"(?<![\w.])"
    r"(?:\d{1,3}\.){3}\d{1,3}"
    r"(?!\w|\.\d)"
)

IPV6_RE = re.compile(r"(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}")

PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s().-]?)?"
    r"(?:\(?\d{2,4}\)?[\s.-]?)"
    r"\d{3,4}[\s.-]?\d{3,4}"
)


@dataclass(frozen=True)
class RedactionSpan:
    start: int
    end: int
    replacement: str
    priority: int


def luhn_valid(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False

    total = 0
    parity = len(digits) % 2

    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit

    return total % 10 == 0


class PiiRedactor:
    """Redact supported patterns. No network calls or external service dependency."""

    def __init__(self, sensitive_values: tuple[str, ...] = ()) -> None:
        self._secrets = tuple(
            sorted(set(sensitive_values), key=len, reverse=True)
        )
        self._secret_patterns = tuple(
            re.compile(re.escape(secret))
            for secret in self._secrets
        )

        # EMAIL_RE has a maximum width below 512 characters. Configuration also
        # bounds secret lengths, making this a finite streaming holdback width.
        self.max_match_width = max(
            512,
            max((len(secret) for secret in self._secrets), default=0),
        )

    @staticmethod
    def _valid_ip(candidate: str) -> bool:
        try:
            ipaddress.ip_address(candidate)
            return True
        except ValueError:
            return False

    def find_spans(self, text: str) -> list[RedactionSpan]:
        matches: list[RedactionSpan] = []

        # Configured sensitive values
        matches.extend(
            RedactionSpan(
                match.start(),
                match.end(),
                "[REDACTED_SECRET]",
                0,
            )
            for pattern in self._secret_patterns
            for match in pattern.finditer(text)
        )

        # IPv4 addresses: reject matches embedded in identifiers or
        # partial matches at the start of longer dotted numeric sequences.
        matches.extend(
            RedactionSpan(
                match.start(),
                match.end(),
                "[REDACTED_IP]",
                3,
            )
            for match in IPV4_RE.finditer(text)
            if self._valid_ip(match.group())
        )

        # Credit card numbers
        matches.extend(
            RedactionSpan(
                match.start(),
                match.end(),
                "[REDACTED_CARD]",
                1,
            )
            for match in CARD_RE.finditer(text)
            if luhn_valid(match.group())
        )

        # Email addresses
        matches.extend(
            RedactionSpan(
                match.start(),
                match.end(),
                "[REDACTED_EMAIL]",
                2,
            )
            for match in EMAIL_RE.finditer(text)
        )

        # IPv6 addresses
        matches.extend(
            RedactionSpan(
                match.start(),
                match.end(),
                "[REDACTED_IP]",
                3,
            )
            for match in IPV6_RE.finditer(text)
            if ":" in match.group() and self._valid_ip(match.group())
        )

        # Phone numbers
        matches.extend(
            RedactionSpan(
                match.start(),
                match.end(),
                "[REDACTED_PHONE]",
                4,
            )
            for match in PHONE_RE.finditer(text)
            if 7 <= len(re.sub(r"\D", "", match.group())) <= 15
        )

        # Prefer the earliest match, then the lowest priority number,
        # then the longest match when the start and priority are equal.
        matches.sort(
            key=lambda item: (
                item.start,
                item.priority,
                -(item.end - item.start),
            )
        )

        selected: list[RedactionSpan] = []
        cursor = 0

        for match in matches:
            if match.start < cursor or match.end <= match.start:
                continue
            selected.append(match)
            cursor = match.end

        return selected

    def redact_text_with_count(self, text: str) -> tuple[str, int]:
        spans = self.find_spans(text)
        if not spans:
            return text, 0

        result: list[str] = []
        cursor = 0

        for span in spans:
            result.append(text[cursor:span.start])
            result.append(span.replacement)
            cursor = span.end

        result.append(text[cursor:])
        return "".join(result), len(spans)

    def redact_text(self, text: str) -> str:
        redacted, _ = self.redact_text_with_count(text)
        return redacted