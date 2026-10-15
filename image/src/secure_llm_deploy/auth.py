"""Bearer-token authentication helpers."""

from __future__ import annotations

import hmac

from pydantic import SecretStr


def authenticate_bearer_token(
    authorization_header: str | None,
    configured_secret: SecretStr | None,
) -> bool:
    """Return whether a single syntactically valid bearer token matches."""
    if configured_secret is None or authorization_header is None:
        return False

    scheme, separator, candidate = authorization_header.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return False

    candidate = candidate.strip()
    if not candidate or " " in candidate or "\t" in candidate:
        return False

    expected = configured_secret.get_secret_value()
    return hmac.compare_digest(
        candidate.encode("utf-8"),
        expected.encode("utf-8"),
    )