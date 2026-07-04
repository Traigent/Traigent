"""Recursive redaction helpers for public SDK result surfaces."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, overload

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
_CREDIT_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_API_KEY_PATTERN = re.compile(
    r"\b(?:sk|pk|ak|rk|api)[-_][A-Za-z0-9][A-Za-z0-9._-]{10,}\b"
)
_BEARER_TOKEN_PATTERN = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE
)
_COMPACT_TIMESTAMP_PATTERN = re.compile(r"^\d{8}[- ]?\d{6}$")

# Canonical, single source of truth for key-*name*-based redaction.
#
# This is the union of three keyword lists that were previously maintained
# independently and had drifted out of sync (traigent.cloud.dataset_converter,
# traigent.observability.decorators, traigent.observability.agent_spans): a
# key redacted by one path could pass through unredacted on another. All
# three now call `is_sensitive_key_name` below. Extend this set - do not
# fork a local copy - when a new sensitive key pattern is identified.
SENSITIVE_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {
        # Credential / secret fragments
        "api_key",
        "apikey",
        "auth",  # also matches "authorization"
        "credential",
        "credit_card",
        "creditcard",
        "password",
        "private_key",
        "secret",
        "token",
        # Free-form content fragments (may carry prompt/response text)
        "actual",
        "completion",
        "expected",
        "output",
        "prompt",
        "response",
    }
)


def is_sensitive_key_name(key: str) -> bool:
    """Return True when a metadata/config key name looks sensitive.

    "Sensitive" covers both credential-like key names (e.g. `api_key`,
    `auth_token`) and free-form content fields (e.g. `prompt`, `response`)
    that may carry PII or secrets. This is the canonical check backing all
    SDK sanitizers that redact-by-key-name; see `SENSITIVE_KEY_FRAGMENTS`.
    """
    normalized = key.strip().lower().replace("-", "_").replace(".", "_")
    return any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS)


def _passes_luhn(digits: str) -> bool:
    """Return True iff the digit string is a valid Luhn checksum (PAN check)."""
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = ord(ch) - 48
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _redact_credit_card(match: re.Match[str]) -> str:
    """Redact only digit runs that pass Luhn — avoid false-positives on timestamps."""
    raw = match.group(0)
    normalized = raw.strip(" -")
    if _COMPACT_TIMESTAMP_PATTERN.fullmatch(normalized):
        digits = "".join(ch for ch in normalized if ch.isdigit())
        try:
            datetime.strptime(digits, "%Y%m%d%H%M%S")
            return raw
        except ValueError:
            pass
    digits = "".join(ch for ch in raw if ch.isdigit())
    if 13 <= len(digits) <= 19 and _passes_luhn(digits):
        return "[REDACTED:credit_card]"
    return raw


@overload
def redact_sensitive_text(value: str) -> str: ...


@overload
def redact_sensitive_text(value: None) -> None: ...


def redact_sensitive_text(value: str | None) -> str | None:
    """Redact common PII and credential-like secrets from text."""
    if value is None:
        return None
    redacted = value
    redacted = _EMAIL_PATTERN.sub("[REDACTED:email]", redacted)
    redacted = _SSN_PATTERN.sub("[REDACTED:ssn]", redacted)
    redacted = _CREDIT_CARD_CANDIDATE.sub(_redact_credit_card, redacted)
    redacted = _API_KEY_PATTERN.sub("[REDACTED:api_key]", redacted)
    redacted = _BEARER_TOKEN_PATTERN.sub("[REDACTED:bearer_token]", redacted)
    return redacted


def redact_sensitive_data(value: Any) -> Any:
    """Return a recursively redacted copy of JSON-like data."""
    if isinstance(value, str):
        return redact_sensitive_text(value)

    if isinstance(value, Mapping):
        return {key: redact_sensitive_data(item) for key, item in value.items()}

    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)

    if isinstance(value, set):
        return {redact_sensitive_data(item) for item in value}

    return value
