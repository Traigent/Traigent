"""Recursive redaction helpers for public SDK result surfaces."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, overload

_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    ("ssn", re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    ),
    (
        "api_key",
        re.compile(r"\b(?:sk|pk|ak|rk|api)[-_][A-Za-z0-9][A-Za-z0-9._-]{10,}\b"),
    ),
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    ),
)


@overload
def redact_sensitive_text(value: str) -> str: ...


@overload
def redact_sensitive_text(value: None) -> None: ...


def redact_sensitive_text(value: str | None) -> str | None:
    """Redact common PII and credential-like secrets from text."""
    if value is None:
        return None
    redacted = value
    for label, pattern in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(f"[REDACTED:{label}]", redacted)
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
