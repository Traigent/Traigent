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
_CREDENTIAL_KEY_REDACTION = "[REDACTED]"

# Canonical, single source of truth for key-*name*-based redaction.
#
# Three sanitizers previously maintained independent keyword lists that had
# drifted out of sync (traigent.cloud.dataset_converter,
# traigent.observability.decorators, traigent.observability.agent_spans): a
# key redacted by one path could pass through unredacted on another. All
# three now consume the sets below. Extend these sets - do not fork a local
# copy - when a new sensitive key pattern is identified.
#
# Two DISTINCT tiers, deliberately not merged into one flat set:
#
# - CREDENTIAL_KEY_FRAGMENTS: key names that denote secrets/credentials
#   (api_key, auth_token, ...). Safe to apply on EVERY sanitizer path — a
#   value stored under such a key is never legitimate telemetry.
# - CONTENT_KEY_FRAGMENTS: key names that denote free-form content fields
#   (prompt, response, output, ...). Only for call sites that must never
#   carry content-shaped fields at all (e.g. agent_spans, which additionally
#   restricts values to numerics). They must NOT be applied to
#   tuned-configuration surfaces: config spaces routinely tune a variable
#   literally named "prompt" (a variant label, not content), and redacting
#   it would blank legitimate portal/trace display of the chosen config.
CREDENTIAL_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {
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
    }
)

CONTENT_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {
        "actual",
        "completion",
        "expected",
        "output",
        "prompt",
        "response",
    }
)


def _normalize_key_name(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(".", "_")


def is_credential_key_name(key: str) -> bool:
    """Return True when a key name looks credential/secret-like.

    Canonical check backing ALL SDK sanitizers that redact-by-key-name;
    see `CREDENTIAL_KEY_FRAGMENTS`.
    """
    normalized = _normalize_key_name(key)
    return any(fragment in normalized for fragment in CREDENTIAL_KEY_FRAGMENTS)


def is_content_key_name(key: str) -> bool:
    """Return True when a key name looks like a free-form content field.

    Only for sanitizer paths that must drop content-shaped fields entirely
    (see `CONTENT_KEY_FRAGMENTS` above for why this must not be applied to
    tuned-configuration metadata).
    """
    normalized = _normalize_key_name(key)
    return any(fragment in normalized for fragment in CONTENT_KEY_FRAGMENTS)


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


def _redact_credential_key_value(value: Any) -> Any:
    """Redact a value that sits under a credential-like key name.

    Once a key is credential-like, the ENTIRE subtree beneath it is treated as
    credential material: every string is masked fully (not value-scanned --
    partial regex masking would leak adjacent unmatched secret material, and a
    ``[REDACTED``-prefixed value must not be trusted as already-safe), while
    non-string leaves are preserved. Containers recurse through THIS function
    (not the value-only scanner) so a secret nested one level down --
    ``{"api_key": {"value": "sk-..."}}`` -- cannot slip through under an
    innocuous inner key.

    Numeric telemetry survives: ``is_credential_key_name`` matches the substring
    ``token``, so ``total_tokens`` / ``max_tokens`` trip the credential check,
    but their integer values pass through unmasked (redacting token counts would
    corrupt usage/cost telemetry). Note the conservative tradeoff of the shared
    substring-matching helper: STRING values under substring-matching keys such
    as ``author`` (matches ``auth``) or ``tokenizer`` (matches ``token``) are
    masked -- a safe-side telemetry loss, not a leak.

    Known limitation: because numeric values are preserved (for token counts), a
    numeric-valued secret -- e.g. ``{"password": 123456}`` -- is NOT masked. This
    is accepted: real credentials (API keys, tokens, passwords, JWTs) are
    strings and are masked; a numeric can't be distinguished from token-count
    telemetry by type without corrupting the latter.
    """
    if isinstance(value, str):
        return _CREDENTIAL_KEY_REDACTION
    if isinstance(value, Mapping):
        return {key: _redact_credential_key_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_credential_key_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_credential_key_value(item) for item in value)
    if isinstance(value, set):
        return {_redact_credential_key_value(item) for item in value}
    return redact_sensitive_data(value)


def redact_sensitive_data(value: Any, *, redact_credential_keys: bool = False) -> Any:
    """Return a recursively redacted copy of JSON-like data.

    ``redact_credential_keys`` additionally masks any value whose KEY NAME is
    credential-like (``is_credential_key_name``) -- not just values that match a
    secret VALUE regex. It is OPT-IN and off by default: it hardens egress of
    ARBITRARY user-supplied bags (observability trace ``metadata`` / ``input`` /
    ``output``), where a low-entropy secret can hide under a credential-named key
    and evade the value scan. It is deliberately NOT applied to the typed /
    bounded call sites (auth metadata, trial serialization, config), because the
    substring key match (``auth``, ``token``) would over-redact legitimate
    non-secret fields there -- e.g. ``auth_source``, ``tokenizer`` -- which do
    not carry arbitrary user keys.
    """
    if isinstance(value, str):
        return redact_sensitive_text(value)

    if isinstance(value, Mapping):
        return {
            key: (
                _redact_credential_key_value(item)
                if redact_credential_keys and is_credential_key_name(str(key))
                else redact_sensitive_data(
                    item, redact_credential_keys=redact_credential_keys
                )
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            redact_sensitive_data(item, redact_credential_keys=redact_credential_keys)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            redact_sensitive_data(item, redact_credential_keys=redact_credential_keys)
            for item in value
        )

    if isinstance(value, set):
        return {
            redact_sensitive_data(item, redact_credential_keys=redact_credential_keys)
            for item in value
        }

    return value
