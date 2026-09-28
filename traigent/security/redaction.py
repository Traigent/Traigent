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
# M5/A3 (addendum R2): a SINGLE canonical sensitive-root list, identical in
# both SDKs, matched against keys normalized by `_normalize_key_name`
# (lowercased, with "_"/"-"/"." stripped entirely -- not just collapsed to a
# single separator) so `privateKey`, `private_key`, and `private-key` are all
# the same normalized fragment ("privatekey"). Every entry below is ALREADY
# in that stripped form. Some entries (`session_token` -> "sessiontoken",
# `client_secret` -> "clientsecret") are substring-redundant with a shorter
# root already in the set (`token`, `secret`) -- kept anyway because the
# addendum names them explicitly as canonical roots, and an explicit entry
# survives a future removal of the shorter root that would otherwise
# silently drop coverage.
CREDENTIAL_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {
        "accesskey",
        "apikey",
        "auth",  # also matches "authorization"
        "bearer",
        "clientsecret",
        "cookie",
        "credential",
        "creditcard",
        "jwt",
        "password",
        "passwd",
        "privatekey",
        "pwd",
        "secret",
        "sessiontoken",
        "token",
    }
)

# M5/A3: the ONLY numeric leaves exempt from credential-key-subtree masking.
# `usage.{prompttokens,completiontokens,totaltokens}` (NORMALIZED keys -- see
# `_normalize_key_name`) -- the immediate parent's normalized key must be
# exactly "usage" -- and a bare `maxtokens` model parameter at any
# depth/parent. A `total_tokens`-shaped key that is NOT nested directly under
# `usage` (including one with no parent at all, or one nested inside an
# already credential-flagged subtree such as `api_key`) is NOT exempt: it is
# masked like any other numeric secret. See `_is_approved_numeric_counter_key`
# and `redact_sensitive_data`. These sets hold already-normalized (separator-
# stripped) forms because `_is_approved_numeric_counter_key` compares against
# `_normalize_key_name(key)`, never the raw key.
_APPROVED_USAGE_COUNTER_PARENT = "usage"
_APPROVED_USAGE_COUNTER_KEYS: frozenset[str] = frozenset(
    {"prompttokens", "completiontokens", "totaltokens"}
)
_APPROVED_MODEL_PARAMETER_KEYS: frozenset[str] = frozenset({"maxtokens"})

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
    """A3: lowercase and STRIP (not just collapse) "_", "-", "." entirely, so
    `privateKey`, `private_key`, and `private-key` all normalize to the same
    string (`privatekey`) and match the same canonical root."""
    normalized = key.strip().lower()
    for separator in ("_", "-", "."):
        normalized = normalized.replace(separator, "")
    return normalized


def _is_approved_numeric_counter_key(*, parent_key: str | None, key: str) -> bool:
    """True iff `key` (under `parent_key`) is one of M5's approved exemptions.

    Only a numeric LEAF at one of these exact (parent, key) shapes is exempt
    from credential-key numeric masking -- the caller must additionally check
    the value is a non-bool `int`/`float` before treating it as exempt (a
    stray string at one of these paths is still masked, safe-side).
    """
    normalized_key = _normalize_key_name(key)
    if normalized_key in _APPROVED_MODEL_PARAMETER_KEYS:
        return True
    if normalized_key not in _APPROVED_USAGE_COUNTER_KEYS:
        return False
    if parent_key is None:
        return False
    return _normalize_key_name(parent_key) == _APPROVED_USAGE_COUNTER_PARENT


def _is_exempt_numeric_counter(value: Any, *, parent_key: str | None, key: str) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return _is_approved_numeric_counter_key(parent_key=parent_key, key=key)


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
    credential material and sensitivity is INHERITED by every descendant
    (M5): every string AND number leaf is masked fully (not value-scanned --
    partial regex masking would leak adjacent unmatched secret material, and a
    ``[REDACTED``-prefixed value must not be trusted as already-safe), while
    booleans and ``None`` pass through unchanged. Containers recurse through
    THIS function (not the value-only scanner) so a secret nested one level
    down -- ``{"api_key": {"value": "sk-..."}}`` -- cannot slip through under
    an innocuous inner key. This function never applies the M5 usage-counter
    exemption: a numeric counter nested inside an already credential-flagged
    subtree (e.g. under ``api_key``) is adversarial-shaped, not legitimate
    telemetry, so it stays masked; the exemption is applied one level up, in
    `redact_sensitive_data`, before a key is ever routed here.

    Note the conservative tradeoff of the shared substring-matching helper:
    values under substring-matching keys such as ``author`` (matches
    ``auth``) or ``tokenizer`` (matches ``token``) are masked too -- a
    safe-side telemetry loss, not a leak.
    """
    if isinstance(value, str):
        return _CREDENTIAL_KEY_REDACTION
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
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
    return _redact_sensitive_data(
        value, redact_credential_keys=redact_credential_keys, parent_key=None
    )


def _redact_sensitive_data(
    value: Any, *, redact_credential_keys: bool, parent_key: str | None
) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value)

    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if redact_credential_keys and _is_exempt_numeric_counter(
                item, parent_key=parent_key, key=key_str
            ):
                # M5 exemption: an approved usage-counter/model-parameter
                # numeric leaf is never routed through the credential-subtree
                # collapse below, even though its key name (e.g.
                # "total_tokens") substring-matches the "token" credential
                # root.
                result[key] = _redact_sensitive_data(
                    item,
                    redact_credential_keys=redact_credential_keys,
                    parent_key=key_str,
                )
            elif redact_credential_keys and is_credential_key_name(key_str):
                result[key] = _redact_credential_key_value(item)
            else:
                result[key] = _redact_sensitive_data(
                    item,
                    redact_credential_keys=redact_credential_keys,
                    parent_key=key_str,
                )
        return result

    if isinstance(value, list):
        return [
            _redact_sensitive_data(
                item,
                redact_credential_keys=redact_credential_keys,
                parent_key=parent_key,
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _redact_sensitive_data(
                item,
                redact_credential_keys=redact_credential_keys,
                parent_key=parent_key,
            )
            for item in value
        )

    if isinstance(value, set):
        return {
            _redact_sensitive_data(
                item,
                redact_credential_keys=redact_credential_keys,
                parent_key=parent_key,
            )
            for item in value
        }

    return value
