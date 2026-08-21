"""Normalize provider-reported prompt-cache token counts.

Traigent#2068. Every supported provider reports cached input tokens, at rates far
below fresh input — measured on Bedrock (Claude Sonnet 4.5, eu-west-1, 2026-07-31),
cache-read input is ``$0.0003/1k`` vs ``$0.003/1k`` fresh, exactly **10x** cheaper.
The SDK captured none of it, so reported cost was materially wrong for any cached
workload, in a product whose headline claim is cost optimization.

Two things make this harder than reading one more key.

**Providers disagree on whether cached tokens are inside the input count.**

    Bedrock   reports cache reads DISJOINTLY from ``inputTokens``. A real probe
              returned ``{"inputTokens": 6, "cacheReadInputTokens": 4609}``.
    OpenAI    reports ``cached_tokens`` as a SUBSET of ``prompt_tokens``.

Adding the two together is wrong for OpenAI; not adding them is wrong for Bedrock.
So this module normalizes to one convention — **``input_tokens`` is fresh input
only, exclusive of cache reads** — and subtracts where the provider is inclusive.
Everything downstream can then do simple arithmetic without knowing who sent it.

**Absent is not zero.** A provider that does not report a cache field is recorded as
``None``, never ``0``. Defaulting a silent provider to zero makes "no cache dimension
reported" indistinguishable from "no cache was used", which yields a confidently
wrong cost rather than an honest unknown — Amazon Nova omits the keys entirely when
no cache is engaged, which is exactly that trap. ``unreported_fields`` then names
which fields were withheld, so a genuine unknown is distinguishable from a producer
that predates the field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CacheUsage", "normalize_cache_usage"]

# Field names, per provider, and whether the provider's input count already
# INCLUDES the cache-read tokens. Ordered so the first shape that matches wins.
#
#   (read_path, creation_path, input_is_inclusive)
#
# A path is a tuple walked through nested dicts, so ``prompt_tokens_details.
# cached_tokens`` is ("prompt_tokens_details", "cached_tokens").
_PROVIDER_SHAPES: tuple[
    tuple[str, tuple[str, ...], tuple[str, ...] | None, bool], ...
] = (
    # OpenAI Chat Completions — cached_tokens is a subset of prompt_tokens.
    ("openai_chat", ("prompt_tokens_details", "cached_tokens"), None, True),
    # OpenAI Responses — same semantics, different key.
    ("openai_responses", ("input_tokens_details", "cached_tokens"), None, True),
    # Anthropic — reported alongside input_tokens, not inside it.
    (
        "anthropic",
        ("cache_read_input_tokens",),
        ("cache_creation_input_tokens",),
        False,
    ),
    # Bedrock Converse — camelCase, disjoint. Verified empirically for both
    # Anthropic-on-Bedrock and Amazon Nova.
    ("bedrock", ("cacheReadInputTokens",), ("cacheWriteInputTokens",), False),
    # Google Gemini — cachedContentTokenCount is a SUBSET of promptTokenCount, not
    # disjoint from it (Traigent#2111). Google's own example:
    # promptTokenCount=696219, cachedContentTokenCount=696190 -- the cached count is
    # smaller than and contained in the prompt count. Treating this as disjoint (the
    # historical, wrong setting) double-counts every cached token once the input key
    # is read at all, producing exactly a 2x overcount of billable input.
    ("gemini", ("cachedContentTokenCount",), None, True),
    # The Google Python client exposes this same usage metadata in snake_case.
    ("gemini", ("cached_content_token_count",), None, True),
)

# Anthropic reports the cache-write TTL split; a single request may contain both.
# The tiers are priced differently (1.25x base input at 5 minutes, 2x at 1 hour), so
# the split has to survive rather than be flattened. See TraigentSchema#383.
_ANTHROPIC_TTL_PATH = ("cache_creation",)
_TTL_KEYS = {
    "ephemeral_5m_input_tokens": "ephemeral_5m",
    "ephemeral_1h_input_tokens": "ephemeral_1h",
}


@dataclass(frozen=True)
class CacheUsage:
    """Normalized prompt-cache counts for one model call.

    ``input_tokens`` is fresh input only — cache reads are never included, whatever
    convention the provider used on the wire.
    """

    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_creation_tokens_by_ttl: dict[str, int] = field(default_factory=dict)
    unreported_fields: tuple[str, ...] = ()
    provider_shape: str | None = None

    @property
    def is_complete(self) -> bool:
        """False when at least one cache dimension was never reported."""
        return not self.unreported_fields

    @property
    def billable_input_tokens(self) -> int | None:
        """Every input token the provider will invoice, cached or not.

        ``None`` when the fresh count itself is unknown. Unknown cache counts
        contribute nothing rather than being guessed at, so check ``is_complete``
        before treating this as a total rather than a lower bound.
        """
        if self.input_tokens is None:
            return None
        return (
            self.input_tokens
            + (self.cache_read_tokens or 0)
            + (self.cache_creation_tokens or 0)
        )

    def as_metadata(self) -> dict[str, Any]:
        """Trial-metadata form, for surfacing cache usage per trial (#2069)."""
        data: dict[str, Any] = {
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_usage_complete": self.is_complete,
        }
        if self.cache_creation_tokens_by_ttl:
            data["cache_creation_tokens_by_ttl"] = dict(
                self.cache_creation_tokens_by_ttl
            )
        if self.unreported_fields:
            data["unreported_usage_fields"] = list(self.unreported_fields)
        if self.provider_shape:
            data["provider_shape"] = self.provider_shape
        return data


def _dig(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _coerce_count(raw: Any) -> int | None:
    """A token count, or None when the value is absent or not a usable count.

    Junk is reported as unknown rather than coerced to zero: a wrong number is worse
    than an honest gap, because only the gap is visible downstream.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, float):
        # inf/-inf/nan are not counts. `json.loads` accepts bare `Infinity`, so a
        # provider payload really can carry one, and `int(inf)` raises
        # OverflowError -- which the handler below did not catch, so the whole
        # normalization crashed instead of reporting an honest gap.
        if not math.isfinite(raw):
            return None
        # A fractional token count is junk, not a count to round. Truncating
        # 4609.7 to 4609 invents a precise-looking number from a broken input,
        # which is the coercion this function's docstring exists to refuse -- and
        # it is what the JS SDK already refuses (`undefined`).
        if not raw.is_integer():
            return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def _read_input_tokens(payload: dict[str, Any]) -> int | None:
    for key in (
        "input_tokens",
        "inputTokens",
        "prompt_tokens",
        "promptTokenCount",
        "prompt_token_count",
    ):
        if key in payload:
            return _coerce_count(payload[key])
    return None


def _read_ttl_split(payload: dict[str, Any]) -> dict[str, int]:
    node = _dig(payload, _ANTHROPIC_TTL_PATH)
    if not isinstance(node, dict):
        return {}
    split: dict[str, int] = {}
    for wire_key, canonical in _TTL_KEYS.items():
        value = _coerce_count(node.get(wire_key))
        if value is not None:
            split[canonical] = value
    return split


def normalize_cache_usage(usage: dict[str, Any] | None) -> CacheUsage:
    """Read prompt-cache counts out of any supported provider's usage payload.

    Returns fresh (cache-exclusive) input alongside the cache dimensions. Fields the
    provider did not report stay ``None`` and are named in ``unreported_fields``.
    """
    if not isinstance(usage, dict):
        return CacheUsage(
            unreported_fields=("cache_read_tokens", "cache_creation_tokens")
        )

    input_tokens = _read_input_tokens(usage)
    read_tokens: int | None = None
    creation_tokens: int | None = None
    shape: str | None = None
    input_is_inclusive = False

    for name, read_path, creation_path, inclusive in _PROVIDER_SHAPES:
        candidate_read = _coerce_count(_dig(usage, read_path))
        candidate_creation = (
            _coerce_count(_dig(usage, creation_path)) if creation_path else None
        )
        if candidate_read is None and candidate_creation is None:
            continue
        read_tokens = candidate_read
        creation_tokens = candidate_creation
        shape = name
        input_is_inclusive = inclusive
        break

    ttl_split = _read_ttl_split(usage)
    if creation_tokens is None and ttl_split:
        # Anthropic can report the per-tier split without a flat total.
        creation_tokens = sum(ttl_split.values())
        shape = shape or "anthropic"

    # Normalize to the disjoint convention: input_tokens excludes cache reads.
    # Without this, the same three numbers mean two different costs depending on
    # which provider sent them.
    if input_is_inclusive and input_tokens is not None and read_tokens:
        input_tokens = max(0, input_tokens - read_tokens)

    unreported = tuple(
        name
        for name, value in (
            ("cache_read_tokens", read_tokens),
            ("cache_creation_tokens", creation_tokens),
        )
        if value is None
    )

    return CacheUsage(
        input_tokens=input_tokens,
        cache_read_tokens=read_tokens,
        cache_creation_tokens=creation_tokens,
        cache_creation_tokens_by_ttl=ttl_split,
        unreported_fields=unreported,
        provider_shape=shape,
    )
