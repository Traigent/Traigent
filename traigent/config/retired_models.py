"""Canonical retired/delisted model-ID denylist (#1936/#1937).

Single source of truth consumed by BOTH the runtime validation fallback
(:meth:`traigent.integrations.model_discovery.base.ModelDiscovery.is_valid_model`)
and the catalog-currency drift tests. The regex pattern fallback exists to
accept unknown-but-plausible NEW model IDs; without a denylist it also
re-admitted every retired ID whose *shape* still matches (e.g.
``claude-3-opus-20240229``, dated ``o1-preview`` aliases, the Gemini 1.5
family, ``models/gemini-pro``) — silently undoing the catalog sweep.

Membership is checked on the NORMALIZED ID (``models/`` prefix, Bedrock-style
vendor prefixes, and Bedrock ``-vN[:M]`` version suffixes stripped) so
provider-prefixed alias forms such as ``anthropic.claude-3-opus-20240229-v1:0``
resolve to the same retired model.
"""

from __future__ import annotations

import re

__all__ = ["RETIRED_MODEL_IDS", "normalize_model_id", "is_retired_model"]

# Known-retired / delisted model IDs that must never be served by discovery
# fallbacks nor re-admitted by shape-based pattern validation. Alias forms
# (dated, -latest, models/-prefixed) are listed explicitly; provider-prefixed
# forms are handled by ``normalize_model_id``.
RETIRED_MODEL_IDS: frozenset[str] = frozenset(
    {
        # OpenAI o1 preview line (superseded), incl. dated aliases
        "o1-preview",
        "o1-mini",
        "o1-preview-2024-09-12",
        "o1-mini-2024-09-12",
        # Anthropic Claude 3 Opus (retirement track), incl. -latest alias
        "claude-3-opus-20240229",
        "claude-3-opus-latest",
        # Gemini 1.x (deprecated) + 2.0 preview, incl. -latest and
        # "models/"-prefixed aliases of the same retired models
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-pro-latest",
        "gemini-2.0-flash-exp",
        "models/gemini-2.0-flash-exp",
        "gemini-1.0-pro",
        "gemini-pro",
        "gemini-pro-vision",
        "models/gemini-pro",
        "models/gemini-pro-vision",
    }
)

# Bedrock-style vendor prefixes (``anthropic.claude-...``, ``meta.llama...``).
_VENDOR_PREFIX_RE = re.compile(r"^(anthropic|amazon|meta|mistral|cohere|ai21)\.")

# Bedrock version suffixes: ``-v1:0``, ``-v2``, ``-v14:0:8k`` …
_BEDROCK_VERSION_SUFFIX_RE = re.compile(r"-v\d+(?::\d+)*$")


def normalize_model_id(model_id: str) -> str:
    """Collapse alias forms of a model ID to its canonical spelling.

    Strips a ``models/`` resource prefix (Gemini), a Bedrock vendor prefix
    (``anthropic.`` …), and a Bedrock ``-vN[:M…]`` version suffix. Does NOT
    collapse ``-latest`` or dated aliases — those are distinct catalog entries
    listed explicitly in :data:`RETIRED_MODEL_IDS`.
    """
    normalized = model_id.strip()
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]
    normalized = _VENDOR_PREFIX_RE.sub("", normalized)
    normalized = _BEDROCK_VERSION_SUFFIX_RE.sub("", normalized)
    return normalized


def is_retired_model(model_id: str) -> bool:
    """True when ``model_id`` (raw or normalized) names a retired model."""
    if not model_id:
        return False
    return (
        model_id in RETIRED_MODEL_IDS
        or normalize_model_id(model_id) in RETIRED_MODEL_IDS
    )
