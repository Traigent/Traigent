"""LLM cost calculation utilities.

Cost resolution is intentionally fail-fast:
- Runtime pricing uses litellm first, then Traigent's curated fallback pricing
  for built-in supported models that litellm does not currently price.
- Unknown models raise with actionable remediation by default.
- Users can provide explicit custom pricing for private/unsupported models.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

import json
import logging
import math
import os
import re
import threading
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from traigent.utils.logging import configure_litellm_logging

logger = logging.getLogger(__name__)

# Privacy default: force litellm to use its bundled pricing map instead of
# fetching the remote one from GitHub on import. This prevents `import traigent`
# from making an outbound network request before any user-initiated optimization
# or pricing call. Users who want the remote pricing map can opt back in by
# setting LITELLM_LOCAL_MODEL_COST_MAP=false in their environment BEFORE
# importing traigent. Previously this gate fired only when the user explicitly
# set TRAIGENT_OFFLINE_MODE=true, which meant the default behavior leaked an
# outbound HTTP request in offline/air-gapped/regulated environments. (See #912.)
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# Import litellm with graceful fallback
try:
    import litellm

    configure_litellm_logging(litellm_module=litellm)
    LITELLM_AVAILABLE = True
except (ImportError, KeyError):
    litellm = None  # type: ignore[assignment]
    LITELLM_AVAILABLE = False

# Backward compatibility alias
TOKENCOST_AVAILABLE = LITELLM_AVAILABLE

# Canonical model name constants (referenced in ESTIMATION_MODEL_PRICING)
_GPT35_TURBO = "gpt-3.5-turbo"
_GPT4O = "gpt-4o"

_CUSTOM_PRICING_FILE_ENV = "TRAIGENT_CUSTOM_MODEL_PRICING_FILE"
_CUSTOM_PRICING_JSON_ENV = "TRAIGENT_CUSTOM_MODEL_PRICING_JSON"
_CUSTOM_PRICING_CACHE: dict[str, tuple[float, float]] | None = None
_CUSTOM_PRICING_CACHE_KEY: tuple[str, str] | None = None
_CUSTOM_PRICING_LOCK = threading.Lock()

# Estimation pricing for pre-optimization cost estimation (per-token costs).
# Post-call cost tracking uses litellm exclusively via cost_from_tokens().
ESTIMATION_MODEL_PRICING = {
    # OpenAI
    _GPT4O: {"input_cost_per_token": 2.5e-6, "output_cost_per_token": 10.0e-6},
    "gpt-4o-mini": {"input_cost_per_token": 0.15e-6, "output_cost_per_token": 0.6e-6},
    "gpt-4-turbo": {"input_cost_per_token": 10.0e-6, "output_cost_per_token": 30.0e-6},
    _GPT35_TURBO: {"input_cost_per_token": 0.5e-6, "output_cost_per_token": 1.5e-6},
    # Anthropic
    "claude-3-5-sonnet-20241022": {
        "input_cost_per_token": 3.0e-6,
        "output_cost_per_token": 15.0e-6,
    },
    "claude-3-5-haiku-20241022": {
        "input_cost_per_token": 0.8e-6,
        "output_cost_per_token": 4.0e-6,
    },
    "claude-3-opus-20240229": {
        "input_cost_per_token": 15.0e-6,
        "output_cost_per_token": 75.0e-6,
    },
    "claude-3-haiku-20240307": {
        "input_cost_per_token": 0.25e-6,
        "output_cost_per_token": 1.25e-6,
    },
    "claude-sonnet-4-20250514": {
        "input_cost_per_token": 3.0e-6,
        "output_cost_per_token": 15.0e-6,
    },
    "claude-opus-4-20250514": {
        "input_cost_per_token": 15.0e-6,
        "output_cost_per_token": 75.0e-6,
    },
    # Google
    "gemini-1.5-pro": {
        "input_cost_per_token": 1.25e-6,
        "output_cost_per_token": 5.0e-6,
    },
    "gemini-1.5-flash": {
        "input_cost_per_token": 0.075e-6,
        "output_cost_per_token": 0.3e-6,
    },
}

# Canonical model-name aliases used by estimation and validation code paths.
# This avoids duplicated alias tables drifting across modules.
#
# NOTE: Bare aliases (e.g. "claude-sonnet") are pinned to *dated* model IDs
# so budget enforcement and cost accounting resolve to concrete model versions
# at runtime instead of provider-latest pointers. Update the targets here only
# after confirming the dated ID has an entry in Traigent's canonical pricing
# table and, ideally, the pinned litellm cost map as well.
MODEL_NAME_ALIASES: dict[str, str] = {
    "gpt-4": "gpt-4-turbo",
    "gpt-4-32k": "gpt-4-turbo",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-haiku": "claude-3-haiku-20240307",
    "claude-3-sonnet": "claude-3-5-sonnet-20241022",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-sonnet-20240229": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-opus": "claude-3-opus-20240229",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
}

# Warn-once cache to de-noise fallback pricing warnings (thread-safe)
_warned_models: set[str] = set()
_warned_models_lock = threading.Lock()


class UnknownModelError(KeyError):
    """Raised when a model's pricing cannot be determined.

    Subclasses KeyError for backward compatibility with code that catches
    KeyError for unknown model lookups (similar to tokencost behavior).
    """

    pass


def _normalize_model_name(model: str) -> str:
    """Normalize model name for litellm lookup.

    Handles:
    - Whitespace stripping
    - Provider prefixes with / (e.g., "openai/gpt-4o" -> "gpt-4o")
    - Provider prefixes with : (e.g., "anthropic:claude-3" -> "claude-3")
    - Short model names after : (e.g., "openai:o1" -> "o1")
    - Bedrock-style numeric version suffixes are preserved (e.g., "model-v1:0" keeps ":0")

    Note:
        Ollama-style "model:tag" identifiers (e.g., "llama3:latest") will have
        the tag stripped. For tagged models, configure explicit aliasing or
        custom pricing if needed.

    Args:
        model: Raw model name

    Returns:
        Normalized model name for lookup
    """
    model = model.strip()

    # Handle / prefix (e.g., "openai/gpt-4o" or "bedrock/anthropic.claude...")
    if "/" in model:
        model = model.split("/")[-1]

    # Handle : prefix ONLY if it looks like "provider:model" pattern
    # NOT Bedrock-style "model-version:revision" (e.g., "claude-3-haiku-20240307-v1:0")
    # Heuristic: only split on : if:
    # 1. Colon appears early (within first 15 chars) - provider names are short
    # 2. Suffix is NOT purely numeric (numeric = Bedrock version like ":0", ":10")
    # This handles:
    # - "anthropic:claude-3" -> "claude-3" (provider prefix stripped)
    # - "openai:o1" -> "o1" (provider prefix stripped, short model name OK)
    # - "model-v1:0" -> preserved (Bedrock version revision)
    # NOTE: Ollama-style "model:tag" (e.g., "llama3:latest") will be stripped to "tag"
    # which may cause lookup issues. Use full model name for Ollama models.
    if ":" in model:
        colon_idx = model.index(":")
        after_colon = model[colon_idx + 1 :]
        # Don't split if colon is late (not a provider prefix)
        # or if suffix is purely numeric (Bedrock version revision like :0, :10)
        if colon_idx < 15 and not after_colon.isdigit():
            model = after_colon

    return model


def _is_model_known_to_litellm(model: str) -> bool:
    """Check if model is in litellm's model_cost database.

    This includes models with 0.0 pricing (e.g., free tiers, self-hosted).
    Performs case-insensitive lookup and handles provider prefixes.
    """
    if not LITELLM_AVAILABLE:
        return False

    # Try multiple variants for robustness
    variants = [
        model,  # Original
        model.lower(),  # Lowercase
        _normalize_model_name(model),  # Normalized
        _normalize_model_name(model).lower(),  # Normalized + lowercase
    ]

    # Build lowercase lookup set for case-insensitive matching
    model_cost_lower = {k.lower(): k for k in litellm.model_cost}

    for variant in variants:
        if variant in litellm.model_cost:
            return True
        if variant.lower() in model_cost_lower:
            return True

    return False


def _resolve_litellm_alias(model: str) -> str:
    """Resolve model aliases through litellm.model_alias_map if present."""
    normalized = _normalize_model_name(model)
    if not LITELLM_AVAILABLE:
        return normalized

    alias_map = getattr(litellm, "model_alias_map", None)
    if not isinstance(alias_map, dict):
        return normalized

    candidates = (model, normalized, model.lower(), normalized.lower())
    for candidate in candidates:
        mapped = alias_map.get(candidate)
        if isinstance(mapped, str) and mapped:
            return mapped

    return normalized


def _parse_custom_pricing_entry(
    model: str, entry: Any, source: str
) -> tuple[float, float]:
    """Parse one custom pricing entry into (input_cost_per_token, output_cost_per_token)."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"Invalid pricing entry for model '{model}' in {source}: expected object"
        )

    input_val = entry.get("input_cost_per_token", entry.get("input"))
    output_val = entry.get("output_cost_per_token", entry.get("output"))

    if input_val is None or output_val is None:
        raise ValueError(
            f"Invalid pricing entry for model '{model}' in {source}: "
            "expected input_cost_per_token/output_cost_per_token"
        )

    try:
        input_cost = float(input_val)
        output_cost = float(output_val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric pricing for model '{model}' in {source}"
        ) from exc

    if not math.isfinite(input_cost) or not math.isfinite(output_cost):
        raise ValueError(f"Invalid non-finite pricing for model '{model}' in {source}")

    if input_cost < 0 or output_cost < 0:
        raise ValueError(f"Invalid negative pricing for model '{model}' in {source}")

    return input_cost, output_cost


def _reject_non_finite_json_constant(token: str) -> None:
    raise ValueError(f"Non-finite literal '{token}' not allowed in custom pricing")


def _normalize_custom_pricing_map(
    raw: dict[str, Any], source: str
) -> dict[str, tuple[float, float]]:
    """Normalize custom pricing payload into lowercase model index."""
    normalized: dict[str, tuple[float, float]] = {}
    for model, entry in raw.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"Invalid model key in {source}: expected non-empty string"
            )
        input_cost, output_cost = _parse_custom_pricing_entry(model, entry, source)
        keys = {model.lower(), _normalize_model_name(model).lower()}
        for key in keys:
            normalized[key] = (input_cost, output_cost)
    return normalized


def _load_custom_pricing_from_sources() -> dict[str, tuple[float, float]]:
    """Load custom model pricing from env-configured file and/or JSON payload."""
    pricing: dict[str, tuple[float, float]] = {}

    file_path = os.environ.get(_CUSTOM_PRICING_FILE_ENV, "").strip()
    if file_path:
        try:
            with open(file_path, encoding="utf-8") as f:
                file_payload = json.load(
                    f, parse_constant=_reject_non_finite_json_constant
                )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse custom pricing file '{file_path}'."
            ) from exc
        if not isinstance(file_payload, dict):
            raise ValueError(
                f"Invalid custom pricing file '{file_path}': root must be an object"
            )
        pricing.update(_normalize_custom_pricing_map(file_payload, file_path))

    env_json = os.environ.get(_CUSTOM_PRICING_JSON_ENV, "").strip()
    if env_json:
        try:
            env_payload = json.loads(
                env_json, parse_constant=_reject_non_finite_json_constant
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse {_CUSTOM_PRICING_JSON_ENV}: invalid JSON"
            ) from exc
        if not isinstance(env_payload, dict):
            raise ValueError(
                f"Invalid {_CUSTOM_PRICING_JSON_ENV}: root must be an object"
            )
        pricing.update(
            _normalize_custom_pricing_map(env_payload, _CUSTOM_PRICING_JSON_ENV)
        )

    return pricing


def _get_custom_pricing_index() -> dict[str, tuple[float, float]]:
    """Get cached custom model pricing index keyed by lowercase model names."""
    file_path = os.environ.get(_CUSTOM_PRICING_FILE_ENV, "").strip()
    env_json = os.environ.get(_CUSTOM_PRICING_JSON_ENV, "").strip()
    cache_key = (file_path, env_json)

    global _CUSTOM_PRICING_CACHE, _CUSTOM_PRICING_CACHE_KEY
    if _CUSTOM_PRICING_CACHE is not None and _CUSTOM_PRICING_CACHE_KEY == cache_key:
        return _CUSTOM_PRICING_CACHE

    with _CUSTOM_PRICING_LOCK:
        if _CUSTOM_PRICING_CACHE is not None and _CUSTOM_PRICING_CACHE_KEY == cache_key:
            return _CUSTOM_PRICING_CACHE
        _CUSTOM_PRICING_CACHE = _load_custom_pricing_from_sources()
        _CUSTOM_PRICING_CACHE_KEY = cache_key
        return _CUSTOM_PRICING_CACHE


def _try_custom_model_pricing(model: str) -> tuple[float, float] | None:
    """Try resolving pricing from explicit user-provided custom pricing."""
    normalized = _normalize_model_name(model).lower()
    index = _get_custom_pricing_index()
    return index.get(model.lower()) or index.get(normalized)


def _unknown_model_resolution_message(model: str) -> str:
    """Build actionable unknown-model remediation instructions."""
    return (
        f"Model '{model}' has no known pricing. "
        "Fix by choosing one of: "
        "1) use a provider model id that litellm can price, "
        "2) configure litellm.model_alias_map for your alias, "
        f"3) provide explicit pricing via {_CUSTOM_PRICING_FILE_ENV} or {_CUSTOM_PRICING_JSON_ENV}. "
        "Example JSON: "
        '{"my-model":{"input_cost_per_token":1e-6,"output_cost_per_token":2e-6}}'
    )


def _try_litellm_prompt_cost(
    model: str, prompt: str | list[dict[str, Any]] | None, tokens: int
) -> tuple[float | None, int]:
    """Attempt prompt cost calculation via litellm.

    Args:
        model: The model name
        prompt: Input prompt (string, message list, or None)
        tokens: Current token count (updated if prompt is provided)

    Returns:
        Tuple of (cost_or_None, token_count). Cost is None when litellm cannot
        determine pricing and the caller should try custom pricing.
    """
    model_known = _is_model_known_to_litellm(model)
    if prompt is not None:
        if isinstance(prompt, list):
            tokens = litellm.token_counter(model=model, messages=prompt)
        else:
            tokens = litellm.token_counter(model=model, text=prompt)
    # Price via the canonical candidate ladder (normalized id + aliases) so that
    # provider-prefixed ids litellm cannot price directly (e.g.
    # "openrouter/openai/gpt-4o-mini", which raises "This model isn't mapped yet"
    # from ``litellm.cost_per_token``/``completion_cost`` even though
    # ``litellm.model_cost`` HAS the underlying "gpt-4o-mini" entry) still resolve.
    litellm_rates = _try_litellm_cost(_build_model_candidates(model), tokens, 0)
    if litellm_rates is not None:
        prompt_cost = litellm_rates[0]
        if prompt_cost > 0 or model_known:
            return float(prompt_cost), tokens
    return None, tokens


def calculate_prompt_cost(
    prompt: str | list[dict[str, Any]] | None, model: str
) -> float:
    """Calculate INPUT cost only using litellm.

    .. deprecated::
        Use :func:`cost_from_tokens` with pre-computed token counts instead.

    Backward-compatible wrapper for tokencost's calculate_prompt_cost.

    Args:
        prompt: The prompt text or message list (may be None)
        model: The model name

    Returns:
        Cost in USD for the input tokens

    Raises:
        UnknownModelError: If the model's pricing cannot be determined from
            litellm or explicit custom pricing configuration.
    """
    warnings.warn(
        "calculate_prompt_cost() is deprecated. Use cost_from_tokens() with "
        "pre-computed token counts instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    tokens = 0

    if LITELLM_AVAILABLE:
        try:
            result, tokens = _try_litellm_prompt_cost(model, prompt, tokens)
            if result is not None:
                return result
        except Exception:
            logger.debug(
                "litellm cost calculation failed for model %r", model, exc_info=True
            )

    custom_pricing = _try_custom_model_pricing(model)
    if custom_pricing is not None:
        input_cost_per_token, _ = custom_pricing
        return float(max(tokens, 0) * input_cost_per_token)

    raise UnknownModelError(_unknown_model_resolution_message(model))


def _try_litellm_completion_cost(
    model: str, completion: str | None, tokens: int
) -> tuple[float | None, int]:
    """Attempt completion cost calculation via litellm.

    Args:
        model: The model name
        completion: Completion text (or None)
        tokens: Current token count (updated if completion is provided)

    Returns:
        Tuple of (cost_or_None, token_count). Cost is None when litellm cannot
        determine pricing and the caller should try custom pricing.
    """
    model_known = _is_model_known_to_litellm(model)
    if completion is not None:
        tokens = litellm.token_counter(model=model, text=completion)
    # Price via the canonical candidate ladder so provider-prefixed ids litellm
    # cannot price directly (e.g. "openrouter/openai/gpt-4o-mini") still resolve
    # against the underlying entry in ``litellm.model_cost``. (See #1423.)
    litellm_rates = _try_litellm_cost(_build_model_candidates(model), 0, tokens)
    if litellm_rates is not None:
        completion_cost = litellm_rates[1]
        if completion_cost > 0 or model_known:
            return float(completion_cost), tokens
    return None, tokens


def calculate_completion_cost(completion: str | None, model: str) -> float:
    """Calculate OUTPUT cost only using litellm.

    .. deprecated::
        Use :func:`cost_from_tokens` with pre-computed token counts instead.

    Backward-compatible wrapper for tokencost's calculate_completion_cost.

    Args:
        completion: The completion text (may be None)
        model: The model name

    Returns:
        Cost in USD for the output tokens

    Raises:
        UnknownModelError: If the model's pricing cannot be determined from
            litellm or explicit custom pricing configuration.
    """
    warnings.warn(
        "calculate_completion_cost() is deprecated. Use cost_from_tokens() with "
        "pre-computed token counts instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    tokens = 0

    if LITELLM_AVAILABLE:
        try:
            result, tokens = _try_litellm_completion_cost(model, completion, tokens)
            if result is not None:
                return result
        except Exception:
            logger.debug(
                "litellm cost calculation failed for model %r", model, exc_info=True
            )

    custom_pricing = _try_custom_model_pricing(model)
    if custom_pricing is not None:
        _, output_cost_per_token = custom_pricing
        return float(max(tokens, 0) * output_cost_per_token)

    raise UnknownModelError(_unknown_model_resolution_message(model))


def _normalize_model_for_fallback(model: str) -> str:
    """Normalize model name for fallback pricing lookup.

    Uses the shared _normalize_model_name function and adds lowercase.

    Args:
        model: Raw model name

    Returns:
        Normalized model name for lookup (lowercase)
    """
    normalized = _normalize_model_name(model)
    return normalized.lower() if normalized else model.lower()


def _find_fallback_pricing(
    base_model: str,
) -> tuple[dict[str, float] | None, str | None]:
    """Find pricing entry for a normalized model name via exact then prefix match.

    Args:
        base_model: Lower-cased, normalized model name

    Returns:
        Tuple of (pricing_dict, matched_key) or (None, None) if not found.
    """
    # Exact match (case-insensitive via normalized base_model)
    for key, value in ESTIMATION_MODEL_PRICING.items():
        if key.lower() == base_model:
            return value, key

    # Prefix matching - prefer LONGEST matching key
    best_match_len = 0
    pricing = None
    matched_key = None
    for model_key, model_pricing in ESTIMATION_MODEL_PRICING.items():
        key_lower = model_key.lower()
        # Forward longest-prefix match only: base_model starts with a known key.
        # A reverse (key startswith base_model) branch mis-priced partial names
        # like 'gpt-4' to the first dict-order family member (e.g. gpt-4o). See
        # issue #1957. Legitimate short aliases resolve via MODEL_NAME_ALIASES
        # before this lookup, so dropping the reverse branch loses no real match.
        if base_model.startswith(key_lower):
            if len(key_lower) > best_match_len:
                best_match_len = len(key_lower)
                pricing = model_pricing
                matched_key = model_key

    return pricing, matched_key


def _try_builtin_per_token_rates(
    candidates: list[str],
) -> tuple[float, float, str] | None:
    """Try Traigent's curated pricing table for candidate matches.

    First attempts exact normalized matches (fast path). If no exact match
    is found, falls back to longest-prefix matching so that dated model
    versions like ``gpt-4o-2024-11-20`` resolve to ``gpt-4o`` pricing.
    """
    pricing_index = {
        model.lower(): (model, pricing)
        for model, pricing in ESTIMATION_MODEL_PRICING.items()
    }

    # Pass 1: exact match (fast path)
    for candidate in candidates:
        normalized = _normalize_model_for_fallback(candidate)
        match = pricing_index.get(normalized)
        if match is not None:
            matched_key, pricing = match
            return (
                float(pricing["input_cost_per_token"]),
                float(pricing["output_cost_per_token"]),
                matched_key,
            )

    # Pass 2: longest-prefix match (handles dated model versions)
    best: tuple[float, float, str] | None = None
    best_prefix_len = 0
    for candidate in candidates:
        normalized = _normalize_model_for_fallback(candidate)
        for key_lower, (matched_key, pricing) in pricing_index.items():
            if normalized.startswith(key_lower) and len(key_lower) > best_prefix_len:
                best_prefix_len = len(key_lower)
                best = (
                    float(pricing["input_cost_per_token"]),
                    float(pricing["output_cost_per_token"]),
                    matched_key,
                )

    if best is not None:
        logger.debug(
            "Builtin pricing: prefix-matched candidates %r to %r",
            candidates,
            best[2],
        )

    return best


def _estimation_cost_from_tokens(
    model: str, input_tokens: int, output_tokens: int, *, _quiet: bool = False
) -> tuple[float, float]:
    """Calculate costs using estimation pricing dictionary.

    Used for pre-optimization estimation only. Post-call tracking should use
    cost_from_tokens() which uses litellm exclusively.

    Args:
        model: Model name (may include provider prefix)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Tuple of (input_cost, output_cost). Returns (0.0, 0.0) if model not found.
    """
    base_model = _normalize_model_for_fallback(model)

    # Resolve canonical aliases before lookup
    for alias, canonical in MODEL_NAME_ALIASES.items():
        if base_model == alias.lower():
            base_model = canonical.lower()
            break

    pricing, matched_key = _find_fallback_pricing(base_model)

    if pricing:
        # Warn-once per model to avoid log spam
        with _warned_models_lock:
            if model not in _warned_models:
                _warned_models.add(model)
                log_fn = logger.debug if _quiet else logger.warning
                log_fn(
                    "Using estimation pricing for model %r (matched %r)",
                    model,
                    matched_key,
                )
        return (
            input_tokens * pricing["input_cost_per_token"],
            output_tokens * pricing["output_cost_per_token"],
        )

    return 0.0, 0.0


def _resolve_builtin_model_alias(normalized: str) -> str:
    """Resolve through Traigent's built-in alias table.

    Args:
        normalized: Already-normalized model name (output of ``_normalize_model_name``).
    """
    return MODEL_NAME_ALIASES.get(normalized.lower(), normalized)


def _strip_date_suffix(model: str) -> str | None:
    """Strip trailing date suffix (``-YYYY-MM-DD``) from a model name.

    Many providers return dated model versions (e.g. ``gpt-4o-2024-11-20``)
    that may not appear in pricing tables.  Stripping the date yields the
    base model name (``gpt-4o``) which usually *is* priced.

    Returns:
        The base model name if a date suffix was found, else ``None``.
    """
    # Match both "gpt-4o-2024-11-20" (YYYY-MM-DD) and "claude-3-5-sonnet-20241022" (YYYYMMDD)
    m = re.search(r"-\d{4}(?:-\d{2}-\d{2}|\d{4})(-v\d+)?$", model)
    if m:
        return model[: m.start()]
    return None


def _build_model_candidates(model_name: str) -> list[str]:
    """Build deduplicated list of model name candidates for pricing lookup.

    Normalization is applied exactly once; all downstream resolvers receive
    the already-normalized form so that mixed-case provider prefixes like
    ``OPENAI/GPT-4O`` resolve correctly.

    Date-suffixed model names (e.g. ``gpt-4o-2024-11-20``) also generate
    their base name (``gpt-4o``) as a candidate so that pricing resolution
    succeeds even when the exact dated version is missing from a table.
    """
    normalized = _normalize_model_name(model_name)
    lowered = normalized.lower()
    builtin_alias = _resolve_builtin_model_alias(lowered)
    litellm_alias = _resolve_litellm_alias(lowered)
    builtin_then_litellm = _resolve_litellm_alias(builtin_alias)

    # Strip date suffix (e.g. "gpt-4o-2024-11-20" → "gpt-4o") to produce
    # a base-model candidate that is more likely to appear in pricing tables.
    base_from_date = _strip_date_suffix(lowered)

    candidates = [
        c
        for c in [
            model_name,
            normalized,
            lowered,
            builtin_alias,
            litellm_alias,
            builtin_then_litellm,
            base_from_date,
        ]
        if c
    ]
    return list(dict.fromkeys(candidates))


def _try_litellm_per_token_rates(
    candidates: list[str],
) -> tuple[float, float] | None:
    """Try litellm.cost_per_token for each candidate, returning per-token rates.

    Returns:
        ``(input_cost_per_token, output_cost_per_token)`` or ``None``.
    """
    if not LITELLM_AVAILABLE:
        return None
    for candidate in candidates:
        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=candidate, prompt_tokens=1, completion_tokens=1
            )
            if input_cost > 0 or output_cost > 0:
                return float(input_cost), float(output_cost)
            if _is_model_known_to_litellm(candidate):
                return float(input_cost), float(output_cost)
        except Exception:
            logger.debug(
                "litellm pricing lookup failed for %r", candidate, exc_info=True
            )
    return None


def get_model_token_pricing(model_name: str) -> tuple[float, float, str]:
    """Get per-token pricing for a model with fail-fast behavior.

    Resolution order:
    1. litellm pricing database
    2. explicit custom pricing overrides (env/file)
    3. Traigent built-in fallback pricing for supported models

    Unknown models raise ``UnknownModelError`` with remediation instructions.

    Args:
        model_name: Model identifier (may include provider prefix).

    Returns:
        ``(input_cost_per_token, output_cost_per_token, estimation_method)``
    """
    if not model_name or not model_name.strip():
        raise UnknownModelError(_unknown_model_resolution_message(model_name))

    candidates = _build_model_candidates(model_name)

    litellm_rates = _try_litellm_per_token_rates(candidates)
    if litellm_rates is not None:
        return litellm_rates[0], litellm_rates[1], "litellm"

    custom_pricing = _try_custom_model_pricing(model_name)
    if custom_pricing is not None:
        input_cost, output_cost = custom_pricing
        return float(input_cost), float(output_cost), "custom_pricing"

    builtin_rates = _try_builtin_per_token_rates(candidates)
    if builtin_rates is not None:
        input_cost, output_cost, _matched_key = builtin_rates
        return input_cost, output_cost, "builtin_pricing"

    raise UnknownModelError(_unknown_model_resolution_message(model_name))


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an LLM request."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_used: str = ""
    mapped_model: str = ""
    calculation_method: str = "unknown"

    def __post_init__(self) -> None:
        """Ensure total cost is sum of input and output costs."""
        if not self.total_cost:
            self.total_cost = self.input_cost + self.output_cost
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class CostCalculator:
    """LLM cost calculator using strict canonical pricing resolution."""

    def __init__(self, logger=None, enable_caching: bool = True) -> None:
        """Initialize the cost calculator.

        Args:
            logger: Optional logger instance for debug/warning messages
            enable_caching: Deprecated and ignored (kept for compatibility)
        """
        self.logger = logger
        self.enable_caching = enable_caching

        if not LITELLM_AVAILABLE:
            raise RuntimeError(
                "litellm is required for CostCalculator but is not installed. "
                "Install it with: pip install litellm"
            )

    def calculate_cost(
        self,
        prompt: str | list[dict[str, Any]] | None = None,
        response: str | None = None,
        model_name: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> CostBreakdown:
        """Calculate cost for an LLM request.

        Args:
            prompt: The input prompt (string or message list)
            response: The model response text
            model_name: The model identifier
            input_tokens: Optional pre-calculated input token count
            output_tokens: Optional pre-calculated output token count

        Returns:
            CostBreakdown with detailed cost information

        Raises:
            UnknownModelError: If model has no pricing — budget-critical,
                callers must handle this to avoid silent budget overruns.
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError(
                "litellm is required for cost calculation but is not installed. "
                "Install it with: pip install litellm"
            )

        if not model_name:
            return CostBreakdown(calculation_method="no_model_name")

        normalized_model = _normalize_model_name(model_name).lower()
        builtin_alias = _resolve_builtin_model_alias(normalized_model)
        effective_model = _resolve_litellm_alias(builtin_alias)

        result = CostBreakdown(
            model_used=model_name, mapped_model=effective_model or ""
        )

        try:
            self._populate_cost(
                result,
                prompt,
                response,
                effective_model,
                input_tokens,
                output_tokens,
            )
        except UnknownModelError:
            raise
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Cost calculation failed for model {model_name}: {e}"
                )
            result.calculation_method = f"error_{type(e).__name__}"

        return result

    def _populate_cost(
        self,
        result: CostBreakdown,
        prompt: str | list[dict[str, Any]] | None,
        response: str | None,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        """Fill cost breakdown using the best available data."""
        if prompt and response:
            result.input_cost = self._safe_calculate_prompt_cost(prompt, model)
            result.output_cost = self._safe_calculate_completion_cost(response, model)
            result.calculation_method = "prompt_and_response"
        elif input_tokens is not None or output_tokens is not None:
            actual_input = max(input_tokens or 0, 0)
            actual_output = max(output_tokens or 0, 0)
            result.input_cost, result.output_cost = self._calculate_from_tokens(
                actual_input, actual_output, model
            )
            result.input_tokens = actual_input
            result.output_tokens = actual_output
            result.calculation_method = "token_counts"
        elif response:
            result.output_cost = self._safe_calculate_completion_cost(response, model)
            result.calculation_method = "response_only"

        result.total_cost = result.input_cost + result.output_cost

    def _safe_calculate_prompt_cost(
        self, prompt: str | list[dict[str, Any]], model: str
    ) -> float:
        """Calculate prompt cost and propagate pricing failures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            return calculate_prompt_cost(prompt, model)

    def _safe_calculate_completion_cost(self, response: str, model: str) -> float:
        """Calculate completion cost and propagate pricing failures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            return calculate_completion_cost(response, model)

    def _calculate_from_tokens(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> tuple[float, float]:
        """Calculate costs from token counts via cost_from_tokens().

        Delegates to the canonical cost_from_tokens() entry point.

        Raises:
            UnknownModelError: If model has no pricing. Callers MUST handle
                this — returning 0.0 would silently break budget enforcement.
        """
        return cost_from_tokens(input_tokens, output_tokens, model, strict=True)

    def get_available_models(self) -> list[str]:
        """Get list of all available litellm models (cached)."""
        if not LITELLM_AVAILABLE:
            return []
        return list(litellm.model_cost.keys())

    def clear_cache(self) -> None:
        """Clear internal pricing caches."""
        global _CUSTOM_PRICING_CACHE, _CUSTOM_PRICING_CACHE_KEY
        with _CUSTOM_PRICING_LOCK:
            _CUSTOM_PRICING_CACHE = None
            _CUSTOM_PRICING_CACHE_KEY = None

    def validate_model_name(self, model_name: str) -> dict[str, Any]:
        """Validate and provide information about a model name.

        This method does not perform fuzzy guessing or implicit family mapping.
        """
        normalized = (
            _normalize_model_name(model_name).lower() if model_name else model_name
        )
        resolved = (
            _resolve_litellm_alias(_resolve_builtin_model_alias(normalized))
            if normalized
            else None
        )
        result = {
            "original": model_name,
            "normalized": normalized,
            "mapped": None,
            "resolved": resolved,
            "available": LITELLM_AVAILABLE,
            "known_to_litellm": False,
            "custom_pricing": False,
            "builtin_pricing": False,
            "exact_match": False,
            "fuzzy_match": False,
            "family_match": False,
            "not_found": False,
        }

        if not model_name:
            result["not_found"] = True
            return result

        if not LITELLM_AVAILABLE:
            result["error"] = "litellm library not available"
            return result

        candidates = _build_model_candidates(model_name)
        for candidate in candidates:
            if _is_model_known_to_litellm(candidate):
                result["known_to_litellm"] = True
                result["mapped"] = candidate
                result["not_found"] = False
                return result

        try:
            custom_pricing = _try_custom_model_pricing(model_name)
        except ValueError as exc:
            result["error"] = str(exc)
            result["not_found"] = True
            return result

        if custom_pricing is not None:
            result["custom_pricing"] = True
            result["mapped"] = resolved or normalized or model_name
            result["not_found"] = False
            return result

        builtin_rates = _try_builtin_per_token_rates(candidates)
        if builtin_rates is not None:
            _input_cost, _output_cost, matched_key = builtin_rates
            result["builtin_pricing"] = True
            result["mapped"] = matched_key
            result["not_found"] = False
            return result

        result["not_found"] = True
        return result


# ---------------------------------------------------------------------------
# Canonical cost-from-tokens entry point
# ---------------------------------------------------------------------------


def _try_litellm_cost(
    candidates: list[str],
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float] | None:
    """Try litellm cost_per_token and model_cost lookup for each candidate.

    Returns:
        ``(input_cost_usd, output_cost_usd)`` or ``None`` if no pricing found.
    """
    # Try cost_per_token API first
    for candidate in candidates:
        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=candidate,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            if input_cost > 0 or output_cost > 0:
                return float(input_cost), float(output_cost)
            if _is_model_known_to_litellm(candidate):
                return float(input_cost), float(output_cost)
        except Exception:
            continue

    # Direct lookup in litellm.model_cost (case-insensitive)
    model_cost_lower = {k.lower(): k for k in litellm.model_cost}
    for candidate in candidates:
        actual_key = model_cost_lower.get(candidate.lower())
        if actual_key:
            cost_info = litellm.model_cost[actual_key]
            input_cpt = cost_info.get("input_cost_per_token", 0.0)
            output_cpt = cost_info.get("output_cost_per_token", 0.0)
            return (
                float(input_tokens * input_cpt),
                float(output_tokens * output_cpt),
            )

    return None


def cost_from_tokens(
    input_tokens: int,
    output_tokens: int,
    model: str,
    *,
    strict: bool = True,
) -> tuple[float, float]:
    """Canonical cost calculation from token counts.

    This is the single entry point for post-call cost tracking. It is
    fail-fast by default and does not guess pricing for unknown models.

    Args:
        input_tokens: Number of input tokens (0 is valid for output-only).
        output_tokens: Number of output tokens (0 is valid for input-only).
        model: Model identifier (with or without provider prefix).
        strict: If True (default), raise UnknownModelError for unpriced models.
                If False, log warning and return (0.0, 0.0).

    Returns:
        Tuple of (input_cost_usd, output_cost_usd).

    Raises:
        UnknownModelError: When strict=True and model has no pricing.
        RuntimeError: When strict=True and litellm is not installed.
        ValueError: When token counts are negative.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"Token counts must be non-negative, got "
            f"input={input_tokens}, output={output_tokens}"
        )

    if not LITELLM_AVAILABLE and strict:
        raise RuntimeError(
            "litellm is required for cost tracking but is not installed. "
            "Install it with: pip install litellm"
        )
    if not LITELLM_AVAILABLE:
        logger.warning("litellm not available — returning zero cost")
        return 0.0, 0.0

    candidates = _build_model_candidates(model)

    # Step 1-2: Try litellm pricing (cost_per_token API + model_cost dict)
    litellm_result = _try_litellm_cost(candidates, input_tokens, output_tokens)
    if litellm_result is not None:
        return litellm_result

    # Step 3: explicit custom pricing (user-provided)
    custom_pricing = _try_custom_model_pricing(model)
    if custom_pricing is not None:
        input_rate, output_rate = custom_pricing
        return (
            float(input_tokens * input_rate),
            float(output_tokens * output_rate),
        )

    # Step 4: Traigent built-in fallback pricing for supported models.
    builtin_rates = _try_builtin_per_token_rates(candidates)
    if builtin_rates is not None:
        input_rate, output_rate, _matched_key = builtin_rates
        return (
            float(input_tokens * input_rate),
            float(output_tokens * output_rate),
        )

    if strict:
        raise UnknownModelError(_unknown_model_resolution_message(model))

    logger.warning(
        "Unknown model %r — returning zero cost (strict=False). "
        "Configure %s or %s to provide explicit pricing.",
        model,
        _CUSTOM_PRICING_FILE_ENV,
        _CUSTOM_PRICING_JSON_ENV,
    )
    return 0.0, 0.0


def model_has_nonzero_price_coverage(model_name: str) -> bool:
    """Return whether a model resolves to non-zero pricing via the real path."""
    try:
        input_cost, output_cost = cost_from_tokens(1, 1, model_name, strict=True)
    except (RuntimeError, UnknownModelError):
        return False
    return (input_cost + output_cost) > 0.0


def find_models_missing_price_coverage(models: Iterable[str]) -> list[str]:
    """Return deduplicated model IDs without non-zero pricing coverage."""
    missing: list[str] = []
    seen: set[str] = set()
    for model_name in models:
        if model_name in seen:
            continue
        seen.add(model_name)
        if not model_has_nonzero_price_coverage(model_name):
            missing.append(model_name)
    return missing


# Convenience functions for simple usage
def calculate_llm_cost(
    prompt: str | list[dict[str, Any]] | None = None,
    response: str | None = None,
    model_name: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    logger=None,
) -> CostBreakdown:
    """Convenience function for quick cost calculation."""
    calculator = CostCalculator(logger=logger)
    return calculator.calculate_cost(
        prompt=prompt,
        response=response,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def validate_model_support(model_name: str, logger=None) -> dict[str, Any]:
    """Convenience function to validate model name support."""
    calculator = CostCalculator(logger=logger)
    return calculator.validate_model_name(model_name)


# Global instance for simple usage with thread safety
_global_calculator = None
_global_calculator_lock = threading.Lock()


def get_cost_calculator(logger=None) -> CostCalculator:
    """Get a global cost calculator instance (thread-safe)."""
    global _global_calculator
    if _global_calculator is None:
        with _global_calculator_lock:
            # Double-checked locking pattern
            if _global_calculator is None:
                _global_calculator = CostCalculator(logger=logger)
    return _global_calculator


def get_model_pricing_per_1k(model_name: str) -> tuple[float, float]:
    """Get model pricing rates in USD per 1K tokens.

    Returns a tuple ``(input_per_1k, output_per_1k)`` via the canonical cost
    pipeline (litellm first, then explicit custom pricing, then built-in
    fallback pricing). Unknown models return ``(0.0, 0.0)``.

    This is a query function (not budget-enforcement), so it uses
    strict=False to avoid raising on unknown models.
    """
    input_per_1k, _ = cost_from_tokens(1000, 0, model_name, strict=False)
    _, output_per_1k = cost_from_tokens(0, 1000, model_name, strict=False)
    return float(input_per_1k), float(output_per_1k)


def completion_cost(
    completion: str | None = None,
    model: str | None = None,
    *,
    output_tokens: int | None = None,
) -> float:
    """Output (completion) cost in USD via the canonical pricing pipeline.

    Unlike ``litellm.completion_cost``, this prices provider-prefixed ids that
    litellm cannot map directly (e.g. ``"openrouter/openai/gpt-4o-mini"``, which
    raises ``"This model isn't mapped yet"`` from litellm even though
    ``litellm.model_cost`` HAS the underlying ``"gpt-4o-mini"`` entry) by routing
    through :func:`_build_model_candidates` / :func:`cost_from_tokens`. (See #1423.)

    Provide either ``completion`` text (tokens are counted with litellm) or a
    pre-computed ``output_tokens`` count.

    Args:
        completion: Completion text whose tokens should be priced.
        model: Model identifier (with or without provider prefix).
        output_tokens: Pre-computed completion token count (overrides ``completion``).

    Returns:
        Cost in USD for the output tokens. ``0.0`` for an unpriced model.
    """
    if not model:
        return 0.0
    tokens = output_tokens
    if tokens is None:
        if completion is None:
            return 0.0
        tokens = (
            litellm.token_counter(model=model, text=completion)
            if LITELLM_AVAILABLE
            else max(1, len(completion) // 4)
        )
    _, out_cost = cost_from_tokens(0, max(int(tokens), 0), model, strict=False)
    return float(out_cost)


def prompt_cost(
    prompt: str | list[dict[str, Any]] | None = None,
    model: str | None = None,
    *,
    input_tokens: int | None = None,
) -> float:
    """Input (prompt) cost in USD via the canonical pricing pipeline.

    Counterpart to :func:`completion_cost`; prices provider-prefixed ids the same
    way. Provide either ``prompt`` text/messages or a pre-computed ``input_tokens``.

    Returns:
        Cost in USD for the input tokens. ``0.0`` for an unpriced model.
    """
    if not model:
        return 0.0
    tokens = input_tokens
    if tokens is None:
        if prompt is None:
            return 0.0
        if LITELLM_AVAILABLE:
            tokens = (
                litellm.token_counter(model=model, messages=prompt)
                if isinstance(prompt, list)
                else litellm.token_counter(model=model, text=prompt)
            )
        else:
            tokens = max(1, len(str(prompt)) // 4)
    in_cost, _ = cost_from_tokens(max(int(tokens), 0), 0, model, strict=False)
    return float(in_cost)


# ---------------------------------------------------------------------------
# Unpriced-at-runtime model registry (issue #1407)
# ---------------------------------------------------------------------------
#
# The pre-run cost-coverage preflight only inspects model ids declared in the
# configuration space; it cannot see a model HARD-CODED inside the user's
# @optimize function body. When such a model is unpriced, the non-strict runtime
# cost path records $0 with only a buried log, silently under-reporting spend.
# This registry lets the runtime cost path record those ids so the optimizer can
# surface them on the OptimizationResult (a user-visible warning), reusing the
# same remediation surface as the preflight. (Strict accounting still RAISES via
# ``cost_from_tokens(strict=True)`` — fail-closed — so the registry only carries
# the non-strict, warn-and-continue case.)
#
# Keyed by model id -> occurrence count (#1597: OpenRouter response-model ids
# litellm cannot price, e.g. a newly-released model not yet in litellm's bundled
# pricing map). The count distinguishes "one unlucky call" from a systematic
# per-trial pattern, and lets callers report how many $0.0 entries are actually
# *unknown* spend rather than verified-free, instead of only naming the model.
_unpriced_runtime_models: dict[str, int] = {}
_unpriced_runtime_lock = threading.Lock()


def record_unpriced_runtime_model(model: str) -> None:
    """Record an occurrence of a model id that priced to $0 at runtime despite non-zero tokens."""
    if not model or not str(model).strip():
        return
    key = str(model).strip()
    with _unpriced_runtime_lock:
        _unpriced_runtime_models[key] = _unpriced_runtime_models.get(key, 0) + 1


def get_unpriced_runtime_models() -> list[str]:
    """Return a snapshot of model ids that priced to $0 at runtime (sorted)."""
    with _unpriced_runtime_lock:
        return sorted(_unpriced_runtime_models)


def get_unpriced_runtime_occurrences() -> dict[str, int]:
    """Return a snapshot of ``{model_id: occurrence_count}`` for unpriced runtime calls.

    Complements :func:`get_unpriced_runtime_models` with quantitative detail
    (#1597) so a caller can tell "one $0 call for this model" apart from "every
    call for this model priced to $0" and report the recorded ``total_cost`` as
    a lower bound rather than verified spend.
    """
    with _unpriced_runtime_lock:
        return dict(sorted(_unpriced_runtime_models.items()))


def reset_unpriced_runtime_models() -> None:
    """Clear the unpriced-at-runtime model registry (call before a fresh run)."""
    with _unpriced_runtime_lock:
        _unpriced_runtime_models.clear()
