"""LLM cost calculation utilities.

Cost resolution is intentionally fail-fast:
- Runtime and pre-estimation pricing use litellm first.
- Unknown models raise with actionable remediation by default.
- Users can provide explicit custom pricing for private/unsupported models.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

import json
import logging
import os
import threading
import warnings
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# CRITICAL: Set offline mode BEFORE importing litellm to prevent network calls
# This must happen before any litellm import to use bundled pricing data
if os.environ.get("TRAIGENT_OFFLINE_MODE", "").lower() == "true":
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# Import litellm with graceful fallback
try:
    import litellm

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

# Backward-compat alias (external code may import the old name)
FALLBACK_MODEL_PRICING = ESTIMATION_MODEL_PRICING

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

    if input_cost < 0 or output_cost < 0:
        raise ValueError(f"Invalid negative pricing for model '{model}' in {source}")

    return input_cost, output_cost


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
                file_payload = json.load(f)
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
            env_payload = json.loads(env_json)
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
    prompt_cost, _ = litellm.cost_per_token(
        model=model, prompt_tokens=tokens, completion_tokens=0
    )
    if prompt_cost > 0:
        return float(prompt_cost), tokens
    if model_known:
        return 0.0, tokens
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
    _, completion_cost = litellm.cost_per_token(
        model=model, prompt_tokens=0, completion_tokens=tokens
    )
    if completion_cost > 0:
        return float(completion_cost), tokens
    if model_known:
        return 0.0, tokens
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


# Legacy model name aliases for fallback pricing lookup.
# Maps names not in ESTIMATION_MODEL_PRICING to their closest canonical equivalent.
_FALLBACK_ALIASES: dict[str, str] = {
    "claude-3-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-sonnet-20240229": "claude-3-5-sonnet-20241022",
}


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
        # Check if base_model starts with key or key starts with base_model
        if base_model.startswith(key_lower):
            if len(key_lower) > best_match_len:
                best_match_len = len(key_lower)
                pricing = model_pricing
                matched_key = model_key
        elif key_lower.startswith(base_model):
            if len(base_model) > best_match_len:
                best_match_len = len(base_model)
                pricing = model_pricing
                matched_key = model_key

    return pricing, matched_key


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

    # Resolve legacy aliases before lookup
    for alias, canonical in _FALLBACK_ALIASES.items():
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


# Backward-compat alias
_fallback_cost_from_tokens = _estimation_cost_from_tokens


def get_model_token_pricing(model_name: str) -> tuple[float, float, str]:
    """Get per-token pricing for a model with fail-fast behavior.

    Resolution order:
    1. litellm pricing database
    2. explicit custom pricing overrides (env/file)

    Unknown models raise ``UnknownModelError`` with remediation instructions.

    Args:
        model_name: Model identifier (may include provider prefix).

    Returns:
        ``(input_cost_per_token, output_cost_per_token, estimation_method)``
    """
    if not model_name or not model_name.strip():
        raise UnknownModelError(_unknown_model_resolution_message(model_name))

    normalized = _normalize_model_name(model_name)
    alias_resolved = _resolve_litellm_alias(normalized)
    candidates = [
        candidate for candidate in [model_name, normalized, alias_resolved] if candidate
    ]
    candidates = list(dict.fromkeys(candidates))

    if LITELLM_AVAILABLE:
        for candidate in candidates:
            try:
                input_cost, output_cost = litellm.cost_per_token(
                    model=candidate, prompt_tokens=1, completion_tokens=1
                )
                if input_cost > 0 or output_cost > 0:
                    return float(input_cost), float(output_cost), "litellm"
                if _is_model_known_to_litellm(candidate):
                    return float(input_cost), float(output_cost), "litellm"
            except Exception:
                logger.debug(
                    "litellm pricing lookup failed for %r", candidate, exc_info=True
                )

    custom_pricing = _try_custom_model_pricing(model_name)
    if custom_pricing is not None:
        input_cost, output_cost = custom_pricing
        return float(input_cost), float(output_cost), "custom_pricing"

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

        normalized_model = _normalize_model_name(model_name)
        effective_model = _resolve_litellm_alias(normalized_model)

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
            return calculate_prompt_cost(prompt, model)

    def _safe_calculate_completion_cost(self, response: str, model: str) -> float:
        """Calculate completion cost and propagate pricing failures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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
        normalized = _normalize_model_name(model_name) if model_name else model_name
        resolved = _resolve_litellm_alias(normalized) if model_name else None
        result = {
            "original": model_name,
            "normalized": normalized,
            "mapped": None,
            "resolved": resolved,
            "available": LITELLM_AVAILABLE,
            "known_to_litellm": False,
            "custom_pricing": False,
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

        candidates = [
            candidate for candidate in [model_name, normalized, resolved] if candidate
        ]
        candidates = list(dict.fromkeys(candidates))
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

        result["not_found"] = True
        return result


# ---------------------------------------------------------------------------
# Canonical cost-from-tokens entry point
# ---------------------------------------------------------------------------


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

    if not LITELLM_AVAILABLE:
        if strict:
            raise RuntimeError(
                "litellm is required for cost tracking but is not installed. "
                "Install it with: pip install litellm"
            )
        logger.warning("litellm not available — returning zero cost")
        return 0.0, 0.0

    normalized = _normalize_model_name(model)
    alias_resolved = _resolve_litellm_alias(normalized)
    candidates = [
        candidate for candidate in [model, normalized, alias_resolved] if candidate
    ]
    candidates = list(dict.fromkeys(candidates))

    # Step 1: Try litellm.cost_per_token (handles provider prefixes internally)
    for candidate in candidates:
        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=candidate,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            if input_cost > 0 or output_cost > 0:
                return float(input_cost), float(output_cost)
            # Model is known but returns (0, 0) — legitimate free tier
            if _is_model_known_to_litellm(candidate):
                return float(input_cost), float(output_cost)
        except Exception:
            continue

    # Step 2: Try direct lookup in litellm.model_cost (case-insensitive)
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

    # Step 3: explicit custom pricing (user-provided)
    custom_pricing = _try_custom_model_pricing(model)
    if custom_pricing is not None:
        input_rate, output_rate = custom_pricing
        return (
            float(input_tokens * input_rate),
            float(output_tokens * output_rate),
        )

    # Model not found in litellm or custom pricing config
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
    pipeline (litellm first, then explicit custom pricing). Unknown models
    return ``(0.0, 0.0)``.

    This is a query function (not budget-enforcement), so it uses
    strict=False to avoid raising on unknown models.
    """
    input_per_1k, _ = cost_from_tokens(1000, 0, model_name, strict=False)
    _, output_per_1k = cost_from_tokens(0, 1000, model_name, strict=False)
    return float(input_per_1k), float(output_per_1k)
