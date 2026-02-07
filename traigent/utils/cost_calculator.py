"""
Intelligent LLM Cost Calculator with Fuzzy Model Matching

This module provides comprehensive cost calculation for LLM API calls with:
- Exact model name mapping for litellm compatibility
- Fuzzy matching with substring search and semantic validation
- Intelligent date/version parsing and selection
- Caching for performance optimization
- Graceful fallback with informative logging

Usage:
    from traigent.utils.cost_calculator import CostCalculator

    calculator = CostCalculator()
    cost = calculator.calculate_cost(
        prompt="Your prompt here",
        response="Model response",
        model_name="claude-3-haiku"
    )

This module can be copied to other repositories for consistent cost calculation.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

import logging
import os
import re
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

# Fallback pricing for models not in litellm's database (per-token costs)
FALLBACK_MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input_cost_per_token": 2.5e-6, "output_cost_per_token": 10.0e-6},
    "gpt-4o-mini": {"input_cost_per_token": 0.15e-6, "output_cost_per_token": 0.6e-6},
    "gpt-4-turbo": {"input_cost_per_token": 10.0e-6, "output_cost_per_token": 30.0e-6},
    "gpt-3.5-turbo": {"input_cost_per_token": 0.5e-6, "output_cost_per_token": 1.5e-6},
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

# Model scoring constants for fuzzy matching preference
# Used in _calculate_model_score to prioritize model selection
LATEST_MODEL_PRIORITY = (9999, 12, 31, 99)  # "latest" models get highest priority
VERSION_FALLBACK_DATE = (
    2024,
    1,
    1,
)  # Base date for version-numbered models (e.g., v1, v2)
UNVERSIONED_FALLBACK_DATE = (2020, 1, 1)  # Base date for unversioned models
DATE_MATCH_PRIORITY = 50  # Priority value for date-matched models

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
        the tag stripped. For Ollama models, use the full model name directly
        or add to FALLBACK_MODEL_PRICING for pricing lookup.

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


def calculate_prompt_cost(
    prompt: str | list[dict[str, Any]] | None, model: str
) -> float:
    """Calculate INPUT cost only using litellm.

    Backward-compatible wrapper for tokencost's calculate_prompt_cost.

    Args:
        prompt: The prompt text or message list (may be None)
        model: The model name

    Returns:
        Cost in USD for the input tokens

    Raises:
        UnknownModelError: If the model's pricing cannot be determined from
            litellm or the fallback pricing table.
    """
    tokens = 0
    model_known = False

    if LITELLM_AVAILABLE:
        model_known = _is_model_known_to_litellm(model)
        try:
            # Count tokens first (guard against None prompt)
            if prompt is not None:
                if isinstance(prompt, list):
                    tokens = litellm.token_counter(model=model, messages=prompt)
                else:
                    tokens = litellm.token_counter(model=model, text=prompt)

            # Get cost for input tokens only (0 completion tokens)
            prompt_cost, _ = litellm.cost_per_token(
                model=model, prompt_tokens=tokens, completion_tokens=0
            )
            if prompt_cost > 0:
                return float(prompt_cost)
            # Zero cost from litellm - if model is known, allow 0.0
            if model_known:
                return 0.0
        except Exception:
            # litellm failed - log for diagnostics and fall through to fallback
            logger.debug(
                "litellm cost calculation failed for model %r", model, exc_info=True
            )
            pass

    # Estimate tokens from prompt if we don't have a count
    if tokens == 0 and prompt is not None:
        # ~3 chars per token (conservative for non-English/code)
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        tokens = max(1, len(prompt_str) // 3)

    # Default to 10 tokens if we still don't have an estimate
    estimated_tokens = tokens if tokens > 0 else 10

    # Try fallback pricing
    fallback_cost = _fallback_cost_from_tokens(model, estimated_tokens, 0)[0]
    if fallback_cost > 0:
        return fallback_cost

    # Unknown model - raise exception (mimics tokencost behavior)
    raise UnknownModelError(
        f"Model '{model}' is not in litellm's pricing database or fallback table. "
        "Add it to FALLBACK_MODEL_PRICING or use a known model."
    )


def calculate_completion_cost(completion: str | None, model: str) -> float:
    """Calculate OUTPUT cost only using litellm.

    Backward-compatible wrapper for tokencost's calculate_completion_cost.

    Args:
        completion: The completion text (may be None)
        model: The model name

    Returns:
        Cost in USD for the output tokens

    Raises:
        UnknownModelError: If the model's pricing cannot be determined from
            litellm or the fallback pricing table.
    """
    tokens = 0
    model_known = False

    if LITELLM_AVAILABLE:
        model_known = _is_model_known_to_litellm(model)
        try:
            # Count tokens (guard against None completion)
            if completion is not None:
                tokens = litellm.token_counter(model=model, text=completion)

            # Get cost for output tokens only (0 prompt tokens)
            _, completion_cost = litellm.cost_per_token(
                model=model, prompt_tokens=0, completion_tokens=tokens
            )
            if completion_cost > 0:
                return float(completion_cost)
            # Zero cost from litellm - if model is known, allow 0.0
            if model_known:
                return 0.0
        except Exception:
            # litellm failed - log for diagnostics and fall through to fallback
            logger.debug(
                "litellm cost calculation failed for model %r", model, exc_info=True
            )
            pass

    # Estimate tokens from completion if we don't have a count
    if tokens == 0 and completion is not None:
        # ~3 chars per token (conservative for non-English/code)
        tokens = max(1, len(completion) // 3)

    # Default to 10 tokens if we still don't have an estimate
    estimated_tokens = tokens if tokens > 0 else 10

    # Try fallback pricing
    fallback_cost = _fallback_cost_from_tokens(model, 0, estimated_tokens)[1]
    if fallback_cost > 0:
        return fallback_cost

    # Unknown model - raise exception (mimics tokencost behavior)
    raise UnknownModelError(
        f"Model '{model}' is not in litellm's pricing database or fallback table. "
        "Add it to FALLBACK_MODEL_PRICING or use a known model."
    )


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


def _fallback_cost_from_tokens(
    model: str, input_tokens: int, output_tokens: int, *, _quiet: bool = False
) -> tuple[float, float]:
    """Calculate costs using fallback pricing dictionary.

    Args:
        model: Model name (may include provider prefix)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Tuple of (input_cost, output_cost). Returns (0.0, 0.0) if model not found.
    """
    base_model = _normalize_model_for_fallback(model)

    # Try exact match first (case-insensitive via normalized base_model)
    pricing = None
    matched_key = None
    for key, value in FALLBACK_MODEL_PRICING.items():
        if key.lower() == base_model:
            pricing = value
            matched_key = key
            break

    # Try prefix matching - prefer LONGEST matching key
    if not pricing:
        best_match_len = 0
        for model_key, model_pricing in FALLBACK_MODEL_PRICING.items():
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

    if pricing:
        # Warn-once per model to avoid log spam
        with _warned_models_lock:
            if model not in _warned_models:
                _warned_models.add(model)
                log_fn = logger.debug if _quiet else logger.warning
                log_fn(
                    "Using fallback pricing for model %r (matched %r)",
                    model,
                    matched_key,
                )
        input_cost = input_tokens * pricing["input_cost_per_token"]
        output_cost = output_tokens * pricing["output_cost_per_token"]
        return input_cost, output_cost

    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Heuristic tier pricing for pre-optimization cost estimation
# ---------------------------------------------------------------------------

# Per-token costs for each pricing tier (conservative estimates)
_TIER_EXPENSIVE = {"input": 10.0e-6, "output": 30.0e-6}  # ~GPT-4-turbo class
_TIER_MID = {"input": 3.0e-6, "output": 15.0e-6}  # ~GPT-4o / Sonnet class
_TIER_CHEAP = {"input": 0.25e-6, "output": 1.25e-6}  # ~Haiku / Mini class

# Ordered regex rules for model tier classification.
# Order is critical: most specific patterns first to avoid substring collisions.
# \b treats `-` as a word boundary, so "gpt-4o-mini" matches r"gpt-4o\b".
# We rely on ordering (gpt-4o-mini checked BEFORE gpt-4o) to handle this.
_TIER_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gpt-4o-mini|gpt-4-mini", re.IGNORECASE), "cheap"),
    (re.compile(r"gpt-4-turbo", re.IGNORECASE), "expensive"),
    (re.compile(r"gpt-4o\b", re.IGNORECASE), "mid"),
    (re.compile(r"gpt-4\b(?!o)", re.IGNORECASE), "expensive"),
    (re.compile(r"gpt-3\.5", re.IGNORECASE), "cheap"),
    (re.compile(r"opus", re.IGNORECASE), "expensive"),
    (re.compile(r"sonnet", re.IGNORECASE), "mid"),
    (re.compile(r"haiku", re.IGNORECASE), "cheap"),
    (re.compile(r"\bmini|flash|nano", re.IGNORECASE), "cheap"),
    (re.compile(r"pro\b", re.IGNORECASE), "mid"),
]

_TIER_PRICING = {
    "expensive": _TIER_EXPENSIVE,
    "mid": _TIER_MID,
    "cheap": _TIER_CHEAP,
}


def _classify_model_tier(model_name: str) -> str:
    """Classify a model name into a pricing tier using regex matching.

    Returns one of ``"expensive"``, ``"mid"``, or ``"cheap"``.
    Falls back to ``"mid"`` (conservative) when no pattern matches.
    """
    for pattern, tier in _TIER_RULES:
        if pattern.search(model_name):
            return tier
    return "mid"


def get_model_token_pricing(model_name: str) -> tuple[float, float, str]:
    """Get per-token pricing for a model, with graceful fallback.

    Uses a 3-tier lookup chain for conservative pre-optimization cost estimation:

    1. **litellm**: exact per-token pricing from litellm's database.
       Models that return ``(0, 0)`` are NOT treated as free — falls through
       to heuristic tier (EMA will correct after the first trial).
    2. **Fallback dict**: prefix-matching against ``FALLBACK_MODEL_PRICING``.
       Does NOT apply ``EXACT_MODEL_MAPPING`` — intentionally skipped so that
       e.g. ``gpt-4`` stays in the EXPENSIVE tier rather than being downgraded
       to ``gpt-4o`` (MID tier). The mapping is correct for runtime cost
       calculation but wrong for conservative pre-estimation.
    3. **Heuristic tier**: regex-based classification into EXPENSIVE / MID / CHEAP.

    Args:
        model_name: Model identifier (may include provider prefix).

    Returns:
        ``(input_cost_per_token, output_cost_per_token, estimation_method)``
    """
    # --- Tier 1: litellm ---
    if LITELLM_AVAILABLE and _is_model_known_to_litellm(model_name):
        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=model_name, prompt_tokens=1, completion_tokens=1
            )
            if input_cost > 0 or output_cost > 0:
                logger.debug(
                    "Model pricing from litellm for %r: input=%.2e, output=%.2e",
                    model_name,
                    input_cost,
                    output_cost,
                )
                return float(input_cost), float(output_cost), "litellm"
            # litellm returned (0, 0) — may lack pricing data. Fall through.
        except Exception:
            logger.debug(
                "litellm pricing lookup failed for %r, trying fallback",
                model_name,
                exc_info=True,
            )

    # --- Tier 2: Fallback dict (prefix matching, no EXACT_MODEL_MAPPING) ---
    normalized = _normalize_model_for_fallback(model_name)
    # Use _quiet=True to log at DEBUG during estimation (not WARNING)
    input_cost_fb, output_cost_fb = _fallback_cost_from_tokens(
        normalized, 1, 1, _quiet=True
    )
    if input_cost_fb > 0 or output_cost_fb > 0:
        logger.debug(
            "Model pricing from fallback dict for %r: input=%.2e, output=%.2e",
            model_name,
            input_cost_fb,
            output_cost_fb,
        )
        return input_cost_fb, output_cost_fb, "fallback_dict"

    # --- Tier 3: Heuristic tier classification ---
    tier = _classify_model_tier(model_name)
    pricing = _TIER_PRICING[tier]
    logger.debug(
        "Model pricing from heuristic tier %r for %r: input=%.2e, output=%.2e",
        tier,
        model_name,
        pricing["input"],
        pricing["output"],
    )
    return pricing["input"], pricing["output"], f"heuristic:{tier}"


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
        if self.total_cost == 0.0:
            self.total_cost = self.input_cost + self.output_cost
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class CostCalculator:
    """Intelligent LLM cost calculator with fuzzy model matching."""

    # Exact model name mappings (high confidence)
    EXACT_MODEL_MAPPING = {
        # Claude models - map user-friendly names to litellm expected names
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-4-sonnet": "claude-sonnet-4-20250514",
        "claude-4-opus": "claude-opus-4-20250514",
        # Add latest versions for convenience
        "claude-haiku": "claude-3-haiku-latest",
        "claude-sonnet": "claude-3-5-sonnet-latest",
        "claude-opus": "claude-3-opus-latest",
        # OpenAI models are usually fine as-is but add common aliases
        "gpt-3.5": "gpt-3.5-turbo",
        "gpt-4": "gpt-4o",
        "gpt4": "gpt-4o",
        "gpt3.5": "gpt-3.5-turbo",
        "gpt-4o-mini": "gpt-4o-mini",  # GPT-4o mini model for cost-effective usage
        # Common alternative spellings
        "claude3-haiku": "claude-3-haiku-20240307",
        "claude3-sonnet": "claude-3-sonnet-20240229",
        "claude3-opus": "claude-3-opus-20240229",
    }

    # Model family defaults for generic names
    FAMILY_DEFAULTS = {
        "claude-3": "claude-3-5-sonnet-latest",  # Most capable general model
        "claude": "claude-3-5-sonnet-latest",
        "gpt-4": "gpt-4o",  # Latest GPT-4 variant
        "gpt": "gpt-4o",  # Default to latest
        "gpt-3.5": "gpt-3.5-turbo",  # Standard 3.5 model
    }

    def __init__(self, logger=None, enable_caching: bool = True) -> None:
        """Initialize the cost calculator.

        Args:
            logger: Optional logger instance for debug/warning messages
            enable_caching: Whether to cache fuzzy match results for performance
        """
        self.logger = logger
        self.enable_caching = enable_caching
        self._fuzzy_match_cache: dict[str, str | None] = {}

        # Validate litellm availability
        if not LITELLM_AVAILABLE and logger:
            logger.warning("litellm not available, cost calculations will return 0")

    def calculate_cost(
        self,
        prompt: str | list[dict[str, Any]] | None = None,
        response: str | None = None,
        model_name: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> CostBreakdown:
        """Calculate cost for an LLM request with multiple fallback methods.

        Args:
            prompt: The input prompt (string or message list)
            response: The model response text
            model_name: The model identifier
            input_tokens: Optional pre-calculated input token count
            output_tokens: Optional pre-calculated output token count

        Returns:
            CostBreakdown with detailed cost information
        """
        if not LITELLM_AVAILABLE:
            return CostBreakdown(
                model_used=model_name or "unknown",
                calculation_method="litellm_unavailable",
            )

        if not model_name:
            return CostBreakdown(calculation_method="no_model_name")

        # Map model name to litellm-compatible format
        mapped_model = self._map_model_name(model_name)

        # Even if mapping fails, try using the original model name with litellm
        # litellm.cost_per_token handles provider prefixes internally
        effective_model = mapped_model or model_name

        result = CostBreakdown(
            model_used=model_name, mapped_model=effective_model or ""
        )

        try:
            # Method 1: Use original prompt and response for most accurate calculation
            if prompt and response:
                result.input_cost = self._safe_calculate_prompt_cost(
                    prompt, effective_model
                )
                result.output_cost = self._safe_calculate_completion_cost(
                    response, effective_model
                )
                result.calculation_method = "prompt_and_response"

            # Method 2: Use token counts if available
            elif (
                input_tokens
                and output_tokens
                and input_tokens > 0
                and output_tokens > 0
            ):
                result.input_cost, result.output_cost = self._calculate_from_tokens(
                    input_tokens, output_tokens, effective_model
                )
                result.input_tokens = input_tokens
                result.output_tokens = output_tokens
                result.calculation_method = "token_counts"

            # Method 3: Response only
            elif response:
                result.output_cost = self._safe_calculate_completion_cost(
                    response, effective_model
                )
                result.calculation_method = "response_only"

            result.total_cost = result.input_cost + result.output_cost

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Cost calculation failed for model {model_name}: {e}"
                )
            result.calculation_method = f"error_{type(e).__name__}"

        return result

    def _normalize_model_name(self, model_name: str) -> str:
        """Strip provider prefixes like 'openai/gpt-4o' -> 'gpt-4o'."""
        if "/" in model_name:
            # Handle nested prefixes like 'openrouter/openai/gpt-4o'
            return model_name.split("/")[-1]
        return model_name

    def _map_model_name(self, model_name: str) -> str | None:
        """Map user-friendly model names to litellm-compatible names with fuzzy matching."""
        if not model_name:
            return None

        # Step 1: Try exact mapping
        exact_match = self.EXACT_MODEL_MAPPING.get(model_name)
        if exact_match:
            if self.logger:
                self.logger.debug(
                    f"Exact model mapping: '{model_name}' -> '{exact_match}'"
                )
            return exact_match

        # Step 2: Try with normalized name (strip provider prefix)
        normalized = self._normalize_model_name(model_name)
        if normalized != model_name:
            exact_match = self.EXACT_MODEL_MAPPING.get(normalized)
            if exact_match:
                if self.logger:
                    self.logger.debug(
                        f"Exact model mapping (normalized): '{model_name}' -> '{exact_match}'"
                    )
                return exact_match

        # Step 3: Try family defaults for generic names
        family_match = self.FAMILY_DEFAULTS.get(model_name)
        if family_match:
            if self.logger:
                self.logger.debug(
                    f"Family model mapping: '{model_name}' -> '{family_match}'"
                )
            return family_match

        # Step 4: Try fuzzy matching
        return self._fuzzy_match_model(model_name)

    def _fuzzy_match_model(self, user_model: str) -> str | None:
        """Find best matching litellm model using fuzzy matching."""
        # Check cache first
        if self.enable_caching and user_model in self._fuzzy_match_cache:
            cached_result = self._fuzzy_match_cache[user_model]
            if self.logger and cached_result:
                self.logger.debug(
                    f"Cached fuzzy match: '{user_model}' -> '{cached_result}'"
                )
            return cached_result

        # Normalize model name for checking
        normalized = self._normalize_model_name(user_model)

        # First check if the model exists as-is in litellm
        if LITELLM_AVAILABLE and normalized in litellm.model_cost:
            if self.logger:
                self.logger.debug(
                    f"Direct model match: '{user_model}' found in litellm"
                )
            if self.enable_caching:
                self._fuzzy_match_cache[user_model] = normalized
            return normalized

        result = self._perform_fuzzy_match(user_model)

        # Cache the result
        if self.enable_caching:
            self._fuzzy_match_cache[user_model] = result

        return result

    def _perform_fuzzy_match(self, user_model: str) -> str | None:
        """Perform actual fuzzy matching algorithm."""
        if not LITELLM_AVAILABLE or len(user_model) < 5:
            return None

        # Get all available litellm models
        available_models = list(litellm.model_cost.keys())

        # Normalize the user model
        normalized_user = self._normalize_model_name(user_model)

        # Find substring matches
        matches = []
        user_lower = normalized_user.lower()

        for litellm_model in available_models:
            # Normalize litellm model name too
            normalized_litellm = self._normalize_model_name(litellm_model)
            if self._is_semantic_match(user_lower, normalized_litellm.lower()):
                matches.append(litellm_model)

        if not matches:
            if self.logger:
                self.logger.debug(f"No fuzzy matches found for '{user_model}'")
            return None

        if len(matches) == 1:
            result = matches[0]
            if self.logger:
                self.logger.info(f"Fuzzy match found: '{user_model}' -> '{result}'")
            return str(result)

        # Multiple matches - select the latest/best one
        best_match = self._select_best_model(matches, user_model)
        if self.logger:
            self.logger.info(
                f"Best fuzzy match: '{user_model}' -> '{best_match}' (from {len(matches)} options)"
            )

        return best_match

    def _is_semantic_match(self, user_model: str, litellm_model: str) -> bool:
        """Check if a litellm model is a meaningful semantic match for user model."""
        # Basic substring matching
        if user_model not in litellm_model:
            return False

        # Avoid over-broad matches
        if len(user_model) < 5:
            return False

        # Special case: ensure model family alignment
        if "gpt" in user_model and "gpt" not in litellm_model:
            return False
        if "claude" in user_model and "claude" not in litellm_model:
            return False

        # Avoid mixing different model generations inappropriately
        if "gpt-4" in user_model and "gpt-3" in litellm_model:
            return False
        if "gpt-3" in user_model and "gpt-4" in litellm_model:
            return False

        # Check for model version mismatches that don't make sense
        if "claude-3" in user_model and "claude-2" in litellm_model:
            return False

        return True

    def _select_best_model(self, matches: list[str], user_model: str) -> str:
        """Select the best model from multiple matches using date/version sorting."""
        if not matches:
            return ""

        if len(matches) == 1:
            return matches[0]

        # Sort by preference: latest > dated versions > version numbers > plain names
        scored_matches = []
        for match in matches:
            score = self._calculate_model_score(match, user_model)
            scored_matches.append((score, match))

        # Sort by score (highest first) and return the best match
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        return scored_matches[0][1]

    def _calculate_model_score(
        self, model_name: str, user_model: str
    ) -> tuple[int, int, int, int]:
        """Calculate a sortable score for model preference."""
        suffix = model_name.lower().replace(user_model.lower(), "")

        # "latest" gets highest priority
        if "latest" in suffix:
            return LATEST_MODEL_PRIORITY

        # Parse YYYYMMDD format
        date_match = re.search(r"(\d{4})(\d{2})(\d{2})", suffix)
        if date_match:
            year, month, day = map(int, date_match.groups())
            return (year, month, day, DATE_MATCH_PRIORITY)

        # Parse YYYY-MM-DD format
        date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", suffix)
        if date_match:
            year, month, day = map(int, date_match.groups())
            return (year, month, day, DATE_MATCH_PRIORITY)

        # Parse version numbers (v1, v2, etc.)
        version_match = re.search(r"v(\d+)", suffix)
        if version_match:
            version = int(version_match.group(1))
            return (*VERSION_FALLBACK_DATE, version)

        # Prefer shorter model names (more specific)
        specificity = 100 - len(model_name)
        return (*UNVERSIONED_FALLBACK_DATE, specificity)

    def _safe_calculate_prompt_cost(
        self, prompt: str | list[dict[str, Any]], model: str
    ) -> float:
        """Safely calculate prompt cost with error handling."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return calculate_prompt_cost(prompt, model)
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Prompt cost calculation failed: {e}")
            return 0.0

    def _safe_calculate_completion_cost(self, response: str, model: str) -> float:
        """Safely calculate completion cost with error handling."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return calculate_completion_cost(response, model)
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Completion cost calculation failed: {e}")
            return 0.0

    def _calculate_from_tokens(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> tuple[float, float]:
        """Calculate costs from token counts using direct pricing information."""
        try:
            # First, try litellm.cost_per_token directly (handles provider prefixes)
            if LITELLM_AVAILABLE:
                try:
                    input_cost, output_cost = litellm.cost_per_token(
                        model=model,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )

                    if self.logger:
                        self.logger.debug(
                            f"litellm token cost calculation for {model}: "
                            f"input={input_tokens}=${input_cost:.8f}, "
                            f"output={output_tokens}=${output_cost:.8f}"
                        )

                    return float(input_cost), float(output_cost)
                except Exception:
                    # Model not in litellm, try direct lookup or fallback
                    pass

            # Try direct lookup in litellm.model_cost
            normalized = self._normalize_model_name(model)
            if LITELLM_AVAILABLE and normalized in litellm.model_cost:
                cost_info = litellm.model_cost[normalized]

                # Get cost per token (not per 1000 tokens)
                input_cost_per_token = cost_info.get("input_cost_per_token", 0.0)
                output_cost_per_token = cost_info.get("output_cost_per_token", 0.0)

                # Calculate actual costs
                input_cost = input_tokens * input_cost_per_token
                output_cost = output_tokens * output_cost_per_token

                if self.logger:
                    self.logger.debug(
                        f"Direct token cost calculation for {model}: "
                        f"input={input_tokens}*{input_cost_per_token:.10f}=${input_cost:.8f}, "
                        f"output={output_tokens}*{output_cost_per_token:.10f}=${output_cost:.8f}"
                    )

                return input_cost, output_cost

            # Fallback to FALLBACK_MODEL_PRICING
            return _fallback_cost_from_tokens(model, input_tokens, output_tokens)

        except Exception as e:
            if self.logger:
                self.logger.debug(f"Token-based cost calculation failed: {e}")
            return 0.0, 0.0

    def get_available_models(self) -> list[str]:
        """Get list of all available litellm models (cached)."""
        if not LITELLM_AVAILABLE:
            return []
        return list(litellm.model_cost.keys())

    def clear_cache(self) -> None:
        """Clear the fuzzy match cache."""
        self._fuzzy_match_cache.clear()
        # Note: get_available_models is not cached, so no need to clear

    def validate_model_name(self, model_name: str) -> dict[str, Any]:
        """Validate and provide information about a model name."""
        result = {
            "original": model_name,
            "mapped": None,
            "available": LITELLM_AVAILABLE,
            "exact_match": False,
            "fuzzy_match": False,
            "family_match": False,
            "not_found": False,
        }

        if not LITELLM_AVAILABLE:
            result["error"] = "litellm library not available"
            return result

        # Check exact mapping
        if model_name in self.EXACT_MODEL_MAPPING:
            result["mapped"] = self.EXACT_MODEL_MAPPING[model_name]
            result["exact_match"] = True
            return result

        # Check family mapping
        if model_name in self.FAMILY_DEFAULTS:
            result["mapped"] = self.FAMILY_DEFAULTS[model_name]
            result["family_match"] = True
            return result

        # Try fuzzy matching
        fuzzy_result = self._perform_fuzzy_match(model_name)
        if fuzzy_result:
            result["mapped"] = fuzzy_result
            result["fuzzy_match"] = True
            return result

        result["not_found"] = True
        return result


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
