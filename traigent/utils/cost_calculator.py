"""
Intelligent LLM Cost Calculator with Fuzzy Model Matching

This module provides comprehensive cost calculation for LLM API calls with:
- Exact model name mapping for tokencost compatibility
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

import re
import threading
import warnings
from dataclasses import dataclass
from typing import Any

# Import tokencost with graceful fallback
try:
    import tokencost
    from tokencost import calculate_completion_cost, calculate_prompt_cost

    TOKENCOST_AVAILABLE = True
except ImportError:
    tokencost = None
    calculate_prompt_cost = None
    calculate_completion_cost = None
    TOKENCOST_AVAILABLE = False

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
        # Claude models - map user-friendly names to tokencost expected names
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

        # Validate tokencost availability
        if not TOKENCOST_AVAILABLE and logger:
            logger.warning("tokencost not available, cost calculations will return 0")

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
        if not TOKENCOST_AVAILABLE:
            return CostBreakdown(
                model_used=model_name or "unknown",
                calculation_method="tokencost_unavailable",
            )

        if not model_name:
            return CostBreakdown(calculation_method="no_model_name")

        # Map model name to tokencost-compatible format
        mapped_model = self._map_model_name(model_name)

        if not mapped_model:
            if self.logger:
                self.logger.warning(
                    f"Model '{model_name}' not found in tokencost, returning $0.00 cost"
                )
            return CostBreakdown(
                model_used=model_name, calculation_method="model_not_found"
            )

        result = CostBreakdown(model_used=model_name, mapped_model=mapped_model)

        try:
            # Method 1: Use original prompt and response for most accurate calculation
            if prompt and response:
                result.input_cost = self._safe_calculate_prompt_cost(
                    prompt, mapped_model
                )
                result.output_cost = self._safe_calculate_completion_cost(
                    response, mapped_model
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
                    input_tokens, output_tokens, mapped_model
                )
                result.input_tokens = input_tokens
                result.output_tokens = output_tokens
                result.calculation_method = "token_counts"

            # Method 3: Response only
            elif response:
                result.output_cost = self._safe_calculate_completion_cost(
                    response, mapped_model
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

    def _map_model_name(self, model_name: str) -> str | None:
        """Map user-friendly model names to tokencost-compatible names with fuzzy matching."""
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

        # Step 2: Try family defaults for generic names
        family_match = self.FAMILY_DEFAULTS.get(model_name)
        if family_match:
            if self.logger:
                self.logger.debug(
                    f"Family model mapping: '{model_name}' -> '{family_match}'"
                )
            return family_match

        # Step 3: Try fuzzy matching
        return self._fuzzy_match_model(model_name)

    def _fuzzy_match_model(self, user_model: str) -> str | None:
        """Find best matching tokencost model using fuzzy matching."""
        # Check cache first
        if self.enable_caching and user_model in self._fuzzy_match_cache:
            cached_result = self._fuzzy_match_cache[user_model]
            if self.logger and cached_result:
                self.logger.debug(
                    f"Cached fuzzy match: '{user_model}' -> '{cached_result}'"
                )
            return cached_result

        # First check if the model exists as-is in tokencost
        if TOKENCOST_AVAILABLE and user_model in tokencost.TOKEN_COSTS:
            if self.logger:
                self.logger.debug(
                    f"Direct model match: '{user_model}' found in tokencost"
                )
            if self.enable_caching:
                self._fuzzy_match_cache[user_model] = user_model
            return user_model

        result = self._perform_fuzzy_match(user_model)

        # Cache the result
        if self.enable_caching:
            self._fuzzy_match_cache[user_model] = result

        return result

    def _perform_fuzzy_match(self, user_model: str) -> str | None:
        """Perform actual fuzzy matching algorithm."""
        if not TOKENCOST_AVAILABLE or len(user_model) < 5:
            return None

        # Get all available tokencost models
        available_models = list(tokencost.TOKEN_COSTS.keys())

        # Find substring matches
        matches = []
        user_lower = user_model.lower()

        for tokencost_model in available_models:
            if self._is_semantic_match(user_lower, tokencost_model.lower()):
                matches.append(tokencost_model)

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

    def _is_semantic_match(self, user_model: str, tokencost_model: str) -> bool:
        """Check if a tokencost model is a meaningful semantic match for user model."""
        # Basic substring matching
        if user_model not in tokencost_model:
            return False

        # Avoid over-broad matches
        if len(user_model) < 5:
            return False

        # Special case: ensure model family alignment
        if "gpt" in user_model and "gpt" not in tokencost_model:
            return False
        if "claude" in user_model and "claude" not in tokencost_model:
            return False

        # Avoid mixing different model generations inappropriately
        if "gpt-4" in user_model and "gpt-3" in tokencost_model:
            return False
        if "gpt-3" in user_model and "gpt-4" in tokencost_model:
            return False

        # Check for model version mismatches that don't make sense
        if "claude-3" in user_model and "claude-2" in tokencost_model:
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
                return float(calculate_prompt_cost(prompt, model))
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Prompt cost calculation failed: {e}")
            return 0.0

    def _safe_calculate_completion_cost(self, response: str, model: str) -> float:
        """Safely calculate completion cost with error handling."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return float(calculate_completion_cost(response, model))
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Completion cost calculation failed: {e}")
            return 0.0

    def _calculate_from_tokens(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> tuple[float, float]:
        """Calculate costs from token counts using direct pricing information."""
        try:
            # First, try to get pricing directly from tokencost
            if TOKENCOST_AVAILABLE and model in tokencost.TOKEN_COSTS:
                cost_info = tokencost.TOKEN_COSTS[model]

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

            # Fallback: try to estimate using tokencost functions
            # Get cost rates by calculating cost for minimal content
            single_char_prompt = [{"role": "user", "content": "a"}]
            single_char_response = "a"

            input_rate = self._safe_calculate_prompt_cost(single_char_prompt, model)
            output_rate = self._safe_calculate_completion_cost(
                single_char_response, model
            )

            # Estimate costs based on actual token counts
            # This is approximate since we're using single character rates
            input_cost = input_rate * input_tokens if input_rate > 0 else 0.0
            output_cost = output_rate * output_tokens if output_rate > 0 else 0.0

            return input_cost, output_cost

        except Exception as e:
            if self.logger:
                self.logger.debug(f"Token-based cost calculation failed: {e}")
            return 0.0, 0.0

    def get_available_models(self) -> list[str]:
        """Get list of all available tokencost models (cached)."""
        if not TOKENCOST_AVAILABLE:
            return []
        return list(tokencost.TOKEN_COSTS.keys())

    def clear_cache(self) -> None:
        """Clear the fuzzy match cache."""
        self._fuzzy_match_cache.clear()
        # Note: get_available_models is not cached, so no need to clear

    def validate_model_name(self, model_name: str) -> dict[str, Any]:
        """Validate and provide information about a model name."""
        result = {
            "original": model_name,
            "mapped": None,
            "available": TOKENCOST_AVAILABLE,
            "exact_match": False,
            "fuzzy_match": False,
            "family_match": False,
            "not_found": False,
        }

        if not TOKENCOST_AVAILABLE:
            result["error"] = "tokencost library not available"
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
