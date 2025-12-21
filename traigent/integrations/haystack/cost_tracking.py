"""Cost tracking for Haystack pipeline evaluations.

This module provides token usage extraction and cost calculation for
Haystack pipelines, integrating with Traigent's existing cost infrastructure.

Example usage:
    from traigent.integrations.haystack.cost_tracking import (
        HaystackCostTracker,
        extract_token_usage,
    )

    # Extract tokens from pipeline output
    tokens = extract_token_usage(pipeline_output)
    print(f"Input: {tokens.input_tokens}, Output: {tokens.output_tokens}")

    # Calculate costs using existing infrastructure
    tracker = HaystackCostTracker(model="gpt-4o")
    cost = tracker.calculate_cost(tokens)
    print(f"Cost: ${cost.total_cost:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from traigent.utils.cost_calculator import CostCalculator, get_cost_calculator
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage from a pipeline execution.

    Attributes:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total tokens (computed if not provided).
        model: Model name if available.
        component: Component name that generated the tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str | None = None
    component: str | None = None

    def __post_init__(self) -> None:
        """Compute total if not provided."""
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two TokenUsage instances together."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model=self.model or other.model,
            component=None,  # Aggregated, no single component
        )


@dataclass
class CostResult:
    """Cost calculation result.

    Attributes:
        input_cost: Cost for input tokens in USD.
        output_cost: Cost for output tokens in USD.
        total_cost: Total cost in USD.
        tokens: Token usage that led to this cost.
        model_used: Model name used for pricing.
    """

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    model_used: str | None = None

    def __add__(self, other: CostResult) -> CostResult:
        """Add two CostResult instances together."""
        return CostResult(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost,
            tokens=self.tokens + other.tokens,
            model_used=self.model_used or other.model_used,
        )


def extract_token_usage(
    output: dict[str, Any] | None,
    component: str | None = None,
) -> TokenUsage:
    """Extract token usage from Haystack pipeline output.

    Supports multiple output formats:
    - OpenAI style: {"usage": {"prompt_tokens": N, "completion_tokens": M}}
    - Anthropic style: {"usage": {"input_tokens": N, "output_tokens": M}}
    - Haystack meta: {"meta": [{"usage": {...}, "model": "..."}]}

    Args:
        output: Pipeline output dictionary.
        component: Optional component name for attribution.

    Returns:
        TokenUsage with extracted values (zeros if not found).
    """
    if output is None:
        return TokenUsage(component=component)

    # Try to find token usage in various locations
    usage_data = None
    model = None

    # Check for direct usage field
    if "usage" in output and isinstance(output["usage"], dict):
        usage_data = output["usage"]

    # Check for meta field (Haystack generators)
    elif "meta" in output and isinstance(output["meta"], list) and output["meta"]:
        meta = output["meta"][0] if output["meta"] else {}
        if isinstance(meta, dict):
            usage_data = meta.get("usage", {})
            model = meta.get("model")

    # Check nested in component outputs (e.g., {"llm": {"meta": [...]}})
    else:
        for _key, value in output.items():
            if isinstance(value, dict):
                if "meta" in value and isinstance(value["meta"], list):
                    meta_list = value["meta"]
                    if meta_list and isinstance(meta_list[0], dict):
                        usage_data = meta_list[0].get("usage", {})
                        model = meta_list[0].get("model")
                        break
                elif "usage" in value and isinstance(value["usage"], dict):
                    usage_data = value["usage"]
                    break

    if not usage_data:
        return TokenUsage(component=component, model=model)

    # Extract tokens (support both OpenAI and Anthropic naming)
    # Use explicit None check to preserve explicit 0 values
    input_tokens = usage_data.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage_data.get("prompt_tokens", 0)

    output_tokens = usage_data.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage_data.get("completion_tokens", 0)

    total_tokens = usage_data.get("total_tokens", 0)

    return TokenUsage(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        total_tokens=int(total_tokens),
        model=model,
        component=component,
    )


def extract_tokens_from_results(
    example_results: list[Any],
    component_prefix: str = "",
) -> list[TokenUsage]:
    """Extract token usage from a list of example results.

    Args:
        example_results: List of ExampleResult objects from execute_with_config.
        component_prefix: Optional prefix for component names.

    Returns:
        List of TokenUsage for each example.
    """
    usages = []
    for i, result in enumerate(example_results):
        output = getattr(result, "output", None)
        component = f"{component_prefix}example_{i}" if component_prefix else None
        usage = extract_token_usage(output, component=component)
        usages.append(usage)
    return usages


class HaystackCostTracker:
    """Cost tracker for Haystack pipeline evaluations.

    Uses Traigent's existing CostCalculator for pricing calculations.
    Aggregates costs across multiple examples and components.

    Attributes:
        model: Default model name for cost calculation.
        calculator: The underlying CostCalculator instance.
    """

    def __init__(
        self,
        model: str | None = None,
        calculator: CostCalculator | None = None,
    ) -> None:
        """Initialize the cost tracker.

        Args:
            model: Default model name (e.g., "gpt-4o").
            calculator: Optional CostCalculator instance.
                Uses singleton if not provided.
        """
        self.model = model
        self._calculator = calculator

    @property
    def calculator(self) -> CostCalculator:
        """Get the cost calculator (lazy initialization)."""
        if self._calculator is None:
            self._calculator = get_cost_calculator()
        return self._calculator

    def calculate_cost(
        self,
        tokens: TokenUsage,
        model: str | None = None,
    ) -> CostResult:
        """Calculate cost for given token usage.

        Args:
            tokens: Token usage to calculate cost for.
            model: Model name override (uses tokens.model or self.model if None).

        Returns:
            CostResult with calculated costs.
        """
        # Determine model to use
        model_name = model or tokens.model or self.model

        if not model_name:
            logger.debug("No model name available for cost calculation")
            return CostResult(tokens=tokens)

        if tokens.input_tokens == 0 and tokens.output_tokens == 0:
            return CostResult(tokens=tokens, model_used=model_name)

        try:
            # Use calculate_cost with token counts (not prompt text)
            breakdown = self.calculator.calculate_cost(
                model_name=model_name,
                input_tokens=tokens.input_tokens,
                output_tokens=tokens.output_tokens,
            )

            return CostResult(
                input_cost=breakdown.input_cost,
                output_cost=breakdown.output_cost,
                total_cost=breakdown.total_cost,
                tokens=tokens,
                model_used=breakdown.model_used,
            )
        except Exception as e:
            logger.debug(f"Cost calculation failed for model '{model_name}': {e}")
            return CostResult(tokens=tokens, model_used=model_name)

    def calculate_total_cost(
        self,
        token_usages: list[TokenUsage],
        model: str | None = None,
    ) -> CostResult:
        """Calculate total cost across multiple token usages.

        Args:
            token_usages: List of TokenUsage to aggregate.
            model: Default model name if not in individual usages.

        Returns:
            Aggregated CostResult.
        """
        total = CostResult()
        for usage in token_usages:
            cost = self.calculate_cost(usage, model=model)
            total = total + cost
        return total

    def extract_and_calculate(
        self,
        outputs: list[dict[str, Any] | None],
        model: str | None = None,
    ) -> CostResult:
        """Extract tokens from outputs and calculate total cost.

        Convenience method that combines extraction and calculation.

        Args:
            outputs: List of pipeline output dictionaries.
            model: Default model name for pricing.

        Returns:
            Aggregated CostResult for all outputs.
        """
        usages = [extract_token_usage(output) for output in outputs]
        return self.calculate_total_cost(usages, model=model)


def get_cost_metrics(cost_result: CostResult) -> dict[str, float]:
    """Convert CostResult to metrics dict for EvaluationResult.

    The returned dict is compatible with extract_cost_from_results()
    in traigent/core/orchestrator_helpers.py.

    Args:
        cost_result: Cost calculation result.

    Returns:
        Dict with cost metrics ready for aggregated_metrics.
    """
    return {
        "total_cost": cost_result.total_cost,
        "input_cost": cost_result.input_cost,
        "output_cost": cost_result.output_cost,
        "input_tokens": cost_result.tokens.input_tokens,
        "output_tokens": cost_result.tokens.output_tokens,
        "total_tokens": cost_result.tokens.total_tokens,
    }
