"""Cost verification tests for Anthropic SDK integration.

Tests verify that Traigent's cost calculation matches Anthropic's official pricing.
Uses claude-3-haiku for cost efficiency (~$0.01 per test).
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from tests.validation.conftest import requires_anthropic
from tests.validation.cost_verification.models import CostVerificationResult

# Official Anthropic pricing page
PRICING_URL = "https://www.anthropic.com/pricing"

# Pricing as of 2026-01 (per 1M tokens)
# claude-3-haiku: $0.25/1M input, $1.25/1M output
CLAUDE_HAIKU_INPUT_PRICE = 0.25 / 1_000_000  # per token
CLAUDE_HAIKU_OUTPUT_PRICE = 1.25 / 1_000_000  # per token


def calculate_expected_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate expected cost based on known pricing."""
    if "haiku" in model.lower():
        return (prompt_tokens * CLAUDE_HAIKU_INPUT_PRICE) + (
            completion_tokens * CLAUDE_HAIKU_OUTPUT_PRICE
        )
    raise ValueError(f"Unknown model: {model}")


@requires_anthropic
class TestAnthropicSDKCost:
    """Cost verification tests for Anthropic SDK."""

    def test_claude_haiku_simple_query(self, cost_tracker) -> CostVerificationResult:
        """Verify claude-3-haiku cost calculation with a simple query."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        client = Anthropic()

        # Simple query to minimize tokens
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
        )

        # Extract usage
        usage = response.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens

        # Calculate costs
        cost_breakdown = calculate_llm_cost(
            model_name="claude-3-haiku-20240307",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "claude-3-haiku", prompt_tokens, completion_tokens
        )

        # Record for tracking
        cost_tracker.record(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
            raw_response=(
                response.model_dump() if hasattr(response, "model_dump") else {}
            ),
        )

        result = CostVerificationResult(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=CLAUDE_HAIKU_INPUT_PRICE,
            expected_output_price_per_token=CLAUDE_HAIKU_OUTPUT_PRICE,
            timestamp=datetime.utcnow(),
            raw_response=(
                response.model_dump() if hasattr(response, "model_dump") else {}
            ),
        )

        # Assert cost matches
        assert result.cost_matches, (
            f"Cost mismatch for claude-3-haiku: "
            f"SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"
        )

        return result

    def test_claude_haiku_longer_response(self, cost_tracker) -> CostVerificationResult:
        """Verify cost calculation with more output tokens."""
        try:
            from anthropic import Anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        client = Anthropic()

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=30,
            messages=[
                {
                    "role": "user",
                    "content": "List 3 primary colors, one per line, no explanation.",
                }
            ],
        )

        usage = response.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens

        cost_breakdown = calculate_llm_cost(
            model_name="claude-3-haiku-20240307",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "claude-3-haiku", prompt_tokens, completion_tokens
        )

        cost_tracker.record(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
        )

        result = CostVerificationResult(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=CLAUDE_HAIKU_INPUT_PRICE,
            expected_output_price_per_token=CLAUDE_HAIKU_OUTPUT_PRICE,
        )

        assert (
            result.cost_matches
        ), f"Cost mismatch: SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"

        return result


def run_anthropic_verification() -> list[CostVerificationResult]:
    """Run all Anthropic verification tests and return results."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return []

    from tests.validation.conftest import CostTracker

    tracker = CostTracker()
    tests = TestAnthropicSDKCost()
    results = []

    try:
        results.append(tests.test_claude_haiku_simple_query(tracker))
    except Exception as e:
        print(f"Anthropic simple query test failed: {e}")

    return results
