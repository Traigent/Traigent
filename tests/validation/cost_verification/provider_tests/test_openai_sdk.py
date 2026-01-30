"""Cost verification tests for OpenAI SDK integration.

Tests verify that Traigent's cost calculation matches OpenAI's official pricing.
Uses gpt-4o-mini for cost efficiency (~$0.01 per test).
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from tests.validation.conftest import requires_openai
from tests.validation.cost_verification.models import CostVerificationResult

# Official OpenAI pricing page
PRICING_URL = "https://openai.com/api/pricing/"

# Pricing as of 2026-01 (per 1M tokens)
# gpt-4o-mini: $0.15/1M input, $0.60/1M output
GPT4O_MINI_INPUT_PRICE = 0.15 / 1_000_000  # per token
GPT4O_MINI_OUTPUT_PRICE = 0.60 / 1_000_000  # per token


def calculate_expected_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate expected cost based on known pricing."""
    if model == "gpt-4o-mini":
        return (prompt_tokens * GPT4O_MINI_INPUT_PRICE) + (
            completion_tokens * GPT4O_MINI_OUTPUT_PRICE
        )
    raise ValueError(f"Unknown model: {model}")


@requires_openai
class TestOpenAISDKCost:
    """Cost verification tests for OpenAI SDK."""

    def test_gpt4o_mini_simple_query(self, cost_tracker) -> CostVerificationResult:
        """Verify gpt-4o-mini cost calculation with a simple query."""
        try:
            from openai import OpenAI
        except ImportError:
            pytest.skip("openai package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        client = OpenAI()

        # Simple query to minimize tokens
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_tokens=10,
        )

        # Extract usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        # Calculate costs
        cost_breakdown = calculate_llm_cost(
            model_name="gpt-4o-mini",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "gpt-4o-mini", prompt_tokens, completion_tokens
        )

        # Record for tracking
        cost_tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
            raw_response=(
                response.model_dump() if hasattr(response, "model_dump") else {}
            ),
        )

        result = CostVerificationResult(
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=GPT4O_MINI_INPUT_PRICE,
            expected_output_price_per_token=GPT4O_MINI_OUTPUT_PRICE,
            timestamp=datetime.utcnow(),
            raw_response=(
                response.model_dump() if hasattr(response, "model_dump") else {}
            ),
        )

        # Assert cost matches
        assert result.cost_matches, (
            f"Cost mismatch for gpt-4o-mini: "
            f"SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"
        )

        return result

    def test_gpt4o_mini_longer_response(self, cost_tracker) -> CostVerificationResult:
        """Verify cost calculation with more output tokens."""
        try:
            from openai import OpenAI
        except ImportError:
            pytest.skip("openai package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        client = OpenAI()

        # Query that generates more output
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "List 3 primary colors, one per line, no explanation.",
                }
            ],
            max_tokens=30,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        cost_breakdown = calculate_llm_cost(
            model_name="gpt-4o-mini",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "gpt-4o-mini", prompt_tokens, completion_tokens
        )

        cost_tracker.record(
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
        )

        result = CostVerificationResult(
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=GPT4O_MINI_INPUT_PRICE,
            expected_output_price_per_token=GPT4O_MINI_OUTPUT_PRICE,
        )

        assert (
            result.cost_matches
        ), f"Cost mismatch: SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"

        return result


def run_openai_verification() -> list[CostVerificationResult]:
    """Run all OpenAI verification tests and return results."""
    if not os.environ.get("OPENAI_API_KEY"):
        return []

    from tests.validation.conftest import CostTracker

    tracker = CostTracker()
    tests = TestOpenAISDKCost()
    results = []

    try:
        results.append(tests.test_gpt4o_mini_simple_query(tracker))
    except Exception as e:
        print(f"OpenAI simple query test failed: {e}")

    return results
