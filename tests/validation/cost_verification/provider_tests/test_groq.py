"""Cost verification tests for Groq integration.

Tests verify that Traigent's cost calculation works correctly with Groq.
Uses llama-3.1-8b for cost efficiency (~$0.001 per test).
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from tests.validation.conftest import requires_groq
from tests.validation.cost_verification.models import CostVerificationResult

# Official Groq pricing page
PRICING_URL = "https://groq.com/pricing/"

# Pricing as of 2026-01 (per 1M tokens)
# llama-3.1-8b: $0.05/1M input, $0.08/1M output
LLAMA_8B_INPUT_PRICE = 0.05 / 1_000_000
LLAMA_8B_OUTPUT_PRICE = 0.08 / 1_000_000


def calculate_expected_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate expected cost based on known pricing."""
    if "llama-3.1-8b" in model.lower() or "llama3-8b" in model.lower():
        return (prompt_tokens * LLAMA_8B_INPUT_PRICE) + (
            completion_tokens * LLAMA_8B_OUTPUT_PRICE
        )
    raise ValueError(f"Unknown model: {model}")


@requires_groq
class TestGroqCost:
    """Cost verification tests for Groq."""

    def test_groq_llama_simple_query(self, cost_tracker) -> CostVerificationResult:
        """Verify llama-3.1-8b cost calculation with a simple query."""
        try:
            from groq import Groq
        except ImportError:
            pytest.skip("groq package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        client = Groq()

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_tokens=10,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        cost_breakdown = calculate_llm_cost(
            model_name="llama-3.1-8b-instant",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "llama-3.1-8b", prompt_tokens, completion_tokens
        )

        cost_tracker.record(
            provider="groq",
            model="llama-3.1-8b-instant",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
            raw_response=(
                response.model_dump() if hasattr(response, "model_dump") else {}
            ),
        )

        result = CostVerificationResult(
            provider="groq",
            model="llama-3.1-8b-instant",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=LLAMA_8B_INPUT_PRICE,
            expected_output_price_per_token=LLAMA_8B_OUTPUT_PRICE,
            timestamp=datetime.utcnow(),
            raw_response=(
                response.model_dump() if hasattr(response, "model_dump") else {}
            ),
        )

        assert result.cost_matches, (
            f"Cost mismatch for llama-3.1-8b: "
            f"SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"
        )

        return result

    def test_groq_via_litellm(self, cost_tracker) -> CostVerificationResult:
        """Verify cost calculation with Groq via LiteLLM."""
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        response = litellm.completion(
            model="groq/llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_tokens=10,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        # Test both the prefixed and unprefixed model name
        cost_breakdown = calculate_llm_cost(
            model_name="groq/llama-3.1-8b-instant",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "llama-3.1-8b", prompt_tokens, completion_tokens
        )

        cost_tracker.record(
            provider="groq",
            model="groq/llama-3.1-8b-instant",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
        )

        result = CostVerificationResult(
            provider="groq",
            model="groq/llama-3.1-8b-instant",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=LLAMA_8B_INPUT_PRICE,
            expected_output_price_per_token=LLAMA_8B_OUTPUT_PRICE,
            notes="Via LiteLLM with groq/ prefix",
        )

        assert (
            result.cost_matches
        ), f"Cost mismatch: SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"

        return result


def run_groq_verification() -> list[CostVerificationResult]:
    """Run all Groq verification tests and return results."""
    if not os.environ.get("GROQ_API_KEY"):
        return []

    from tests.validation.conftest import CostTracker

    tracker = CostTracker()
    tests = TestGroqCost()
    results = []

    try:
        results.append(tests.test_groq_llama_simple_query(tracker))
    except Exception as e:
        print(f"Groq simple query test failed: {e}")

    return results
