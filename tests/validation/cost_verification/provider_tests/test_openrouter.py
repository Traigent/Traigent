"""Cost verification tests for OpenRouter integration.

Tests verify that Traigent's cost calculation works correctly with OpenRouter.
Uses mistral-7b for cost efficiency (~$0.001 per test).
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from tests.validation.conftest import requires_openrouter
from tests.validation.cost_verification.models import CostVerificationResult

# Official OpenRouter pricing page
PRICING_URL = "https://openrouter.ai/models"

# Pricing varies by model on OpenRouter
# mistral-7b-instruct: approximately $0.07/1M input, $0.07/1M output
MISTRAL_7B_INPUT_PRICE = 0.07 / 1_000_000
MISTRAL_7B_OUTPUT_PRICE = 0.07 / 1_000_000


def calculate_expected_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate expected cost based on known pricing."""
    if "mistral-7b" in model.lower():
        return (prompt_tokens * MISTRAL_7B_INPUT_PRICE) + (
            completion_tokens * MISTRAL_7B_OUTPUT_PRICE
        )
    raise ValueError(f"Unknown model: {model}")


@requires_openrouter
class TestOpenRouterCost:
    """Cost verification tests for OpenRouter."""

    def test_openrouter_mistral_simple_query(
        self, cost_tracker
    ) -> CostVerificationResult:
        """Verify mistral-7b cost calculation with a simple query."""
        try:
            from openai import OpenAI
        except ImportError:
            pytest.skip("openai package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        # OpenRouter uses OpenAI-compatible API
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_tokens=10,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        cost_breakdown = calculate_llm_cost(
            model_name="mistralai/mistral-7b-instruct",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "mistral-7b", prompt_tokens, completion_tokens
        )

        cost_tracker.record(
            provider="openrouter",
            model="mistralai/mistral-7b-instruct",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
            raw_response=(
                response.model_dump() if hasattr(response, "model_dump") else {}
            ),
        )

        result = CostVerificationResult(
            provider="openrouter",
            model="mistralai/mistral-7b-instruct",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=MISTRAL_7B_INPUT_PRICE,
            expected_output_price_per_token=MISTRAL_7B_OUTPUT_PRICE,
            timestamp=datetime.utcnow(),
            raw_response=(
                response.model_dump() if hasattr(response, "model_dump") else {}
            ),
            tolerance=0.05,  # OpenRouter pricing can vary, use looser tolerance (5%)
        )

        # Use the result's cost_matches property which respects tolerance
        matches = result.cost_matches
        cost_diff = abs(sdk_cost - expected_cost)

        if not matches:
            pct_diff = (cost_diff / expected_cost * 100) if expected_cost > 0 else 0
            result.notes = (
                f"Cost difference: {cost_diff:.8f} "
                f"({pct_diff:.2f}% - may reflect OpenRouter pricing)"
            )

        assert matches, (
            f"Cost mismatch for mistral-7b on OpenRouter: "
            f"SDK={sdk_cost:.8f}, expected={expected_cost:.8f} (diff={cost_diff:.8f})"
        )

        return result

    def test_openrouter_via_litellm(self, cost_tracker) -> CostVerificationResult:
        """Verify cost calculation with OpenRouter via LiteLLM."""
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        response = litellm.completion(
            model="openrouter/mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_tokens=10,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        cost_breakdown = calculate_llm_cost(
            model_name="openrouter/mistralai/mistral-7b-instruct",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "mistral-7b", prompt_tokens, completion_tokens
        )

        cost_tracker.record(
            provider="openrouter",
            model="openrouter/mistralai/mistral-7b-instruct",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
        )

        result = CostVerificationResult(
            provider="openrouter",
            model="openrouter/mistralai/mistral-7b-instruct",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=MISTRAL_7B_INPUT_PRICE,
            expected_output_price_per_token=MISTRAL_7B_OUTPUT_PRICE,
            notes="Via LiteLLM with openrouter/ prefix",
            tolerance=0.05,  # OpenRouter pricing can vary, use looser tolerance (5%)
        )

        # Use the result's cost_matches property which respects tolerance
        matches = result.cost_matches
        cost_diff = abs(sdk_cost - expected_cost)

        if not matches:
            result.notes += f" - Cost difference: {cost_diff:.8f}"

        assert (
            matches
        ), f"Cost mismatch: SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"

        return result


def run_openrouter_verification() -> list[CostVerificationResult]:
    """Run all OpenRouter verification tests and return results."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        return []

    from tests.validation.conftest import CostTracker

    tracker = CostTracker()
    tests = TestOpenRouterCost()
    results = []

    try:
        results.append(tests.test_openrouter_mistral_simple_query(tracker))
    except Exception as e:
        print(f"OpenRouter simple query test failed: {e}")

    return results
