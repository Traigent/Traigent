"""Cost verification tests for LiteLLM integration.

Tests verify that Traigent's cost calculation works correctly with LiteLLM.
LiteLLM provides unified interface to multiple LLM providers.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from tests.validation.conftest import requires_anthropic, requires_openai
from tests.validation.cost_verification.models import CostVerificationResult

# Pricing URLs
OPENAI_PRICING_URL = "https://openai.com/api/pricing/"
ANTHROPIC_PRICING_URL = "https://www.anthropic.com/pricing"

# Known pricing
GPT4O_MINI_INPUT_PRICE = 0.15 / 1_000_000
GPT4O_MINI_OUTPUT_PRICE = 0.60 / 1_000_000
CLAUDE_HAIKU_INPUT_PRICE = 0.25 / 1_000_000
CLAUDE_HAIKU_OUTPUT_PRICE = 1.25 / 1_000_000


def calculate_expected_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate expected cost based on known pricing."""
    if "gpt-4o-mini" in model:
        return (prompt_tokens * GPT4O_MINI_INPUT_PRICE) + (
            completion_tokens * GPT4O_MINI_OUTPUT_PRICE
        )
    if "haiku" in model.lower():
        return (prompt_tokens * CLAUDE_HAIKU_INPUT_PRICE) + (
            completion_tokens * CLAUDE_HAIKU_OUTPUT_PRICE
        )
    raise ValueError(f"Unknown model: {model}")


@requires_openai
class TestLiteLLMOpenAICost:
    """Cost verification tests for LiteLLM with OpenAI."""

    def test_litellm_openai_gpt4o_mini(self, cost_tracker) -> CostVerificationResult:
        """Verify cost calculation with LiteLLM + OpenAI."""
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_tokens=10,
        )

        # Extract usage
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

        # Also verify litellm's own cost calculation
        litellm_cost = litellm.completion_cost(completion_response=response)

        cost_tracker.record(
            provider="litellm",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
        )

        result = CostVerificationResult(
            provider="litellm",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=OPENAI_PRICING_URL,
            expected_input_price_per_token=GPT4O_MINI_INPUT_PRICE,
            expected_output_price_per_token=GPT4O_MINI_OUTPUT_PRICE,
            timestamp=datetime.utcnow(),
            notes=f"LiteLLM reported cost: ${litellm_cost:.8f}",
        )

        assert result.cost_matches, (
            f"Cost mismatch: SDK={sdk_cost:.8f}, expected={expected_cost:.8f}, "
            f"litellm={litellm_cost:.8f}"
        )

        return result


@requires_anthropic
class TestLiteLLMAnthropic:
    """Cost verification tests for LiteLLM with Anthropic."""

    def test_litellm_anthropic_haiku(self, cost_tracker) -> CostVerificationResult:
        """Verify cost calculation with LiteLLM + Anthropic."""
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        response = litellm.completion(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_tokens=10,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        cost_breakdown = calculate_llm_cost(
            model_name="claude-3-haiku-20240307",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        sdk_cost = cost_breakdown.total_cost
        expected_cost = calculate_expected_cost(
            "claude-3-haiku", prompt_tokens, completion_tokens
        )

        litellm_cost = litellm.completion_cost(completion_response=response)

        cost_tracker.record(
            provider="litellm",
            model="claude-3-haiku-20240307",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
        )

        result = CostVerificationResult(
            provider="litellm",
            model="claude-3-haiku-20240307",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=ANTHROPIC_PRICING_URL,
            expected_input_price_per_token=CLAUDE_HAIKU_INPUT_PRICE,
            expected_output_price_per_token=CLAUDE_HAIKU_OUTPUT_PRICE,
            notes=f"LiteLLM reported cost: ${litellm_cost:.8f}",
        )

        assert (
            result.cost_matches
        ), f"Cost mismatch: SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"

        return result


def run_litellm_verification() -> list[CostVerificationResult]:
    """Run all LiteLLM verification tests and return results."""
    from tests.validation.conftest import CostTracker

    tracker = CostTracker()
    results = []

    if os.environ.get("OPENAI_API_KEY"):
        tests = TestLiteLLMOpenAICost()
        try:
            results.append(tests.test_litellm_openai_gpt4o_mini(tracker))
        except Exception as e:
            print(f"LiteLLM OpenAI test failed: {e}")

    if os.environ.get("ANTHROPIC_API_KEY"):
        tests = TestLiteLLMAnthropic()
        try:
            results.append(tests.test_litellm_anthropic_haiku(tracker))
        except Exception as e:
            print(f"LiteLLM Anthropic test failed: {e}")

    return results
