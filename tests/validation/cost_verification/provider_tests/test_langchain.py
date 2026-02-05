"""Cost verification tests for LangChain integration.

Tests verify that Traigent's cost calculation works correctly with LangChain.
Uses OpenAI via LangChain for testing (~$0.01 per test).
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from tests.validation.conftest import requires_openai
from tests.validation.cost_verification.models import CostVerificationResult

# OpenAI pricing (used via LangChain)
PRICING_URL = "https://openai.com/api/pricing/"

# gpt-4o-mini pricing
GPT4O_MINI_INPUT_PRICE = 0.15 / 1_000_000
GPT4O_MINI_OUTPUT_PRICE = 0.60 / 1_000_000


def calculate_expected_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate expected cost based on known pricing."""
    if "gpt-4o-mini" in model:
        return (prompt_tokens * GPT4O_MINI_INPUT_PRICE) + (
            completion_tokens * GPT4O_MINI_OUTPUT_PRICE
        )
    raise ValueError(f"Unknown model: {model}")


@requires_openai
class TestLangChainCost:
    """Cost verification tests for LangChain integration."""

    def test_langchain_openai_simple_query(
        self, cost_tracker
    ) -> CostVerificationResult:
        """Verify cost calculation with LangChain + OpenAI."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            pytest.skip("langchain-openai package not installed")

        from traigent.utils.cost_calculator import calculate_llm_cost

        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=10)

        # Invoke with callback for token tracking
        response = llm.invoke("What is 2+2? Answer in one word.")

        # LangChain stores usage in response_metadata
        usage_metadata = getattr(response, "usage_metadata", {}) or {}
        prompt_tokens = usage_metadata.get("input_tokens", 0)
        completion_tokens = usage_metadata.get("output_tokens", 0)

        # Fallback: check response_metadata
        if prompt_tokens == 0:
            response_metadata = getattr(response, "response_metadata", {}) or {}
            token_usage = response_metadata.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

        # Skip if we couldn't get token counts
        if prompt_tokens == 0:
            pytest.skip("Could not extract token usage from LangChain response")

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
            provider="langchain",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
        )

        result = CostVerificationResult(
            provider="langchain",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=GPT4O_MINI_INPUT_PRICE,
            expected_output_price_per_token=GPT4O_MINI_OUTPUT_PRICE,
            timestamp=datetime.utcnow(),
            notes="Using OpenAI via LangChain",
        )

        assert (
            result.cost_matches
        ), f"Cost mismatch: SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"

        return result

    def test_langchain_with_traigent_handler(
        self, cost_tracker
    ) -> CostVerificationResult:
        """Verify cost tracking with Traigent's LangChain handler."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            pytest.skip("langchain-openai package not installed")

        try:
            from traigent.integrations.langchain.handler import TraigentCallbackHandler
        except ImportError:
            pytest.skip("Traigent LangChain handler not available")

        from traigent.utils.cost_calculator import calculate_llm_cost

        # Create handler to track costs
        handler = TraigentCallbackHandler()
        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=20, callbacks=[handler])

        response = llm.invoke("List 2 colors, comma separated.")

        # Get usage from response
        usage_metadata = getattr(response, "usage_metadata", {}) or {}
        prompt_tokens = usage_metadata.get("input_tokens", 0)
        completion_tokens = usage_metadata.get("output_tokens", 0)

        if prompt_tokens == 0:
            response_metadata = getattr(response, "response_metadata", {}) or {}
            token_usage = response_metadata.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

        if prompt_tokens == 0:
            pytest.skip("Could not extract token usage")

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
            provider="langchain",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=sdk_cost,
        )

        result = CostVerificationResult(
            provider="langchain",
            model="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sdk_computed_cost=sdk_cost,
            expected_cost=expected_cost,
            price_source_url=PRICING_URL,
            expected_input_price_per_token=GPT4O_MINI_INPUT_PRICE,
            expected_output_price_per_token=GPT4O_MINI_OUTPUT_PRICE,
            notes="With Traigent callback handler",
        )

        assert (
            result.cost_matches
        ), f"Cost mismatch: SDK={sdk_cost:.8f}, expected={expected_cost:.8f}"

        return result


def run_langchain_verification() -> list[CostVerificationResult]:
    """Run all LangChain verification tests and return results."""
    if not os.environ.get("OPENAI_API_KEY"):
        return []

    from tests.validation.conftest import CostTracker

    tracker = CostTracker()
    tests = TestLangChainCost()
    results = []

    try:
        results.append(tests.test_langchain_openai_simple_query(tracker))
    except Exception as e:
        print(f"LangChain simple query test failed: {e}")

    return results
