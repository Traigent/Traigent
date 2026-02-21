"""Unit tests for cost_calculator public behaviors."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import traigent.utils.cost_calculator as cc
from traigent.utils.cost_calculator import (
    CostBreakdown,
    CostCalculator,
    UnknownModelError,
    calculate_llm_cost,
    get_cost_calculator,
    get_model_pricing_per_1k,
    validate_model_support,
)


class TestCostBreakdown:
    def test_defaults(self) -> None:
        breakdown = CostBreakdown()
        assert breakdown.total_cost == 0.0
        assert breakdown.total_tokens == 0

    def test_post_init_computes_totals(self) -> None:
        breakdown = CostBreakdown(
            input_cost=0.1, output_cost=0.2, input_tokens=10, output_tokens=5
        )
        assert breakdown.total_cost == pytest.approx(0.3)
        assert breakdown.total_tokens == 15


class TestCostCalculatorInit:
    def test_init_default(self) -> None:
        calculator = CostCalculator()
        assert calculator.logger is None
        assert calculator.enable_caching is True

    @patch("traigent.utils.cost_calculator.LITELLM_AVAILABLE", False)
    def test_init_requires_litellm(self) -> None:
        with pytest.raises(RuntimeError, match="litellm is required"):
            CostCalculator()


class TestCostCalculatorCalculateCost:
    def test_no_model_name_returns_empty_breakdown(self) -> None:
        calculator = CostCalculator()
        result = calculator.calculate_cost(prompt="hi", response="hello", model_name=None)
        assert result.calculation_method == "no_model_name"
        assert result.total_cost == 0.0

    def test_token_counts_path_for_known_model(self) -> None:
        calculator = CostCalculator()
        result = calculator.calculate_cost(
            model_name="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        assert result.calculation_method == "token_counts"
        assert result.total_cost > 0
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_token_counts_path_unknown_model_raises(self) -> None:
        calculator = CostCalculator()
        with pytest.raises(UnknownModelError, match="has no known pricing"):
            calculator.calculate_cost(
                model_name="unknown-pricing-model-x",
                input_tokens=100,
                output_tokens=50,
            )

    def test_prompt_response_path_known_model(self) -> None:
        calculator = CostCalculator()
        result = calculator.calculate_cost(
            prompt="hello",
            response="world",
            model_name="gpt-4o-mini",
        )
        assert result.calculation_method == "prompt_and_response"
        assert result.total_cost >= 0

    def test_response_only_path_known_model(self) -> None:
        calculator = CostCalculator()
        result = calculator.calculate_cost(
            response="world",
            model_name="gpt-4o-mini",
        )
        assert result.calculation_method == "response_only"
        assert result.total_cost >= 0

    def test_alias_needs_explicit_litellm_alias(self) -> None:
        calculator = CostCalculator()
        with pytest.raises(UnknownModelError, match="has no known pricing"):
            calculator.calculate_cost(
                model_name="short-alias-x",
                input_tokens=100,
                output_tokens=50,
            )

        with patch(
            "traigent.utils.cost_calculator.litellm.model_alias_map",
            {"short-alias-x": "gpt-4o"},
        ):
            result = calculator.calculate_cost(
                model_name="short-alias-x",
                input_tokens=100,
                output_tokens=50,
            )
            assert result.total_cost > 0


class TestCustomPricingAndValidation:
    @pytest.fixture(autouse=True)
    def reset_custom_cache(self):
        old_cache = cc._CUSTOM_PRICING_CACHE
        old_key = cc._CUSTOM_PRICING_CACHE_KEY
        cc._CUSTOM_PRICING_CACHE = None
        cc._CUSTOM_PRICING_CACHE_KEY = None
        try:
            yield
        finally:
            cc._CUSTOM_PRICING_CACHE = old_cache
            cc._CUSTOM_PRICING_CACHE_KEY = old_key

    def test_calculate_cost_uses_custom_pricing_json(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
            json.dumps(
                {
                    "my-private-model": {
                        "input_cost_per_token": 1e-6,
                        "output_cost_per_token": 2e-6,
                    }
                }
            ),
        )
        monkeypatch.delenv("TRAIGENT_CUSTOM_MODEL_PRICING_FILE", raising=False)
        calculator = CostCalculator()
        result = calculator.calculate_cost(
            model_name="my-private-model",
            input_tokens=100,
            output_tokens=50,
        )
        assert result.input_cost == pytest.approx(100 * 1e-6)
        assert result.output_cost == pytest.approx(50 * 2e-6)

    def test_validate_model_name_known(self) -> None:
        calculator = CostCalculator()
        result = calculator.validate_model_name("gpt-4o")
        assert result["known_to_litellm"] is True
        assert result["not_found"] is False

    def test_validate_model_name_custom(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
            '{"foo-model":{"input_cost_per_token":1e-6,"output_cost_per_token":2e-6}}',
        )
        calculator = CostCalculator()
        result = calculator.validate_model_name("foo-model")
        assert result["custom_pricing"] is True
        assert result["not_found"] is False

    def test_validate_model_name_unknown(self) -> None:
        calculator = CostCalculator()
        result = calculator.validate_model_name("missing-model-zzz")
        assert result["not_found"] is True

    def test_clear_cache_resets_custom_pricing_cache(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
            '{"foo-model":{"input_cost_per_token":1e-6,"output_cost_per_token":2e-6}}',
        )
        calculator = CostCalculator()
        assert cc._CUSTOM_PRICING_CACHE is None
        calculator.validate_model_name("foo-model")
        assert cc._CUSTOM_PRICING_CACHE is not None
        calculator.clear_cache()
        assert cc._CUSTOM_PRICING_CACHE is None
        assert cc._CUSTOM_PRICING_CACHE_KEY is None


class TestConvenienceFunctions:
    def test_calculate_llm_cost_wrapper(self) -> None:
        result = calculate_llm_cost(
            model_name="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        assert result.total_cost > 0

    def test_validate_model_support_wrapper(self) -> None:
        result = validate_model_support("gpt-4o")
        assert result["known_to_litellm"] is True

    def test_get_cost_calculator_returns_singleton(self) -> None:
        c1 = get_cost_calculator()
        c2 = get_cost_calculator()
        assert c1 is c2

    def test_get_model_pricing_per_1k_known_model(self) -> None:
        input_per_1k, output_per_1k = get_model_pricing_per_1k("gpt-4o")
        assert input_per_1k > 0
        assert output_per_1k > 0

    def test_get_model_pricing_per_1k_unknown_model_returns_zero(self) -> None:
        input_per_1k, output_per_1k = get_model_pricing_per_1k("unknown-model-xyz")
        assert input_per_1k == 0.0
        assert output_per_1k == 0.0
