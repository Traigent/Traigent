"""Unit tests for cost_calculator public behaviors."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import traigent.utils.cost_calculator as cc
from traigent.utils.cost_calculator import (
    CostBreakdown,
    CostCalculator,
    UnknownModelError,
    _estimation_cost_from_tokens,
    calculate_completion_cost,
    calculate_llm_cost,
    calculate_prompt_cost,
    get_cost_calculator,
    get_model_pricing_per_1k,
    get_model_token_pricing,
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
        result = calculator.calculate_cost(
            prompt="hi", response="hello", model_name=None
        )
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

    def test_calculate_cost_logs_unexpected_exception(self) -> None:
        mock_logger = MagicMock()
        calculator = CostCalculator(logger=mock_logger)
        with patch.object(
            calculator, "_populate_cost", side_effect=RuntimeError("boom")
        ):
            result = calculator.calculate_cost(
                model_name="gpt-4o",
                input_tokens=100,
                output_tokens=50,
            )
        assert result.calculation_method == "error_RuntimeError"
        mock_logger.warning.assert_called_once()


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

    def test_validate_model_name_builtin_pricing_fallback(self) -> None:
        calculator = CostCalculator()
        result = calculator.validate_model_name("claude-sonnet")
        # Model resolves via litellm known models or builtin pricing depending
        # on litellm version. Either way, the model should be found.
        assert result["not_found"] is False
        assert (
            result.get("builtin_pricing") is True
            or result.get("known_to_litellm") is True
        )

    def test_validate_model_name_unknown(self) -> None:
        calculator = CostCalculator()
        result = calculator.validate_model_name("missing-model-zzz")
        assert result["not_found"] is True

    def test_validate_model_name_none_and_empty(self) -> None:
        calculator = CostCalculator()
        assert calculator.validate_model_name(None)["not_found"] is True
        assert calculator.validate_model_name("")["not_found"] is True

    def test_validate_model_name_litellm_unavailable(self) -> None:
        calculator = CostCalculator()
        with patch("traigent.utils.cost_calculator.LITELLM_AVAILABLE", False):
            result = calculator.validate_model_name("gpt-4o")
        assert result["error"] == "litellm library not available"
        assert result["available"] is False

    def test_validate_model_name_custom_pricing_invalid(self, monkeypatch) -> None:
        monkeypatch.setenv("TRAIGENT_CUSTOM_MODEL_PRICING_JSON", "{bad-json")
        calculator = CostCalculator()
        result = calculator.validate_model_name("foo-model")
        assert result["not_found"] is True
        assert "invalid JSON" in result["error"]

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


class TestPricingHelpers:
    def test_get_model_token_pricing_empty_raises(self) -> None:
        with pytest.raises(UnknownModelError, match="has no known pricing"):
            get_model_token_pricing("")

    def test_get_model_token_pricing_known_model_zero_rate_path(self) -> None:
        mock_litellm = MagicMock()
        mock_litellm.cost_per_token.return_value = (0.0, 0.0)
        with (
            patch("traigent.utils.cost_calculator.litellm", mock_litellm),
            patch(
                "traigent.utils.cost_calculator._is_model_known_to_litellm",
                return_value=True,
            ),
            patch("traigent.utils.cost_calculator.LITELLM_AVAILABLE", True),
        ):
            inp, out, method = get_model_token_pricing("known-zero-model")
        assert inp == 0.0
        assert out == 0.0
        assert method == "litellm"

    def test_fallback_cost_from_tokens_paths(self) -> None:
        exact = _estimation_cost_from_tokens("gpt-4o", 100, 50, _quiet=True)
        alias = _estimation_cost_from_tokens("claude-3-sonnet", 100, 50, _quiet=True)
        prefix = _estimation_cost_from_tokens("gpt-4o-2024-08-06", 100, 50, _quiet=True)
        reverse_prefix = _estimation_cost_from_tokens("gpt-4", 100, 50, _quiet=True)
        unknown = _estimation_cost_from_tokens("unknown-model", 100, 50, _quiet=True)
        assert exact[0] > 0
        assert alias[0] > 0
        assert prefix[0] > 0
        assert reverse_prefix[0] > 0
        assert unknown == (0.0, 0.0)


class TestDeprecatedPromptCompletionCost:
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

    def test_calculate_prompt_cost_with_message_list(self) -> None:
        with (
            patch(
                "traigent.utils.cost_calculator.litellm.token_counter", return_value=17
            ),
            patch(
                "traigent.utils.cost_calculator.litellm.cost_per_token",
                return_value=(17e-6, 0.0),
            ),
            patch(
                "traigent.utils.cost_calculator._is_model_known_to_litellm",
                return_value=True,
            ),
        ):
            cost = calculate_prompt_cost(
                [{"role": "user", "content": "hello"}],
                "gpt-4o",
            )
        assert cost == pytest.approx(17e-6)

    def test_calculate_prompt_cost_unknown_raises_after_litellm_failure(self) -> None:
        with patch(
            "traigent.utils.cost_calculator._try_litellm_prompt_cost",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(UnknownModelError, match="has no known pricing"):
                calculate_prompt_cost("hello", "unknown-model-z")

    def test_calculate_prompt_cost_uses_custom_pricing(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
            '{"custom-model":{"input_cost_per_token":1e-6,"output_cost_per_token":2e-6}}',
        )
        with patch(
            "traigent.utils.cost_calculator._try_litellm_prompt_cost",
            return_value=(None, 123),
        ):
            cost = calculate_prompt_cost("hello", "custom-model")
        assert cost == pytest.approx(123e-6)

    def test_calculate_completion_cost_known_zero_model_branch(self) -> None:
        with (
            patch(
                "traigent.utils.cost_calculator.litellm.token_counter", return_value=12
            ),
            patch(
                "traigent.utils.cost_calculator.litellm.cost_per_token",
                return_value=(0.0, 0.0),
            ),
            patch(
                "traigent.utils.cost_calculator._is_model_known_to_litellm",
                return_value=True,
            ),
        ):
            cost = calculate_completion_cost("world", "gpt-4o")
        assert cost == 0.0

    def test_calculate_completion_cost_unknown_raises_after_litellm_failure(
        self,
    ) -> None:
        with patch(
            "traigent.utils.cost_calculator._try_litellm_completion_cost",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(UnknownModelError, match="has no known pricing"):
                calculate_completion_cost("world", "unknown-model-z")

    def test_calculate_completion_cost_uses_custom_pricing(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
            '{"custom-model":{"input_cost_per_token":1e-6,"output_cost_per_token":2e-6}}',
        )
        with patch(
            "traigent.utils.cost_calculator._try_litellm_completion_cost",
            return_value=(None, 77),
        ):
            cost = calculate_completion_cost("world", "custom-model")
        assert cost == pytest.approx(154e-6)
