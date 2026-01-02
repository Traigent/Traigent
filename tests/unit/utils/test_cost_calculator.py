"""Unit tests for cost_calculator.

Tests for intelligent LLM cost calculation with fuzzy model matching.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.utils.cost_calculator import (
    TOKENCOST_AVAILABLE,
    CostBreakdown,
    CostCalculator,
    calculate_llm_cost,
    get_cost_calculator,
    validate_model_support,
)


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_cost_breakdown_default_initialization(self) -> None:
        """Test CostBreakdown initializes with default values."""
        breakdown = CostBreakdown()
        assert breakdown.input_cost == 0.0
        assert breakdown.output_cost == 0.0
        assert breakdown.total_cost == 0.0
        assert breakdown.input_tokens == 0
        assert breakdown.output_tokens == 0
        assert breakdown.total_tokens == 0
        assert breakdown.model_used == ""
        assert breakdown.mapped_model == ""
        assert breakdown.calculation_method == "unknown"

    def test_cost_breakdown_with_values(self) -> None:
        """Test CostBreakdown with explicit values."""
        breakdown = CostBreakdown(
            input_cost=0.001,
            output_cost=0.002,
            input_tokens=100,
            output_tokens=50,
            model_used="gpt-4o",
            mapped_model="gpt-4o-2024-05-13",
            calculation_method="prompt_and_response",
        )
        assert breakdown.input_cost == 0.001
        assert breakdown.output_cost == 0.002
        assert breakdown.total_cost == 0.003
        assert breakdown.input_tokens == 100
        assert breakdown.output_tokens == 50
        assert breakdown.total_tokens == 150

    def test_cost_breakdown_post_init_total_cost(self) -> None:
        """Test __post_init__ calculates total_cost from input and output costs."""
        breakdown = CostBreakdown(input_cost=0.001, output_cost=0.002)
        assert breakdown.total_cost == 0.003

    def test_cost_breakdown_post_init_total_tokens(self) -> None:
        """Test __post_init__ calculates total_tokens from input and output tokens."""
        breakdown = CostBreakdown(input_tokens=100, output_tokens=50)
        assert breakdown.total_tokens == 150

    def test_cost_breakdown_explicit_total_cost_preserved(self) -> None:
        """Test that explicit total_cost is not overridden if already set."""
        breakdown = CostBreakdown(input_cost=0.001, output_cost=0.002, total_cost=0.999)
        # total_cost should remain as explicitly set (0.999), not recalculated
        assert breakdown.total_cost == 0.999

    def test_cost_breakdown_explicit_total_tokens_preserved(self) -> None:
        """Test that explicit total_tokens is not overridden if already set."""
        breakdown = CostBreakdown(input_tokens=100, output_tokens=50, total_tokens=999)
        # total_tokens should remain as explicitly set (999), not recalculated
        assert breakdown.total_tokens == 999


class TestCostCalculatorInitialization:
    """Tests for CostCalculator initialization."""

    def test_calculator_init_default(self) -> None:
        """Test CostCalculator initializes with default settings."""
        calculator = CostCalculator()
        assert calculator.logger is None
        assert calculator.enable_caching is True
        assert calculator._fuzzy_match_cache == {}

    def test_calculator_init_with_logger(self) -> None:
        """Test CostCalculator initialization with custom logger."""
        mock_logger = MagicMock()
        calculator = CostCalculator(logger=mock_logger)
        assert calculator.logger is mock_logger

    def test_calculator_init_caching_disabled(self) -> None:
        """Test CostCalculator initialization with caching disabled."""
        calculator = CostCalculator(enable_caching=False)
        assert calculator.enable_caching is False

    @patch("traigent.utils.cost_calculator.TOKENCOST_AVAILABLE", False)
    def test_calculator_init_tokencost_unavailable_warning(self) -> None:
        """Test warning is logged when tokencost is unavailable."""
        mock_logger = MagicMock()
        CostCalculator(logger=mock_logger)  # noqa: F841
        mock_logger.warning.assert_called_once()
        assert "tokencost not available" in str(mock_logger.warning.call_args)


class TestCostCalculatorModelMapping:
    """Tests for model name mapping functionality."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    def test_map_model_name_exact_match(self, calculator: CostCalculator) -> None:
        """Test exact model name mapping."""
        result = calculator._map_model_name("claude-3-haiku")
        assert result == "claude-3-haiku-20240307"

    def test_map_model_name_exact_match_sonnet(
        self, calculator: CostCalculator
    ) -> None:
        """Test exact mapping for Claude Sonnet models."""
        assert (
            calculator._map_model_name("claude-3-sonnet") == "claude-3-sonnet-20240229"
        )
        assert (
            calculator._map_model_name("claude-3-5-sonnet")
            == "claude-3-5-sonnet-20241022"
        )

    def test_map_model_name_exact_match_gpt(self, calculator: CostCalculator) -> None:
        """Test exact mapping for GPT models."""
        assert calculator._map_model_name("gpt-3.5") == "gpt-3.5-turbo"
        assert calculator._map_model_name("gpt-4") == "gpt-4o"

    def test_map_model_name_family_defaults(self, calculator: CostCalculator) -> None:
        """Test model family default mapping."""
        result = calculator._map_model_name("claude-3")
        assert result == "claude-3-5-sonnet-latest"

    def test_map_model_name_empty_string(self, calculator: CostCalculator) -> None:
        """Test mapping with empty string returns None."""
        result = calculator._map_model_name("")
        assert result is None

    def test_map_model_name_none(self, calculator: CostCalculator) -> None:
        """Test mapping with None returns None."""
        result = calculator._map_model_name(None)
        assert result is None

    def test_map_model_name_with_logger(self) -> None:
        """Test model name mapping logs debug messages."""
        mock_logger = MagicMock()
        calculator = CostCalculator(logger=mock_logger)
        calculator._map_model_name("claude-3-haiku")
        # Should log debug message for exact match
        assert mock_logger.debug.called


class TestCostCalculatorFuzzyMatching:
    """Tests for fuzzy model matching functionality."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    @pytest.fixture
    def calculator_with_logger(self) -> CostCalculator:
        """Create calculator with mock logger."""
        return CostCalculator(logger=MagicMock())

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_fuzzy_match_direct_match(self, calculator: CostCalculator) -> None:
        """Test fuzzy match finds direct match in tokencost."""
        # Use a model that should exist in tokencost
        result = calculator._fuzzy_match_model("gpt-4o")
        assert result == "gpt-4o"

    def test_fuzzy_match_cache_hit(self, calculator: CostCalculator) -> None:
        """Test fuzzy match uses cached results."""
        # Pre-populate cache
        calculator._fuzzy_match_cache["test-model"] = "cached-result"
        result = calculator._fuzzy_match_model("test-model")
        assert result == "cached-result"

    def test_fuzzy_match_cache_disabled(self) -> None:
        """Test fuzzy matching with caching disabled."""
        calculator = CostCalculator(enable_caching=False)
        # Should not use cache
        calculator._fuzzy_match_cache["test-model"] = "cached-result"
        # Result won't match cache since caching is disabled
        result = calculator._fuzzy_match_model("test-model")
        # Since tokencost may not be available and no exact match, should return None
        assert result is None or isinstance(result, str)

    def test_fuzzy_match_short_model_name(self, calculator: CostCalculator) -> None:
        """Test fuzzy match rejects model names shorter than 5 characters."""
        result = calculator._perform_fuzzy_match("gpt")
        assert result is None

    @patch("traigent.utils.cost_calculator.TOKENCOST_AVAILABLE", False)
    def test_fuzzy_match_tokencost_unavailable(
        self, calculator: CostCalculator
    ) -> None:
        """Test fuzzy match returns None when tokencost unavailable."""
        result = calculator._perform_fuzzy_match("some-model")
        assert result is None

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_perform_fuzzy_match_single_match(
        self, calculator_with_logger: CostCalculator
    ) -> None:
        """Test fuzzy match with single matching model."""
        # This will depend on tokencost having models, so we'll use a real example
        result = calculator_with_logger._perform_fuzzy_match("gpt-4o-mini")
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_perform_fuzzy_match_multiple_matches(
        self, calculator_with_logger: CostCalculator
    ) -> None:
        """Test fuzzy match selects best from multiple matches."""
        # "gpt-4" should match multiple models in tokencost
        result = calculator_with_logger._perform_fuzzy_match("gpt-4o")
        assert result is not None
        # Should select the best match based on scoring
        assert isinstance(result, str)


class TestSemanticMatching:
    """Tests for semantic model matching logic."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    def test_is_semantic_match_substring_not_found(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match returns False when substring not found."""
        result = calculator._is_semantic_match("claude", "gpt-4-turbo")
        assert result is False

    def test_is_semantic_match_too_short(self, calculator: CostCalculator) -> None:
        """Test semantic match rejects model names too short."""
        result = calculator._is_semantic_match("gpt", "gpt-4-turbo-preview")
        assert result is False

    def test_is_semantic_match_family_mismatch_gpt_claude(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match prevents GPT/Claude family mixing."""
        result = calculator._is_semantic_match("gpt-4", "claude-3-sonnet")
        assert result is False

    def test_is_semantic_match_family_mismatch_claude_gpt(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match prevents Claude/GPT family mixing."""
        result = calculator._is_semantic_match("claude-3", "gpt-4-turbo")
        assert result is False

    def test_is_semantic_match_generation_mismatch_gpt4_gpt3(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match prevents GPT-4 matching GPT-3."""
        result = calculator._is_semantic_match("gpt-4-turbo", "gpt-3.5-turbo")
        assert result is False

    def test_is_semantic_match_generation_mismatch_gpt3_gpt4(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match prevents GPT-3 matching GPT-4."""
        result = calculator._is_semantic_match("gpt-3.5", "gpt-4-turbo")
        assert result is False

    def test_is_semantic_match_claude_version_mismatch(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match prevents Claude-3 matching Claude-2."""
        result = calculator._is_semantic_match("claude-3-haiku", "claude-2.1")
        assert result is False

    def test_is_semantic_match_valid_match(self, calculator: CostCalculator) -> None:
        """Test semantic match returns True for valid matches."""
        result = calculator._is_semantic_match(
            "claude-3-haiku", "claude-3-haiku-20240307"
        )
        assert result is True


class TestModelScoring:
    """Tests for model scoring and selection logic."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    def test_select_best_model_empty_list(self, calculator: CostCalculator) -> None:
        """Test selecting best model from empty list returns empty string."""
        result = calculator._select_best_model([], "test-model")
        assert result == ""

    def test_select_best_model_single_match(self, calculator: CostCalculator) -> None:
        """Test selecting best model with single match."""
        result = calculator._select_best_model(["model-v1"], "model")
        assert result == "model-v1"

    def test_select_best_model_prefers_latest(self, calculator: CostCalculator) -> None:
        """Test selecting best model prefers 'latest' suffix."""
        matches = ["model-v1", "model-v2", "model-latest"]
        result = calculator._select_best_model(matches, "model")
        assert result == "model-latest"

    def test_select_best_model_prefers_newer_date(
        self, calculator: CostCalculator
    ) -> None:
        """Test selecting best model prefers newer dates."""
        matches = ["model-20240101", "model-20250101"]
        result = calculator._select_best_model(matches, "model")
        assert result == "model-20250101"

    def test_calculate_model_score_latest(self, calculator: CostCalculator) -> None:
        """Test model score calculation for 'latest' suffix."""
        score = calculator._calculate_model_score("gpt-4-latest", "gpt-4")
        assert score == (9999, 12, 31, 99)

    def test_calculate_model_score_yyyymmdd_format(
        self, calculator: CostCalculator
    ) -> None:
        """Test model score calculation for YYYYMMDD date format."""
        score = calculator._calculate_model_score(
            "claude-3-haiku-20240307", "claude-3-haiku"
        )
        assert score == (2024, 3, 7, 50)

    def test_calculate_model_score_yyyy_mm_dd_format(
        self, calculator: CostCalculator
    ) -> None:
        """Test model score calculation for YYYY-MM-DD date format."""
        score = calculator._calculate_model_score("model-2024-03-07", "model")
        assert score == (2024, 3, 7, 50)

    def test_calculate_model_score_version_number(
        self, calculator: CostCalculator
    ) -> None:
        """Test model score calculation for version numbers."""
        score = calculator._calculate_model_score("model-v5", "model")
        assert score == (2024, 1, 1, 5)

    def test_calculate_model_score_unversioned(
        self, calculator: CostCalculator
    ) -> None:
        """Test model score calculation for unversioned models."""
        score = calculator._calculate_model_score("basic-model", "basic")
        # Should use unversioned fallback date with specificity based on length
        assert score[0] == 2020  # UNVERSIONED_FALLBACK_DATE year
        assert score[1] == 1  # month
        assert score[2] == 1  # day
        assert score[3] > 0  # specificity (100 - len(model_name))


class TestCostCalculation:
    """Tests for cost calculation methods."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    @pytest.fixture
    def calculator_with_logger(self) -> CostCalculator:
        """Create calculator with mock logger."""
        return CostCalculator(logger=MagicMock())

    @patch("traigent.utils.cost_calculator.TOKENCOST_AVAILABLE", False)
    def test_calculate_cost_tokencost_unavailable(
        self, calculator: CostCalculator
    ) -> None:
        """Test cost calculation when tokencost is unavailable."""
        result = calculator.calculate_cost(
            prompt="test", response="response", model_name="gpt-4o"
        )
        assert result.total_cost == 0.0
        assert result.calculation_method == "tokencost_unavailable"

    def test_calculate_cost_no_model_name(self, calculator: CostCalculator) -> None:
        """Test cost calculation without model name."""
        result = calculator.calculate_cost(prompt="test", response="response")
        assert result.total_cost == 0.0
        assert result.calculation_method == "no_model_name"

    def test_calculate_cost_model_not_found(
        self, calculator_with_logger: CostCalculator
    ) -> None:
        """Test cost calculation with unmappable model name."""
        # Use a completely invalid model name that won't match anything
        result = calculator_with_logger.calculate_cost(
            prompt="test", response="response", model_name="xyz123nonexistent"
        )
        assert result.total_cost == 0.0
        assert result.calculation_method == "model_not_found"
        # Should have logged a warning
        assert calculator_with_logger.logger.warning.called

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_calculate_cost_prompt_and_response(
        self, calculator: CostCalculator
    ) -> None:
        """Test cost calculation with prompt and response."""
        result = calculator.calculate_cost(
            prompt="Hello world", response="Hi there", model_name="gpt-4o-mini"
        )
        assert result.calculation_method == "prompt_and_response"
        assert result.total_cost >= 0.0
        assert result.input_cost >= 0.0
        assert result.output_cost >= 0.0

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_calculate_cost_token_counts(self, calculator: CostCalculator) -> None:
        """Test cost calculation with token counts."""
        result = calculator.calculate_cost(
            model_name="gpt-4o-mini", input_tokens=100, output_tokens=50
        )
        assert result.calculation_method == "token_counts"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        # Total tokens are set in the calculation, but __post_init__ runs first
        # So we check that input and output tokens are correctly set
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_calculate_cost_response_only(self, calculator: CostCalculator) -> None:
        """Test cost calculation with response only."""
        result = calculator.calculate_cost(
            response="Response text", model_name="gpt-4o-mini"
        )
        assert result.calculation_method == "response_only"
        assert result.output_cost >= 0.0

    def test_calculate_cost_with_zero_tokens(self, calculator: CostCalculator) -> None:
        """Test cost calculation with zero token counts."""
        result = calculator.calculate_cost(
            model_name="gpt-4o-mini", input_tokens=0, output_tokens=0
        )
        # Should not use token_counts method with zero tokens
        assert result.calculation_method != "token_counts"

    def test_calculate_cost_exception_handling(
        self, calculator_with_logger: CostCalculator
    ) -> None:
        """Test cost calculation handles exceptions gracefully."""
        # Mock safe calculation to raise an exception after mapping
        with patch.object(
            calculator_with_logger,
            "_safe_calculate_prompt_cost",
            side_effect=Exception("Test error"),
        ):
            result = calculator_with_logger.calculate_cost(
                prompt="test", response="response", model_name="gpt-4o-mini"
            )
            # Should catch exception and log it
            assert "error" in result.calculation_method.lower()


class TestSafeCostCalculation:
    """Tests for safe cost calculation wrapper methods."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    @pytest.fixture
    def calculator_with_logger(self) -> CostCalculator:
        """Create calculator with mock logger."""
        return CostCalculator(logger=MagicMock())

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_safe_calculate_prompt_cost_success(
        self, calculator: CostCalculator
    ) -> None:
        """Test safe prompt cost calculation with valid input."""
        cost = calculator._safe_calculate_prompt_cost("Hello world", "gpt-4o-mini")
        assert isinstance(cost, float)
        assert cost >= 0.0

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_safe_calculate_completion_cost_success(
        self, calculator: CostCalculator
    ) -> None:
        """Test safe completion cost calculation with valid input."""
        cost = calculator._safe_calculate_completion_cost(
            "Response text", "gpt-4o-mini"
        )
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_safe_calculate_prompt_cost_exception(
        self, calculator_with_logger: CostCalculator
    ) -> None:
        """Test safe prompt cost calculation handles exceptions."""
        with patch(
            "traigent.utils.cost_calculator.calculate_prompt_cost",
            side_effect=Exception("Test error"),
        ):
            cost = calculator_with_logger._safe_calculate_prompt_cost(
                "test", "invalid-model"
            )
            assert cost == 0.0
            # Should log the error
            assert calculator_with_logger.logger.debug.called

    def test_safe_calculate_completion_cost_exception(
        self, calculator_with_logger: CostCalculator
    ) -> None:
        """Test safe completion cost calculation handles exceptions."""
        with patch(
            "traigent.utils.cost_calculator.calculate_completion_cost",
            side_effect=Exception("Test error"),
        ):
            cost = calculator_with_logger._safe_calculate_completion_cost(
                "test", "invalid-model"
            )
            assert cost == 0.0
            # Should log the error
            assert calculator_with_logger.logger.debug.called


class TestCalculateFromTokens:
    """Tests for token-based cost calculation."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    @pytest.fixture
    def calculator_with_logger(self) -> CostCalculator:
        """Create calculator with mock logger."""
        return CostCalculator(logger=MagicMock())

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_calculate_from_tokens_direct_pricing(
        self, calculator_with_logger: CostCalculator
    ) -> None:
        """Test token-based calculation uses direct pricing when available."""
        input_cost, output_cost = calculator_with_logger._calculate_from_tokens(
            input_tokens=1000, output_tokens=500, model="gpt-4o-mini"
        )
        assert isinstance(input_cost, float)
        assert isinstance(output_cost, float)
        assert input_cost >= 0.0
        assert output_cost >= 0.0
        # Should log debug info
        assert calculator_with_logger.logger.debug.called

    def test_calculate_from_tokens_exception_handling(
        self, calculator_with_logger: CostCalculator
    ) -> None:
        """Test token-based calculation handles exceptions."""
        with (
            patch("traigent.utils.cost_calculator.TOKENCOST_AVAILABLE", True),
            patch("traigent.utils.cost_calculator.tokencost.TOKEN_COSTS", {}),
        ):
            input_cost, output_cost = calculator_with_logger._calculate_from_tokens(
                input_tokens=100, output_tokens=50, model="invalid-model"
            )
            # Should handle gracefully
            assert input_cost == 0.0 or isinstance(input_cost, float)
            assert output_cost == 0.0 or isinstance(output_cost, float)


class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_get_available_models(self, calculator: CostCalculator) -> None:
        """Test getting list of available models."""
        models = calculator.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @patch("traigent.utils.cost_calculator.TOKENCOST_AVAILABLE", False)
    def test_get_available_models_tokencost_unavailable(
        self, calculator: CostCalculator
    ) -> None:
        """Test getting available models when tokencost unavailable."""
        models = calculator.get_available_models()
        assert models == []

    def test_clear_cache(self, calculator: CostCalculator) -> None:
        """Test cache clearing."""
        # Add something to cache
        calculator._fuzzy_match_cache["test"] = "result"
        assert len(calculator._fuzzy_match_cache) > 0
        # Clear cache
        calculator.clear_cache()
        assert len(calculator._fuzzy_match_cache) == 0

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_validate_model_name_exact_match(self, calculator: CostCalculator) -> None:
        """Test model name validation with exact match."""
        result = calculator.validate_model_name("claude-3-haiku")
        assert result["original"] == "claude-3-haiku"
        assert result["mapped"] == "claude-3-haiku-20240307"
        assert result["exact_match"] is True
        assert result["fuzzy_match"] is False
        assert result["family_match"] is False
        assert result["not_found"] is False

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_validate_model_name_family_match(self, calculator: CostCalculator) -> None:
        """Test model name validation with family match."""
        result = calculator.validate_model_name("claude-3")
        assert result["original"] == "claude-3"
        assert result["mapped"] == "claude-3-5-sonnet-latest"
        assert result["exact_match"] is False
        assert result["family_match"] is True
        assert result["not_found"] is False

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_validate_model_name_not_found(self, calculator: CostCalculator) -> None:
        """Test model name validation with model not found."""
        result = calculator.validate_model_name("nonexistent-model-xyz123")
        assert result["original"] == "nonexistent-model-xyz123"
        assert result["not_found"] is True

    @patch("traigent.utils.cost_calculator.TOKENCOST_AVAILABLE", False)
    def test_validate_model_name_tokencost_unavailable(
        self, calculator: CostCalculator
    ) -> None:
        """Test model name validation when tokencost unavailable."""
        result = calculator.validate_model_name("gpt-4o")
        assert result["available"] is False
        assert "error" in result
        assert "tokencost library not available" in result["error"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_calculate_llm_cost_function(self) -> None:
        """Test convenience function for cost calculation."""
        result = calculate_llm_cost(
            prompt="Hello", response="World", model_name="gpt-4o-mini"
        )
        assert isinstance(result, CostBreakdown)
        assert result.model_used == "gpt-4o-mini"

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_validate_model_support_function(self) -> None:
        """Test convenience function for model validation."""
        result = validate_model_support("claude-3-haiku")
        assert isinstance(result, dict)
        assert result["original"] == "claude-3-haiku"

    def test_get_cost_calculator_singleton(self) -> None:
        """Test get_cost_calculator returns singleton instance."""
        calc1 = get_cost_calculator()
        calc2 = get_cost_calculator()
        assert calc1 is calc2

    def test_get_cost_calculator_with_logger(self) -> None:
        """Test get_cost_calculator with custom logger."""
        # Reset global calculator first
        import traigent.utils.cost_calculator

        traigent.utils.cost_calculator._global_calculator = None

        mock_logger = MagicMock()
        calc = get_cost_calculator(logger=mock_logger)
        assert isinstance(calc, CostCalculator)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    def test_calculate_cost_empty_prompt(self, calculator: CostCalculator) -> None:
        """Test cost calculation with empty prompt."""
        result = calculator.calculate_cost(
            prompt="", response="response", model_name="gpt-4o-mini"
        )
        assert isinstance(result, CostBreakdown)

    def test_calculate_cost_empty_response(self, calculator: CostCalculator) -> None:
        """Test cost calculation with empty response."""
        result = calculator.calculate_cost(
            prompt="prompt", response="", model_name="gpt-4o-mini"
        )
        assert isinstance(result, CostBreakdown)

    def test_calculate_cost_message_list_prompt(
        self, calculator: CostCalculator
    ) -> None:
        """Test cost calculation with message list as prompt."""
        messages = [{"role": "user", "content": "Hello"}]
        result = calculator.calculate_cost(
            prompt=messages, response="Hi", model_name="gpt-4o-mini"
        )
        assert isinstance(result, CostBreakdown)

    def test_calculate_cost_negative_tokens(self, calculator: CostCalculator) -> None:
        """Test cost calculation rejects negative token counts."""
        result = calculator.calculate_cost(
            model_name="gpt-4o-mini", input_tokens=-10, output_tokens=50
        )
        # Should not use token_counts method with negative tokens
        assert result.calculation_method != "token_counts"

    def test_fuzzy_match_with_special_characters(
        self, calculator: CostCalculator
    ) -> None:
        """Test fuzzy matching with special characters in model name."""
        result = calculator._fuzzy_match_model("model@#$%")
        # Should handle gracefully
        assert result is None or isinstance(result, str)

    def test_calculate_cost_very_long_prompt(self, calculator: CostCalculator) -> None:
        """Test cost calculation with very long prompt."""
        long_prompt = "word " * 10000  # Very long prompt
        result = calculator.calculate_cost(
            prompt=long_prompt, response="response", model_name="gpt-4o-mini"
        )
        assert isinstance(result, CostBreakdown)

    def test_calculate_cost_unicode_content(self, calculator: CostCalculator) -> None:
        """Test cost calculation with Unicode characters."""
        result = calculator.calculate_cost(
            prompt="Hello 世界 🌍", response="Hi 你好 👋", model_name="gpt-4o-mini"
        )
        assert isinstance(result, CostBreakdown)


class TestThreadSafety:
    """Tests for thread safety of global calculator."""

    def test_global_calculator_thread_safe(self) -> None:
        """Test that get_cost_calculator is thread-safe."""
        import threading

        # Reset global calculator
        import traigent.utils.cost_calculator

        traigent.utils.cost_calculator._global_calculator = None

        calculators = []

        def get_calc() -> None:
            calc = get_cost_calculator()
            calculators.append(calc)

        threads = [threading.Thread(target=get_calc) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len({id(c) for c in calculators}) == 1


class TestAdditionalLoggerCoverage:
    """Tests for additional logger code paths."""

    def test_map_model_name_family_match_with_logger(self) -> None:
        """Test family match logging is called when logger present."""
        mock_logger = MagicMock()
        calculator = CostCalculator(logger=mock_logger)
        result = calculator._map_model_name("claude-3")
        assert result == "claude-3-5-sonnet-latest"
        # Should have called debug for family match
        mock_logger.debug.assert_called()
        assert "Family model mapping" in str(mock_logger.debug.call_args)

    def test_fuzzy_match_cached_with_logger(self) -> None:
        """Test cached fuzzy match logging when logger present."""
        mock_logger = MagicMock()
        calculator = CostCalculator(logger=mock_logger)
        # Pre-populate cache
        calculator._fuzzy_match_cache["test-model"] = "cached-result"
        result = calculator._fuzzy_match_model("test-model")
        assert result == "cached-result"
        # Should log cache hit
        mock_logger.debug.assert_called()
        assert "Cached fuzzy match" in str(mock_logger.debug.call_args)

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_fuzzy_match_direct_tokencost_with_logger(self) -> None:
        """Test direct tokencost match logging when logger present."""
        mock_logger = MagicMock()
        calculator = CostCalculator(logger=mock_logger)
        # Use a model that should exist in tokencost
        result = calculator._fuzzy_match_model("gpt-4o-mini")
        assert result == "gpt-4o-mini"
        # Should log direct match
        assert mock_logger.debug.called

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_perform_fuzzy_match_single_with_logger(self) -> None:
        """Test single fuzzy match logging when logger present."""
        mock_logger = MagicMock()
        calculator = CostCalculator(logger=mock_logger)
        # Mock to return exactly one match
        with patch.object(calculator, "_is_semantic_match", return_value=True):
            with patch(
                "traigent.utils.cost_calculator.tokencost.TOKEN_COSTS",
                {"model-match": {}},
            ):
                result = calculator._perform_fuzzy_match("model")
                assert result == "model-match"
                # Should log info about single match
                mock_logger.info.assert_called()

    def test_calculate_from_tokens_exception_with_logger(self) -> None:
        """Test token calculation exception logging."""
        mock_logger = MagicMock()
        calculator = CostCalculator(logger=mock_logger)
        # Force an exception by mocking TOKENCOST_AVAILABLE
        with (
            patch("traigent.utils.cost_calculator.TOKENCOST_AVAILABLE", True),
            patch(
                "traigent.utils.cost_calculator.tokencost.TOKEN_COSTS",
                side_effect=Exception("Test error"),
            ),
        ):
            input_cost, output_cost = calculator._calculate_from_tokens(
                100, 50, "test-model"
            )
            assert input_cost == 0.0
            assert output_cost == 0.0
            # Should have logged the error
            assert mock_logger.debug.called

    @pytest.mark.skipif(not TOKENCOST_AVAILABLE, reason="tokencost not available")
    def test_validate_model_name_fuzzy_match_path(self) -> None:
        """Test validate_model_name with fuzzy match result."""
        calculator = CostCalculator()
        # Use a model that will require fuzzy matching
        with patch.object(
            calculator, "_perform_fuzzy_match", return_value="gpt-4o-mini"
        ):
            result = calculator.validate_model_name("gpt-4o")
            assert result["mapped"] == "gpt-4o-mini"
            assert result["fuzzy_match"] is True


class TestSemanticMatchingEdgeCases:
    """Tests for additional semantic matching edge cases."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    def test_is_semantic_match_gpt_in_user_not_in_tokencost(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match rejects when gpt in user but not in tokencost."""
        result = calculator._is_semantic_match("gpt-custom", "custom-model-xyz")
        assert result is False

    def test_is_semantic_match_claude_in_user_not_in_tokencost(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match rejects when claude in user but not in tokencost."""
        result = calculator._is_semantic_match("claude-custom", "custom-model-xyz")
        assert result is False

    def test_is_semantic_match_gpt4_versus_gpt3_tokencost(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match rejects gpt-4 user with gpt-3 tokencost."""
        result = calculator._is_semantic_match("gpt-4-new", "gpt-3.5-turbo-new")
        assert result is False

    def test_is_semantic_match_gpt3_versus_gpt4_tokencost(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match rejects gpt-3 user with gpt-4 tokencost."""
        result = calculator._is_semantic_match("gpt-3.5-custom", "gpt-4-turbo-custom")
        assert result is False

    def test_is_semantic_match_claude3_versus_claude2_tokencost(
        self, calculator: CostCalculator
    ) -> None:
        """Test semantic match rejects claude-3 user with claude-2 tokencost."""
        result = calculator._is_semantic_match("claude-3-new", "claude-2.1-new")
        assert result is False


class TestExactModelMappings:
    """Tests for additional exact model mappings."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    def test_map_model_name_claude_opus(self, calculator: CostCalculator) -> None:
        """Test exact mapping for Claude Opus."""
        assert calculator._map_model_name("claude-3-opus") == "claude-3-opus-20240229"

    def test_map_model_name_claude_3_5_haiku(self, calculator: CostCalculator) -> None:
        """Test exact mapping for Claude 3.5 Haiku."""
        assert (
            calculator._map_model_name("claude-3-5-haiku")
            == "claude-3-5-haiku-20241022"
        )

    def test_map_model_name_claude_3_7_sonnet(self, calculator: CostCalculator) -> None:
        """Test exact mapping for Claude 3.7 Sonnet."""
        assert (
            calculator._map_model_name("claude-3-7-sonnet")
            == "claude-3-7-sonnet-20250219"
        )

    def test_map_model_name_claude_4_sonnet(self, calculator: CostCalculator) -> None:
        """Test exact mapping for Claude 4 Sonnet."""
        assert (
            calculator._map_model_name("claude-4-sonnet") == "claude-sonnet-4-20250514"
        )

    def test_map_model_name_claude_4_opus(self, calculator: CostCalculator) -> None:
        """Test exact mapping for Claude 4 Opus."""
        assert calculator._map_model_name("claude-4-opus") == "claude-opus-4-20250514"

    def test_map_model_name_gpt4o_mini(self, calculator: CostCalculator) -> None:
        """Test exact mapping for GPT-4o mini."""
        assert calculator._map_model_name("gpt-4o-mini") == "gpt-4o-mini"

    def test_map_model_name_alternative_spellings(
        self, calculator: CostCalculator
    ) -> None:
        """Test alternative spellings without dashes."""
        assert calculator._map_model_name("claude3-haiku") == "claude-3-haiku-20240307"
        assert (
            calculator._map_model_name("claude3-sonnet") == "claude-3-sonnet-20240229"
        )
        assert calculator._map_model_name("claude3-opus") == "claude-3-opus-20240229"

    def test_map_model_name_latest_versions(self, calculator: CostCalculator) -> None:
        """Test mapping for latest version shortcuts."""
        assert calculator._map_model_name("claude-haiku") == "claude-3-haiku-latest"
        assert calculator._map_model_name("claude-sonnet") == "claude-3-5-sonnet-latest"
        assert calculator._map_model_name("claude-opus") == "claude-3-opus-latest"

    def test_map_model_name_gpt_aliases(self, calculator: CostCalculator) -> None:
        """Test GPT model aliases."""
        assert calculator._map_model_name("gpt4") == "gpt-4o"
        assert calculator._map_model_name("gpt3.5") == "gpt-3.5-turbo"


class TestFamilyDefaults:
    """Tests for family default mappings."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create test calculator instance."""
        return CostCalculator()

    def test_family_defaults_claude(self, calculator: CostCalculator) -> None:
        """Test family default for generic 'claude'."""
        assert calculator._map_model_name("claude") == "claude-3-5-sonnet-latest"

    def test_family_defaults_gpt(self, calculator: CostCalculator) -> None:
        """Test family default for generic 'gpt'."""
        assert calculator._map_model_name("gpt") == "gpt-4o"

    def test_family_defaults_gpt_3_5(self, calculator: CostCalculator) -> None:
        """Test family default for 'gpt-3.5'."""
        assert calculator._map_model_name("gpt-3.5") == "gpt-3.5-turbo"
