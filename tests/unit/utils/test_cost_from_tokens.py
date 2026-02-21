"""Unit tests for cost_from_tokens() — the canonical cost entry point.

Tests validate:
- Known models return positive costs (via litellm)
- Partial token data (input-only, output-only) is handled correctly
- Unknown models raise UnknownModelError (strict) or return (0,0) (non-strict)
- Provider-prefixed models resolve correctly
- litellm unavailability raises RuntimeError (strict) or returns (0,0)
- Negative tokens raise ValueError
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from traigent.utils.cost_calculator import (
    CostCalculator,
    UnknownModelError,
    cost_from_tokens,
)


class TestCostFromTokensKnownModels:
    """Tests for known models that litellm can price."""

    def test_known_model_both_tokens_positive(self) -> None:
        """Known model with both input and output tokens returns positive costs."""
        input_cost, output_cost = cost_from_tokens(100, 50, "gpt-4o")
        assert input_cost > 0, "Input cost should be positive for known model"
        assert output_cost > 0, "Output cost should be positive for known model"

    def test_known_model_input_only(self) -> None:
        """Known model with output_tokens=0 still calculates input cost."""
        input_cost, output_cost = cost_from_tokens(100, 0, "gpt-4o")
        assert input_cost > 0, "Input cost should be positive"
        assert output_cost == 0.0, "Output cost should be zero with 0 output tokens"

    def test_known_model_output_only(self) -> None:
        """Known model with input_tokens=0 still calculates output cost."""
        input_cost, output_cost = cost_from_tokens(0, 50, "gpt-4o")
        assert input_cost == 0.0, "Input cost should be zero with 0 input tokens"
        assert output_cost > 0, "Output cost should be positive"

    def test_known_model_zero_both(self) -> None:
        """Known model with both tokens=0 returns legitimate zeros."""
        input_cost, output_cost = cost_from_tokens(0, 0, "gpt-4o")
        assert input_cost == 0.0
        assert output_cost == 0.0

    def test_cost_scales_linearly(self) -> None:
        """Cost doubles when token count doubles."""
        cost_100, _ = cost_from_tokens(100, 0, "gpt-4o")
        cost_200, _ = cost_from_tokens(200, 0, "gpt-4o")
        assert abs(cost_200 - 2 * cost_100) < 1e-12, "Cost should scale linearly"

    def test_different_models_different_costs(self) -> None:
        """Different models have different per-token rates."""
        cost_4o_in, cost_4o_out = cost_from_tokens(1000, 1000, "gpt-4o")
        cost_mini_in, cost_mini_out = cost_from_tokens(1000, 1000, "gpt-4o-mini")
        assert cost_4o_in > cost_mini_in, "gpt-4o should cost more than gpt-4o-mini"
        assert cost_4o_out > cost_mini_out, "gpt-4o should cost more than gpt-4o-mini"


class TestCostFromTokensModelResolution:
    """Tests for model name resolution (EXACT_MODEL_MAPPING + normalization)."""

    def test_provider_prefixed_model(self) -> None:
        """Provider-prefixed model resolves correctly."""
        cost_prefixed = cost_from_tokens(100, 50, "openai/gpt-4o")
        cost_plain = cost_from_tokens(100, 50, "gpt-4o")
        # Should produce identical results
        assert cost_prefixed[0] == pytest.approx(cost_plain[0], rel=1e-6)
        assert cost_prefixed[1] == pytest.approx(cost_plain[1], rel=1e-6)

    def test_exact_model_mapping(self) -> None:
        """Short model names resolve via EXACT_MODEL_MAPPING."""
        # "claude-3-haiku" maps to "claude-3-haiku-20240307"
        input_cost, output_cost = cost_from_tokens(100, 50, "claude-3-haiku")
        assert (
            input_cost > 0 or output_cost > 0
        ), "Mapped model should produce positive cost"

    def test_claude_haiku_alias_maps_to_priced_model(self) -> None:
        """Legacy alias claude-haiku resolves to priced dated model."""
        alias_cost = cost_from_tokens(100, 50, "claude-haiku")
        dated_cost = cost_from_tokens(100, 50, "claude-3-haiku-20240307")
        assert alias_cost[0] == pytest.approx(dated_cost[0], rel=1e-6)
        assert alias_cost[1] == pytest.approx(dated_cost[1], rel=1e-6)

    def test_colon_prefixed_model(self) -> None:
        """Colon-prefixed provider model resolves."""
        cost_prefixed = cost_from_tokens(100, 50, "anthropic:claude-3-haiku")
        cost_plain = cost_from_tokens(100, 50, "claude-3-haiku")
        assert cost_prefixed[0] == pytest.approx(cost_plain[0], rel=1e-6)
        assert cost_prefixed[1] == pytest.approx(cost_plain[1], rel=1e-6)


class TestCostFromTokensUnknownModels:
    """Tests for unknown models (strict vs non-strict)."""

    def test_unknown_model_strict_raises(self) -> None:
        """Unknown model with strict=True raises UnknownModelError."""
        with pytest.raises(UnknownModelError, match="no pricing in litellm"):
            cost_from_tokens(100, 50, "totally-nonexistent-model-xyz")

    def test_unknown_model_nonstrict_returns_zero(self) -> None:
        """Unknown model with strict=False returns (0.0, 0.0)."""
        input_cost, output_cost = cost_from_tokens(
            100, 50, "totally-nonexistent-model-xyz", strict=False
        )
        assert input_cost == 0.0
        assert output_cost == 0.0

    def test_unknown_model_nonstrict_logs_warning(self, caplog) -> None:
        """Unknown model with strict=False logs a warning."""
        with caplog.at_level(logging.WARNING, logger="traigent.utils.cost_calculator"):
            cost_from_tokens(100, 50, "totally-nonexistent-model-xyz", strict=False)
        assert any(
            "Unknown model" in record.message and "zero cost" in record.message
            for record in caplog.records
        ), f"Expected warning about unknown model, got: {[r.message for r in caplog.records]}"

    def test_unknown_model_error_is_key_error(self) -> None:
        """UnknownModelError is a subclass of KeyError (backward compat)."""
        with pytest.raises(KeyError):
            cost_from_tokens(100, 50, "totally-nonexistent-model-xyz")


class TestCostFromTokensValidation:
    """Tests for input validation."""

    def test_negative_input_tokens_raises(self) -> None:
        """Negative input tokens raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            cost_from_tokens(-1, 50, "gpt-4o")

    def test_negative_output_tokens_raises(self) -> None:
        """Negative output tokens raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            cost_from_tokens(100, -1, "gpt-4o")

    def test_both_negative_raises(self) -> None:
        """Both negative tokens raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            cost_from_tokens(-5, -10, "gpt-4o")


class TestCostFromTokensLitellmUnavailable:
    """Tests for when litellm is not installed."""

    @patch("traigent.utils.cost_calculator.LITELLM_AVAILABLE", False)
    def test_litellm_unavailable_strict_raises(self) -> None:
        """strict=True raises RuntimeError when litellm is unavailable."""
        with pytest.raises(RuntimeError, match="litellm is required"):
            cost_from_tokens(100, 50, "gpt-4o", strict=True)

    @patch("traigent.utils.cost_calculator.LITELLM_AVAILABLE", False)
    def test_litellm_unavailable_nonstrict_returns_zero(self) -> None:
        """strict=False returns (0.0, 0.0) when litellm is unavailable."""
        input_cost, output_cost = cost_from_tokens(100, 50, "gpt-4o", strict=False)
        assert input_cost == 0.0
        assert output_cost == 0.0

    @patch("traigent.utils.cost_calculator.LITELLM_AVAILABLE", False)
    def test_litellm_unavailable_nonstrict_logs_warning(self, caplog) -> None:
        """strict=False logs warning when litellm is unavailable."""
        with caplog.at_level(logging.WARNING, logger="traigent.utils.cost_calculator"):
            cost_from_tokens(100, 50, "gpt-4o", strict=False)
        assert any(
            "litellm not available" in record.message for record in caplog.records
        )


class TestCostFromTokensEdgeCases:
    """Edge case tests."""

    def test_large_token_counts(self) -> None:
        """Large token counts produce proportionally large costs."""
        input_cost, output_cost = cost_from_tokens(1_000_000, 500_000, "gpt-4o")
        assert input_cost > 0
        assert output_cost > 0

    def test_returns_floats(self) -> None:
        """Return values are always Python floats."""
        input_cost, output_cost = cost_from_tokens(100, 50, "gpt-4o")
        assert isinstance(input_cost, float)
        assert isinstance(output_cost, float)

    def test_litellm_cost_per_token_exception_falls_through(self) -> None:
        """If litellm.cost_per_token raises, falls through to model_cost lookup."""
        mock_litellm = MagicMock()
        mock_litellm.cost_per_token.side_effect = Exception("test error")
        mock_litellm.model_cost = {
            "gpt-4o": {
                "input_cost_per_token": 2.5e-6,
                "output_cost_per_token": 10.0e-6,
            }
        }

        with (
            patch("traigent.utils.cost_calculator.litellm", mock_litellm),
            patch("traigent.utils.cost_calculator.LITELLM_AVAILABLE", True),
            patch(
                "traigent.utils.cost_calculator._is_model_known_to_litellm",
                return_value=False,
            ),
        ):
            input_cost, output_cost = cost_from_tokens(100, 50, "gpt-4o")
            assert input_cost == pytest.approx(100 * 2.5e-6)
            assert output_cost == pytest.approx(50 * 10.0e-6)
