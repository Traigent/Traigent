"""Tests for model tier classification and pricing lookup (cost_calculator.py).

Tests the get_model_token_pricing() function and _classify_model_tier() helper
added for Phase 4 model-aware cost estimation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.utils.cost_calculator import (
    _TIER_CHEAP,
    _TIER_EXPENSIVE,
    _TIER_MID,
    _classify_model_tier,
    get_model_token_pricing,
)

# ---------------------------------------------------------------------------
# _classify_model_tier
# ---------------------------------------------------------------------------


class TestClassifyModelTier:
    """Tier classification with ordered regex matching."""

    @pytest.mark.parametrize(
        "model,expected_tier",
        [
            # GPT-4 family — ordering is critical
            ("gpt-4o-mini", "cheap"),
            ("gpt-4-mini", "cheap"),
            ("gpt-4-turbo", "expensive"),
            ("gpt-4-turbo-2024-04-09", "expensive"),
            ("gpt-4o", "mid"),
            ("gpt-4o-2024-08-06", "mid"),
            ("gpt-4", "expensive"),
            ("gpt-4-0613", "expensive"),
            ("gpt-3.5-turbo", "cheap"),
            ("gpt-3.5-turbo-0125", "cheap"),
            # Anthropic models
            ("claude-3-opus-20240229", "expensive"),
            ("claude-opus-4-20250514", "expensive"),
            ("claude-3-5-sonnet-20241022", "mid"),
            ("claude-sonnet-4-20250514", "mid"),
            ("claude-3-haiku-20240307", "cheap"),
            ("claude-3-5-haiku-20241022", "cheap"),
            # Generic patterns
            ("gemini-1.5-flash", "cheap"),
            ("gemini-2.0-flash", "cheap"),
            ("gemini-1.5-pro", "mid"),
            ("deepseek-chat-mini", "cheap"),
            ("some-nano-model", "cheap"),
            # Default (unknown)
            ("totally-unknown-model-xyz", "mid"),
        ],
    )
    def test_tier_classification(self, model: str, expected_tier: str) -> None:
        assert _classify_model_tier(model) == expected_tier

    def test_regex_ordering_lock_in_gpt4o_mini_before_gpt4o(self) -> None:
        """gpt-4o-mini must be CHEAP, not MID (\\b treats - as boundary).

        This test locks the ordering so future refactors can't silently
        reorder rules and break classification.
        """
        assert _classify_model_tier("gpt-4o-mini") == "cheap"
        assert _classify_model_tier("gpt-4o") == "mid"
        # Verify they differ
        assert _classify_model_tier("gpt-4o-mini") != _classify_model_tier("gpt-4o")

    def test_gpt4_stays_expensive_not_downgraded(self) -> None:
        """gpt-4 must classify as EXPENSIVE, not MID.

        EXACT_MODEL_MAPPING maps gpt-4 -> gpt-4o in runtime cost calculation,
        which would downgrade it. Tier classification intentionally skips
        EXACT_MODEL_MAPPING for conservative pre-estimation.
        """
        assert _classify_model_tier("gpt-4") == "expensive"

    def test_case_insensitive(self) -> None:
        assert _classify_model_tier("GPT-4-TURBO") == "expensive"
        assert _classify_model_tier("Claude-3-Opus") == "expensive"


# ---------------------------------------------------------------------------
# get_model_token_pricing
# ---------------------------------------------------------------------------


class TestGetModelTokenPricing:
    def test_returns_three_tuple(self) -> None:
        """Return type is (input, output, method) with positive costs."""
        inp, out, method = get_model_token_pricing("gpt-4o")
        assert inp > 0
        assert out > 0
        assert isinstance(method, str)

    def test_heuristic_fallback_for_unknown_model(self) -> None:
        """Completely unknown model → heuristic mid-tier."""
        inp, out, method = get_model_token_pricing("totally-unknown-model-abc")
        assert method == "heuristic:mid"
        assert inp == pytest.approx(_TIER_MID["input"])
        assert out == pytest.approx(_TIER_MID["output"])

    def test_litellm_zero_falls_through(self) -> None:
        """When litellm returns (0, 0), should NOT treat as free."""
        import traigent.utils.cost_calculator as _mod

        mock_litellm = MagicMock()
        mock_litellm.cost_per_token.return_value = (0.0, 0.0)
        with (
            patch.object(_mod, "_is_model_known_to_litellm", return_value=True),
            patch.object(_mod, "litellm", mock_litellm),
        ):
            inp, out, method = get_model_token_pricing("free-model-test")
            # Should NOT return litellm method — should fall through
            assert method != "litellm"
            assert inp > 0 or out > 0  # Non-zero from fallback/heuristic

    def test_litellm_positive_is_used(self) -> None:
        """When litellm returns positive pricing, use it."""
        import traigent.utils.cost_calculator as _mod

        mock_litellm = MagicMock()
        mock_litellm.cost_per_token.return_value = (5e-6, 15e-6)
        with (
            patch.object(_mod, "_is_model_known_to_litellm", return_value=True),
            patch.object(_mod, "litellm", mock_litellm),
        ):
            inp, out, method = get_model_token_pricing("test-model")
            assert method == "litellm"
            assert inp == pytest.approx(5e-6)
            assert out == pytest.approx(15e-6)

    def test_litellm_exception_falls_through(self) -> None:
        """When litellm throws, falls through to fallback/heuristic."""
        import traigent.utils.cost_calculator as _mod

        mock_litellm = MagicMock()
        mock_litellm.cost_per_token.side_effect = RuntimeError("boom")
        with (
            patch.object(_mod, "_is_model_known_to_litellm", return_value=True),
            patch.object(_mod, "litellm", mock_litellm),
        ):
            inp, out, method = get_model_token_pricing("test-model")
            assert method != "litellm"

    def test_fallback_dict_match(self) -> None:
        """Model in ESTIMATION_MODEL_PRICING → uses fallback_dict method."""
        # gpt-4-turbo is in ESTIMATION_MODEL_PRICING
        # Disable litellm to force fallback path
        with patch("traigent.utils.cost_calculator.LITELLM_AVAILABLE", False):
            inp, out, method = get_model_token_pricing("gpt-4-turbo")
            assert method == "fallback_dict"
            assert inp > 0
            assert out > 0

    def test_expensive_tier_pricing_values(self) -> None:
        assert _TIER_EXPENSIVE["input"] == pytest.approx(10.0e-6)
        assert _TIER_EXPENSIVE["output"] == pytest.approx(30.0e-6)

    def test_mid_tier_pricing_values(self) -> None:
        assert _TIER_MID["input"] == pytest.approx(3.0e-6)
        assert _TIER_MID["output"] == pytest.approx(15.0e-6)

    def test_cheap_tier_pricing_values(self) -> None:
        assert _TIER_CHEAP["input"] == pytest.approx(0.25e-6)
        assert _TIER_CHEAP["output"] == pytest.approx(1.25e-6)


# ---------------------------------------------------------------------------
# _fallback_cost_from_tokens _quiet parameter
# ---------------------------------------------------------------------------


class TestFallbackQuietParameter:
    def test_quiet_does_not_raise(self) -> None:
        """_quiet=True should not change return values."""
        from traigent.utils.cost_calculator import _fallback_cost_from_tokens

        # Known model in fallback dict
        cost_normal = _fallback_cost_from_tokens("gpt-4o-quiet-test-1", 100, 50)
        cost_quiet = _fallback_cost_from_tokens(
            "gpt-4o-quiet-test-2", 100, 50, _quiet=True
        )
        # Both should return the same values (same pricing, different model keys for warn-once)
        assert cost_normal == cost_quiet
