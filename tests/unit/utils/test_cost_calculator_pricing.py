"""Tests for strict model pricing lookup in cost_calculator.py."""

from __future__ import annotations

import json

import pytest

import traigent.utils.cost_calculator as cc
from traigent.utils.cost_calculator import (
    MODEL_NAME_ALIASES,
    UnknownModelError,
    _build_model_candidates,
    get_model_token_pricing,
)


class TestGetModelTokenPricing:
    def test_returns_three_tuple_for_known_model(self) -> None:
        inp, out, method = get_model_token_pricing("gpt-4o")
        assert inp > 0
        assert out > 0
        assert method == "litellm"

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("claude-haiku", "claude-3-haiku-20240307"),
            ("claude-sonnet", "claude-3-5-sonnet-20241022"),
            ("claude-opus", "claude-3-opus-20240229"),
            ("claude-3.5-sonnet", "claude-3-5-sonnet-20241022"),
            ("claude-3.5-haiku", "claude-3-5-haiku-20241022"),
        ],
    )
    def test_builtin_claude_aliases_resolve(self, alias: str, canonical: str) -> None:
        alias_pricing = get_model_token_pricing(alias)
        canonical_pricing = get_model_token_pricing(canonical)
        assert alias_pricing[0] == pytest.approx(canonical_pricing[0], rel=1e-6)
        assert alias_pricing[1] == pytest.approx(canonical_pricing[1], rel=1e-6)
        assert alias_pricing[2] == "litellm"

    def test_provider_prefixed_model_resolves(self) -> None:
        prefixed = get_model_token_pricing("openai/gpt-4o")
        plain = get_model_token_pricing("gpt-4o")
        assert prefixed[0] == pytest.approx(plain[0], rel=1e-6)
        assert prefixed[1] == pytest.approx(plain[1], rel=1e-6)
        assert prefixed[2] == "litellm"

    @pytest.mark.parametrize(
        "name",
        [
            "OPENAI/GPT-4O",
            "OpenAI/gpt-4o",
            "Anthropic/claude-3-haiku-20240307",
        ],
    )
    def test_mixed_case_provider_prefix_resolves(self, name: str) -> None:
        """Weird-case provider prefixes must resolve, not raise."""
        inp, out, method = get_model_token_pricing(name)
        assert inp > 0
        assert out > 0
        assert method == "litellm"

    def test_unknown_model_raises_with_actionable_message(self) -> None:
        with pytest.raises(UnknownModelError, match="has no known pricing"):
            get_model_token_pricing("totally-unknown-model-abc")

    def test_custom_pricing_json_override(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
            json.dumps(
                {
                    "custom-estimator-model": {
                        "input_cost_per_token": 8e-6,
                        "output_cost_per_token": 9e-6,
                    }
                }
            ),
        )
        monkeypatch.delenv("TRAIGENT_CUSTOM_MODEL_PRICING_FILE", raising=False)
        cc._CUSTOM_PRICING_CACHE = None
        cc._CUSTOM_PRICING_CACHE_KEY = None

        inp, out, method = get_model_token_pricing("custom-estimator-model")
        assert inp == pytest.approx(8e-6)
        assert out == pytest.approx(9e-6)
        assert method == "custom_pricing"


class TestBuildModelCandidates:
    """Smoke tests for the candidate chain itself.

    These verify that alias resolution produces the right dated model IDs
    independently of litellm's bundled pricing database, so they are not
    gated on LITELLM_LOCAL_MODEL_COST_MAP or TRAIGENT_OFFLINE_MODE.
    """

    @pytest.mark.parametrize(
        ("raw_input", "expected_candidate"),
        [
            ("claude-sonnet", "claude-3-5-sonnet-20241022"),
            ("claude-haiku", "claude-3-haiku-20240307"),
            ("claude-opus", "claude-3-opus-20240229"),
            ("claude-3.5-sonnet", "claude-3-5-sonnet-20241022"),
            ("claude-3.5-haiku", "claude-3-5-haiku-20241022"),
            ("OPENAI/GPT-4O", "GPT-4O"),  # normalized strips prefix
        ],
    )
    def test_candidate_chain_contains_expected(
        self, raw_input: str, expected_candidate: str
    ) -> None:
        candidates = _build_model_candidates(raw_input)
        assert any(
            c.lower() == expected_candidate.lower() for c in candidates
        ), f"Expected '{expected_candidate}' in candidates {candidates}"

    def test_candidates_are_deduplicated(self) -> None:
        candidates = _build_model_candidates("gpt-4o")
        assert len(candidates) == len(set(candidates))

    def test_lowered_candidate_present_for_uppercase_input(self) -> None:
        """Mixed-case input must produce a lowered candidate for litellm lookup."""
        candidates = _build_model_candidates("OPENAI/GPT-4O")
        assert "gpt-4o" in candidates


class TestAliasCoverageSmoke:
    """Smoke coverage for the full built-in alias table.

    This intentionally runs under the test suite's bundled litellm pricing setup
    to verify every built-in alias still resolves to a priced model ID in the
    pinned local cost map.
    """

    @pytest.mark.parametrize("alias", sorted(MODEL_NAME_ALIASES))
    def test_all_builtin_aliases_resolve_in_local_cost_map(self, alias: str) -> None:
        inp, out, method = get_model_token_pricing(alias)
        assert inp >= 0.0
        assert out >= 0.0
        assert method == "litellm"
