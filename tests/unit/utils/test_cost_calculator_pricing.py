"""Tests for strict model pricing lookup in cost_calculator.py."""

from __future__ import annotations

import json

import pytest

import traigent.utils.cost_calculator as cc
from traigent.utils.cost_calculator import UnknownModelError, get_model_token_pricing


class TestGetModelTokenPricing:
    def test_returns_three_tuple_for_known_model(self) -> None:
        inp, out, method = get_model_token_pricing("gpt-4o")
        assert inp > 0
        assert out > 0
        assert method == "litellm"

    def test_provider_prefixed_model_resolves(self) -> None:
        prefixed = get_model_token_pricing("openai/gpt-4o")
        plain = get_model_token_pricing("gpt-4o")
        assert prefixed[0] == pytest.approx(plain[0], rel=1e-6)
        assert prefixed[1] == pytest.approx(plain[1], rel=1e-6)
        assert prefixed[2] == "litellm"

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
