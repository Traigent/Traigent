"""#2701: Claude 5.5 models get bundled first-party prices when litellm lacks them."""

import pytest

litellm = pytest.importorskip("litellm")

from traigent.utils import cost_calculator as cc  # noqa: E402

# Anthropic first-party rates per 1M tokens (USD), from the Anthropic model
# pricing table cached 2026-10-06 (issue #2701). Cache-write prices are not
# listed for these models and are deliberately not invented here.
CLAUDE_55_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.00, 50.00),
}


def _without(model_cost, keys):
    return {k: v for k, v in model_cost.items() if k not in keys}


def test_each_claude_55_model_has_bundled_price_after_merge(monkeypatch):
    stripped = _without(litellm.model_cost, set(CLAUDE_55_PER_MTOK))
    monkeypatch.setattr(litellm, "model_cost", stripped)

    cc.ensure_bundled_provider_prices()

    for model, (input_mtok, output_mtok) in CLAUDE_55_PER_MTOK.items():
        assert model in litellm.model_cost, model
        assert cc.model_pricing_resolves(model), model
        input_cost, output_cost = cc.cost_from_tokens(1_000_000, 1_000_000, model)
        assert input_cost == pytest.approx(input_mtok, rel=1e-9), model
        assert output_cost == pytest.approx(output_mtok, rel=1e-9), model


def test_existing_litellm_entry_is_not_overwritten(monkeypatch):
    sentinel = {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}
    stripped = _without(litellm.model_cost, set(CLAUDE_55_PER_MTOK))
    stripped["claude-opus-5-5"] = sentinel
    monkeypatch.setattr(litellm, "model_cost", stripped)

    cc.ensure_bundled_provider_prices()

    assert litellm.model_cost["claude-opus-5-5"] is sentinel
    # The other ids are still filled in.
    assert "claude-sonnet-5-5" in litellm.model_cost


def test_merge_is_idempotent_for_claude_55(monkeypatch):
    stripped = _without(litellm.model_cost, set(CLAUDE_55_PER_MTOK))
    monkeypatch.setattr(litellm, "model_cost", stripped)

    cc.ensure_bundled_provider_prices()

    assert cc.ensure_bundled_provider_prices() == 0
