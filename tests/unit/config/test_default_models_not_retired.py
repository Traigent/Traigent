"""Guard: shipped examples, defaults and probes must not name retired models.

Regression for #2650 (default model tables).
"""

from __future__ import annotations

from pathlib import Path

from traigent.config.retired_models import (
    is_retired_model,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


_SHUT_DOWN_DEFAULTS = {"gemini-2.0-flash", "gpt-3.5-turbo"}


def test_default_model_tables_have_no_retired_models() -> None:
    from traigent.api.parameter_ranges import Choices
    from traigent.integrations import providers

    offenders: list[str] = []
    tables = list(providers._FALLBACK_MODELS.values())
    for tiers in providers._MODEL_TIERS.values():
        tables.extend(tiers.values())
    for models in tables:
        offenders += [m for m in models if is_retired_model(m)]
    for provider in (None, "openai", "anthropic", "gemini", "google"):
        for tier in ("fast", "balanced", "quality"):
            offenders += [
                m
                for m in Choices.model(provider=provider, tier=tier).values
                if is_retired_model(m)
            ]
    # Shut down by the provider (gemini-2.0-flash 2026-06-01) but still
    # accepted for explicit references, so not in the runtime denylist.
    # (gpt-3.5-turbo, retired 2026-10-23, is now in the denylist too.)
    offenders += [m for t in tables for m in t if m in _SHUT_DOWN_DEFAULTS]
    assert not offenders, sorted(set(offenders))


def test_choices_model_gemini_and_case_insensitive_provider(monkeypatch) -> None:
    from traigent.api.parameter_ranges import Choices

    for prov in ("GEMINI", "TRAIGENT_MODELS_DEFAULT"):
        for tier in ("FAST", "BALANCED", "QUALITY"):
            monkeypatch.delenv(f"TRAIGENT_MODELS_{prov}_{tier}", raising=False)
    for provider in ("gemini", "google", "Gemini", "Anthropic"):
        models = Choices.model(provider=provider, tier="balanced").to_list()
        assert models, provider
        assert not any(m.startswith("gpt-") for m in models), (provider, models)


def test_choices_model_unknown_provider_raises_naming_supported(monkeypatch) -> None:
    import pytest

    from traigent.api.parameter_ranges import Choices

    monkeypatch.delenv("TRAIGENT_MODELS_ANTROPIC_FAST", raising=False)
    with pytest.raises(ValueError) as excinfo:
        Choices.model(provider="antropic", tier="fast")
    message = str(excinfo.value)
    for supported in ("openai", "anthropic", "gemini"):
        assert supported in message
    assert "TRAIGENT_MODELS_ANTROPIC_FAST" in message


def test_embedding_model_is_case_insensitive_and_never_falls_back() -> None:
    import pytest

    from traigent.api.parameter_ranges import Choices

    assert (
        Choices.embedding_model(provider="OpenAI").to_list()
        == Choices.embedding_model(provider="openai").to_list()
    )
    assert Choices.embedding_model().to_list()  # provider=None keeps common set
    for provider in ("gemini", "Anthropic", "antropic"):
        with pytest.raises(ValueError, match="openai"):
            Choices.embedding_model(provider=provider)


def test_pricing_report_lists_no_retired_models() -> None:
    from traigent.analytics.intelligence import CostOptimizationAI

    report = CostOptimizationAI().fetch_current_pricing(["openai", "anthropic"])
    listed = [m for models in report.values() for m in models]
    assert listed
    assert not [m for m in listed if is_retired_model(m)], listed
    # The cloud bridge default is pinned in tests/unit/cloud/test_backend_bridges.py.


def test_pricing_report_models_have_nonzero_prices() -> None:
    """A pricing-data drop for a reported model must not be silent."""
    from traigent.analytics.intelligence import CostOptimizationAI
    from traigent.utils.cost_calculator import get_model_pricing_per_1k

    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-4o"):
        input_price, output_price = get_model_pricing_per_1k(model)
        assert input_price > 0 and output_price > 0, model

    report = CostOptimizationAI().fetch_current_pricing(["openai"])["openai"]
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-4o"):
        assert model in report, (model, sorted(report))
        assert report[model]["input"] > 0 and report[model]["output"] > 0, model


def test_choices_model_rejects_non_string_provider_and_quotes_input() -> None:
    import pytest

    from traigent.api.parameter_ranges import Choices

    for call in (Choices.model, Choices.embedding_model):
        with pytest.raises(ValueError, match="provider must be a string"):
            call(provider=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"'x\\nTRAIGENT_FAKE'"):
        Choices.embedding_model(provider="x\nTRAIGENT_FAKE")
    with pytest.raises(ValueError, match=r"TRAIGENT_MODELS_ZZ_FAST'"):
        Choices.model(provider="zz", tier="fast")


def test_choices_empty_provider_raises_but_none_is_default(monkeypatch) -> None:
    import pytest

    from traigent.api.parameter_ranges import Choices

    for tier in ("FAST", "BALANCED", "QUALITY"):
        monkeypatch.delenv(f"TRAIGENT_MODELS_DEFAULT_{tier}", raising=False)
    assert Choices.model(provider=None).to_list()
    assert Choices.embedding_model(provider=None).to_list()
    with pytest.raises(ValueError, match="Unknown provider ''"):
        Choices.model(provider="")
    with pytest.raises(ValueError, match="provider ''"):
        Choices.embedding_model(provider="")


def test_choices_model_unknown_tier_is_not_blamed_on_provider(monkeypatch) -> None:
    import pytest

    from traigent.api.parameter_ranges import Choices

    monkeypatch.delenv("TRAIGENT_MODELS_OPENAI_BOGUS", raising=False)
    with pytest.raises(ValueError) as excinfo:
        Choices.model(provider="openai", tier="bogus")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "Unknown tier 'bogus'" in message
    for tier in ("fast", "balanced", "quality"):
        assert tier in message
    assert "provider" not in message.lower().replace("explicit model list", "")
    with pytest.raises(ValueError, match="Unknown provider 'nope'"):
        Choices.model(provider="nope", tier="fast")


def test_choices_model_non_string_tier_raises_value_error() -> None:
    import pytest

    from traigent.api.parameter_ranges import Choices

    with pytest.raises(ValueError, match="tier must be a string"):
        Choices.model(provider="openai", tier=3)  # type: ignore[arg-type]


def test_empty_provider_error_has_no_malformed_env_hint() -> None:
    import pytest

    from traigent.api.parameter_ranges import Choices

    with pytest.raises(ValueError) as excinfo:
        Choices.model(provider="")
    assert "TRAIGENT_MODELS__" not in str(excinfo.value)
