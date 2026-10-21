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
    # Shut down by the provider (gemini-2.0-flash 2026-06-01) or within days
    # (gpt-3.5-turbo 2026-10-23) but still accepted for explicit references,
    # so not in the runtime denylist.
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
