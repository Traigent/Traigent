"""Guard: the shipped model catalog must agree with the retired-model denylist.

Regression for #2840: ``retired_models.py`` documented Claude Sonnet 3.5 as
retired while ``models.yaml`` still shipped its dated and ``-latest`` ids as
current, and a plugin preset used one as its "balanced" fallback.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from traigent.config.provider_support import load_models_yaml
from traigent.config.retired_models import is_retired_model
from traigent.integrations.model_discovery.anthropic_discovery import (
    KNOWN_ANTHROPIC_MODELS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PRESET_FILE = (
    REPO_ROOT
    / "plugins"
    / "traigent-tuned-variables"
    / "traigent_tuned_variables"
    / "presets"
    / "llm.py"
)


@pytest.mark.parametrize(
    "model_id",
    [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ],
)
def test_sonnet_3_5_ids_are_retired(model_id: str) -> None:
    assert is_retired_model(model_id), model_id


def test_models_yaml_known_models_are_not_retired() -> None:
    config = load_models_yaml()
    assert config, "models.yaml failed to load"
    offenders = [
        (provider, model)
        for provider, entry in config.items()
        for model in (entry or {}).get("known_models", [])
        if is_retired_model(model)
    ]
    assert not offenders, offenders


def test_anthropic_discovery_snapshot_is_not_retired() -> None:
    offenders = [m for m in KNOWN_ANTHROPIC_MODELS if is_retired_model(m)]
    assert not offenders, offenders


def test_tuned_variables_preset_fallbacks_are_not_retired() -> None:
    spec = importlib.util.spec_from_file_location("_tv_presets_llm", PRESET_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    offenders = [
        (provider, tier, model)
        for provider in (None, "openai", "anthropic")
        for tier in ("fast", "balanced", "quality")
        for model in module._get_fallback_models(provider, tier)
        if is_retired_model(model)
    ]
    assert not offenders, offenders
