"""Guard: shipped examples, defaults and probes must not name retired models.

Regression for #2646 (provider examples manifest, prefixed/Anthropic denylist forms).
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from traigent.config.retired_models import (
    RETIRED_MODEL_IDS,
    is_retired_model,
    normalize_model_id,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "model_id",
    [
        "claude-3-haiku-20240307",
        "claude-3-7-sonnet-20250219",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3.5-sonnet",
        "openrouter/anthropic/claude-3.5-sonnet",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-pro",
        "vertex_ai/gemini-1.5-flash",
        "vertex_ai/gemini-1.5-pro",
    ],
)
def test_retired_models_recognises_prefixed_and_anthropic_forms(model_id: str) -> None:
    assert is_retired_model(model_id), model_id


def test_normalize_keeps_current_models_untouched() -> None:
    for current in ("claude-sonnet-4-6", "gpt-4o-mini", "gemini-3.8-flash"):
        assert normalize_model_id(current) == current
        assert not is_retired_model(current)
        assert not is_retired_model(f"openai/{current}")


def test_provider_manifest_has_no_retired_models() -> None:
    manifest = json.loads(
        (resources.files("traigent.examples.providers") / "manifest.json").read_text()
    )
    offenders = [
        (p["id"], m)
        for p in manifest["providers"]
        for m in p.get("models", [])
        if is_retired_model(m)
    ]
    assert not offenders, offenders


def test_denylist_still_lists_anthropic_3_family_core() -> None:
    assert "claude-3-haiku-20240307" in RETIRED_MODEL_IDS
