"""Regression tests for hooks price lookup and init allowlist (#2649)."""

# Traceability: CONC-Quality-Reliability REQ-ORCH-003

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from traigent.config.retired_models import is_retired_model
from traigent.hooks.config import create_default_config
from traigent.hooks.validator import MODEL_COST_PER_1K, AgentValidator


def test_default_init_allowlist_has_no_retired_or_unpriced_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = create_default_config(tmp_path / "traigent.yml")
    allowed = yaml.safe_load(path.read_text())["constraints"]["allowed_models"]
    assert allowed
    for model in allowed:
        assert not is_retired_model(model), model
        assert AgentValidator._lookup_cost_per_1k(model) is not None, model


@pytest.mark.parametrize("name", ["gpt", "claude", "g", "gpt-5", "o"])
def test_loose_substrings_are_not_priced(name: str) -> None:
    assert AgentValidator._lookup_cost_per_1k(name) is None


@pytest.mark.parametrize(
    ("name", "expected_key"),
    [
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("gpt-4-turbo-2024-04-09", "gpt-4-turbo"),
        ("GPT-4O", "gpt-4o"),
    ],
)
def test_versioned_names_use_longest_known_prefix(name: str, expected_key: str) -> None:
    assert AgentValidator._lookup_cost_per_1k(name) == MODEL_COST_PER_1K[expected_key]


def test_default_init_allowlist_uses_current_sdk_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = create_default_config(tmp_path / "traigent.yml")
    allowed = yaml.safe_load(path.read_text())["constraints"]["allowed_models"]
    assert "claude-haiku-4-5-20251001" in allowed
    assert "claude-sonnet-4-6" in allowed


def test_haiku_45_snapshot_is_not_priced_as_haiku_3() -> None:
    haiku45 = AgentValidator._lookup_cost_per_1k("claude-haiku-4-5-20251001")
    assert haiku45 is not None
    # Haiku 4.5: $1/M in, $5/M out -> blended $3/M -> 0.003 per 1K
    assert haiku45 == pytest.approx(0.003)
    assert haiku45 != MODEL_COST_PER_1K["claude-haiku"]
    assert AgentValidator._lookup_cost_per_1k("claude-sonnet-4-6") == pytest.approx(
        0.009
    )


def test_dated_snapshot_of_same_id_is_priced() -> None:
    assert AgentValidator._lookup_cost_per_1k(
        "claude-haiku-4-5-20260101"
    ) == AgentValidator._lookup_cost_per_1k("claude-haiku-4-5")


@pytest.mark.parametrize(
    "name",
    [
        "claude-haiku-5-5",  # different generation, must not match bare family alias
        "claude-haiku-5-5-20270101",
        "claude-sonnet-9",
        "gpt-4-turbo-preview",  # not a dated snapshot
    ],
)
def test_other_generations_and_non_dated_suffixes_are_unpriced(name: str) -> None:
    assert AgentValidator._lookup_cost_per_1k(name) is None
