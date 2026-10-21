"""#1808: a default_config baseline trial counts against max_trials."""

from __future__ import annotations

import pytest

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE", "1")
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD", raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))


def _run(default_config, max_trials):
    seen: list[str] = []

    @traigent.optimize(
        eval_dataset=Dataset([EvaluationExample({"text": "q"}, "x")], name="w"),
        objectives=["accuracy"],
        scoring_function=lambda p, e: 1.0,
        configuration_space={"m": ["a", "b"]},
        default_config=default_config,
        algorithm="grid",
        execution={"offline": True},
        injection_mode="parameter",
        config_param="cfg",
    )
    def agent(text, cfg=None):
        seen.append(cfg["m"])
        return "x"

    agent.optimize_sync(max_trials=max_trials)
    return seen


def test_grid_without_default_covers_whole_grid():
    assert sorted(set(_run(None, 2))) == ["a", "b"]


def test_default_config_baseline_with_tight_max_trials_warns(caplog):
    with caplog.at_level("WARNING", logger="traigent.core.orchestrator"):
        _run({"m": "a"}, 2)

    assert "will not be fully evaluated" in caplog.text
    assert "max_trials >= 3" in caplog.text


def test_default_config_baseline_with_enough_max_trials_does_not_warn(caplog):
    with caplog.at_level("WARNING", logger="traigent.core.orchestrator"):
        _run({"m": "a"}, 3)

    assert "will not be fully evaluated" not in caplog.text


def test_no_default_config_never_warns(caplog):
    with caplog.at_level("WARNING", logger="traigent.core.orchestrator"):
        _run(None, 2)

    assert "will not be fully evaluated" not in caplog.text
