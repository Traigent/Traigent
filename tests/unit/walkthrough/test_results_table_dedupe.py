"""Regression coverage for walkthrough results-table de-duplication."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from walkthrough.utils import helpers
from walkthrough.utils.helpers import build_results_table_callback


def _trial(
    config: dict[str, Any],
    metrics: dict[str, float],
    metadata: dict[str, Any] | None = None,
    trial_id: str = "t",
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata=metadata or {},
    )


def _result(trials: list[TrialResult]) -> MagicMock:
    result = MagicMock()
    result.trials = trials
    result.metadata = {}
    result.best_trial_id = None
    result.best_config = trials[0].config
    result.best_score = None
    result.calculate_weighted_scores.return_value = {"weighted_scores": []}
    return result


def test_walkthrough_helper_no_longer_exports_forked_renderer() -> None:
    assert not hasattr(helpers, "print_results_table")
    for forked_name in (
        "_format_config_value",
        "_format_metric_value",
        "_find_best_per_objective",
        "_find_best_trial_index",
        "_trial_is_best_candidate",
    ):
        assert not hasattr(helpers, forked_name)


def test_walkthrough_callback_prints_single_table_and_uses_failed_guard(
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = build_results_table_callback(is_mock=False)
    callback.on_optimization_start({"model": ["m1", "m2"]}, ["accuracy"], "grid")
    callback.on_optimization_complete(
        _result(
            [
                _trial(
                    {"model": "m1"},
                    {"accuracy": 0.0},
                    metadata={"successful_examples": 0, "examples_attempted": 2},
                    trial_id="t1",
                ),
                _trial(
                    {"model": "m2"},
                    {"accuracy": 0.0},
                    metadata={"successful_examples": 0, "examples_attempted": 2},
                    trial_id="t2",
                ),
            ]
        )
    )

    out = capsys.readouterr().out
    assert out.count("Trial Results") == 1
    assert "REAL - 2 trials" in out
    assert "All trials failed" in out
    assert "Overall Best" not in out
