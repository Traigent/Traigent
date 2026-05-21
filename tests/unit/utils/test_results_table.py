"""Tests for the trial-results table renderer.

Regression: when every trial produced zero successful examples (e.g. all
provider calls failed with a 404 because of a deprecated model ID), the
table must NOT crown a winner with the ★ Overall Best framing. Instead it
should emit a clear "all trials failed" banner.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.results_table import _trials_all_failed, print_results_table


def _trial(
    config: dict[str, Any],
    metrics: dict[str, float],
    metadata: dict[str, Any] | None = None,
    status: TrialStatus = TrialStatus.COMPLETED,
) -> TrialResult:
    return TrialResult(
        trial_id="t",
        config=config,
        metrics=metrics,
        status=status,
        duration=0.1,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata=metadata or {},
    )


class TestTrialsAllFailed:
    def test_all_zero_accuracy_no_successful_examples(self) -> None:
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.0, "cost": 0.0}),
            _trial({"model": "m2"}, {"accuracy": 0.0, "cost": 0.0}),
        ]
        assert _trials_all_failed(trials) is True

    def test_one_trial_with_successful_examples(self) -> None:
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.0}, metadata={"successful_examples": 0}),
            _trial({"model": "m2"}, {"accuracy": 0.0}, metadata={"successful_examples": 3}),
        ]
        assert _trials_all_failed(trials) is False

    def test_one_trial_with_positive_accuracy(self) -> None:
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.0}),
            _trial({"model": "m2"}, {"accuracy": 0.5}),
        ]
        assert _trials_all_failed(trials) is False

    def test_cost_only_metric_does_not_imply_success(self) -> None:
        # A trial whose only positive metric is "cost" or "latency" still failed —
        # we don't want to treat $0 cost as a success signal either way; the
        # check ignores cost/latency entirely.
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.0, "cost": 1.5}),
            _trial({"model": "m2"}, {"accuracy": 0.0, "cost": 0.0}),
        ]
        assert _trials_all_failed(trials) is True

    def test_non_numeric_metric_value_is_skipped(self) -> None:
        # Defensive: shouldn't crash on a stray non-numeric metric.
        trials = [_trial({"model": "m1"}, {"accuracy": "n/a"})]  # type: ignore[dict-item]
        assert _trials_all_failed(trials) is True


class TestPrintResultsTableBanner:
    @staticmethod
    def _build_results(trials: list[TrialResult]) -> MagicMock:
        results = MagicMock()
        results.trials = trials
        results.calculate_weighted_scores.return_value = {
            "best_weighted_config": trials[0].config if trials else {},
        }
        return results

    def test_emits_failed_banner_when_no_trial_succeeded(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trials = [
            _trial(
                {"model": "m1"},
                {"accuracy": 0.0},
                metadata={"successful_examples": 0},
            ),
            _trial(
                {"model": "m2"},
                {"accuracy": 0.0},
                metadata={"successful_examples": 0},
            ),
        ]
        print_results_table(
            self._build_results(trials),
            config_space={"model": ["m1", "m2"]},
            objectives=["accuracy"],
        )

        out = capsys.readouterr().out
        assert "All trials failed" in out
        assert "no examples succeeded" in out
        # The legend line "Overall Best" must NOT appear when everything failed.
        assert "Overall Best" not in out
        # The ★ marker must not appear on any row.
        assert "★" not in out

    def test_emits_legend_when_at_least_one_trial_succeeded(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.0}),
            _trial({"model": "m2"}, {"accuracy": 0.5}),
        ]
        print_results_table(
            self._build_results(trials),
            config_space={"model": ["m1", "m2"]},
            objectives=["accuracy"],
        )

        out = capsys.readouterr().out
        assert "All trials failed" not in out
        assert "Overall Best" in out
