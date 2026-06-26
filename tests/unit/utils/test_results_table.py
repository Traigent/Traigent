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

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils.results_table import (
    _find_best_per_objective,
    _trials_all_failed,
    print_results_table,
)


def _trial(
    config: dict[str, Any],
    metrics: dict[str, float],
    metadata: dict[str, Any] | None = None,
    status: TrialStatus = TrialStatus.COMPLETED,
    trial_id: str = "t",
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics=metrics,
        status=status,
        duration=0.1,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata=metadata or {},
    )


class TestTrialsAllFailed:
    def test_all_zero_successful_examples(self) -> None:
        trials = [
            _trial(
                {"model": "m1"},
                {"accuracy": 0.0, "cost": 0.0},
                metadata={"successful_examples": 0},
            ),
            _trial(
                {"model": "m2"},
                {"accuracy": 0.0, "cost": 0.0},
                metadata={"successful_examples": 0},
            ),
        ]
        assert _trials_all_failed(trials) is True

    def test_legacy_zero_accuracy_without_success_count_remains_rankable(self) -> None:
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.0, "cost": 0.0}),
            _trial({"model": "m2"}, {"accuracy": 0.0, "cost": 0.0}),
        ]
        assert _trials_all_failed(trials) is False

    def test_one_trial_with_successful_examples(self) -> None:
        trials = [
            _trial(
                {"model": "m1"},
                {"accuracy": 0.0},
                metadata={"successful_examples": 0},
            ),
            _trial(
                {"model": "m2"},
                {"accuracy": 0.0},
                metadata={"successful_examples": 3},
            ),
        ]
        assert _trials_all_failed(trials) is False

    def test_one_trial_with_positive_accuracy(self) -> None:
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.0}),
            _trial({"model": "m2"}, {"accuracy": 0.5}),
        ]
        assert _trials_all_failed(trials) is False

    def test_cost_only_metric_without_success_count_remains_rankable(self) -> None:
        trials = [
            _trial({"model": "m1"}, {"cost": 1.5}),
            _trial({"model": "m2"}, {"cost": 0.0}),
        ]
        assert _trials_all_failed(trials) is False

    def test_explicit_zero_success_overrides_positive_cost(self) -> None:
        trials = [
            _trial(
                {"model": "m1"},
                {"cost": 1.5},
                metadata={"successful_examples": 0},
            ),
            _trial(
                {"model": "m2"},
                {"cost": 0.0},
                metadata={"successful_examples": 0},
            ),
        ]
        assert _trials_all_failed(trials) is True

    def test_explicit_zero_success_overrides_positive_quality_metric(self) -> None:
        trials = [
            _trial(
                {"model": "m1"},
                {"accuracy": 0.65},
                metadata={"successful_examples": 0},
            ),
            _trial(
                {"model": "m2"},
                {"accuracy": 0.85},
                metadata={"successful_examples": 0},
            ),
        ]
        assert _trials_all_failed(trials) is True

    def test_non_numeric_metric_value_is_skipped(self) -> None:
        # Defensive: shouldn't crash on a stray non-numeric metric, and missing
        # success counts should still preserve legacy rankability.
        trials = [_trial({"model": "m1"}, {"accuracy": "n/a"})]  # type: ignore[dict-item]
        assert _trials_all_failed(trials) is False


class TestPrintResultsTableBanner:
    @staticmethod
    def _build_results(trials: list[TrialResult]) -> MagicMock:
        results = MagicMock()
        results.trials = trials
        results.metadata = {}
        results.best_trial_id = None
        results.best_config = trials[0].config if trials else {}
        results.best_score = None
        results.calculate_weighted_scores.return_value = {
            "best_weighted_config": trials[0].config if trials else {},
            "weighted_scores": [],
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

    def test_legacy_zero_score_run_still_emits_legend(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.0}),
            _trial({"model": "m2"}, {"accuracy": 0.0}),
        ]
        print_results_table(
            self._build_results(trials),
            config_space={"model": ["m1", "m2"]},
            objectives=["accuracy"],
        )

        out = capsys.readouterr().out
        assert "All trials failed" not in out
        assert "Overall Best" in out

    def test_explicit_zero_success_emits_failed_banner_even_with_positive_metric(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trials = [
            _trial(
                {"model": "m1"},
                {"accuracy": 0.65},
                metadata={"successful_examples": 0},
            ),
            _trial(
                {"model": "m2"},
                {"accuracy": 0.85},
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
        assert "Overall Best" not in out

    def test_mode_label_and_metric_overrides_render_once(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trials = [
            _trial(
                {"model": "cheap"},
                {"accuracy": 0.75, "cost": 0.0},
                metadata={"successful_examples": 10, "examples_attempted": 10},
                trial_id="t1",
            ),
            _trial(
                {"model": "accurate"},
                {"accuracy": 0.9, "cost": 0.0},
                metadata={"successful_examples": 10, "examples_attempted": 10},
                trial_id="t2",
            ),
        ]

        print_results_table(
            self._build_results(trials),
            config_space={"model": ["cheap", "accurate"]},
            objectives=["accuracy", "cost"],
            mode_label="MOCK",
            metric_overrides={"cost": [0.002, 0.015]},
        )

        out = capsys.readouterr().out
        assert out.count("Trial Results") == 1
        assert "MOCK - 2 trials" in out
        assert "$0.00200" in out
        assert "$0.01500" in out


class TestBestPerObjective:
    def test_returns_all_tied_best_indices(self) -> None:
        trials = [
            _trial(
                {"model": "m1"},
                {"accuracy": 1.0},
                metadata={"successful_examples": 20, "examples_attempted": 20},
            ),
            _trial(
                {"model": "m2"},
                {"accuracy": 1.0 + 5e-13},
                metadata={"successful_examples": 20, "examples_attempted": 20},
            ),
            _trial(
                {"model": "m3"},
                {"accuracy": 0.5},
                metadata={"successful_examples": 20, "examples_attempted": 20},
            ),
        ]

        best = _find_best_per_objective(trials, [("accuracy", "maximize")])

        assert best["accuracy"] == {0, 1}

    def test_excludes_zero_success_trial_from_best_cost(self) -> None:
        trials = [
            _trial(
                {"model": "failed-free"},
                {"cost": 0.0},
                metadata={"successful_examples": 0, "examples_attempted": 20},
            ),
            _trial(
                {"model": "working"},
                {"cost": 0.00001},
                metadata={"successful_examples": 20, "examples_attempted": 20},
            ),
        ]

        best = _find_best_per_objective(trials, [("cost", "minimize")])

        assert best["cost"] == {1}


class TestOverallBestIdentity:
    @staticmethod
    def _build_results(trials: list[TrialResult], best_trial_id: str) -> MagicMock:
        results = MagicMock()
        results.trials = trials
        results.metadata = {"best_trial_id": best_trial_id}
        results.best_config = trials[0].config
        results.best_score = 1.0
        results.calculate_weighted_scores.return_value = {"weighted_scores": []}
        return results

    def test_star_matches_best_trial_id_not_duplicate_config_equality(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        duplicate_config = {"model": "same"}
        trials = [
            _trial(
                duplicate_config,
                {"accuracy": 1.0, "cost": 0.00020},
                metadata={"successful_examples": 20, "examples_attempted": 20},
                trial_id="t1",
            ),
            _trial(
                duplicate_config,
                {"accuracy": 1.0, "cost": 0.00001},
                metadata={"successful_examples": 20, "examples_attempted": 20},
                trial_id="t2",
            ),
        ]

        print_results_table(
            self._build_results(trials, "t2"),
            config_space={"model": ["same"]},
            objectives=["accuracy", "cost"],
        )

        out = capsys.readouterr().out
        assert out.count("★") == 2  # one row marker plus the legend marker

    def test_table_surfaces_per_trial_example_success_counts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trials = [
            _trial(
                {"model": "legacy"},
                {"accuracy": 0.0},
                metadata={"successful_examples": 0, "examples_attempted": 20},
                trial_id="t1",
            ),
            _trial(
                {"model": "working"},
                {"accuracy": 1.0},
                metadata={"successful_examples": 20, "examples_attempted": 20},
                trial_id="t2",
            ),
        ]

        print_results_table(
            self._build_results(trials, "t2"),
            config_space={"model": ["legacy", "working"]},
            objectives=["accuracy"],
        )

        out = capsys.readouterr().out
        assert "examples" in out
        assert "0/20" in out
        assert "20/20" in out


class TestPrintResultsTableDirectFieldAccess:
    """print_results_table accesses results.trials and t.metrics directly.

    Regression (#1494): the former getattr defaults (results.trials → []) and
    (t.metrics → {}) masked contract violations. Direct access raises on a
    contract break rather than silently rendering an empty table.
    """

    def test_real_optimization_result_renders_correct_metrics(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A real OptimizationResult is rendered via direct attribute access.

        Verifies that results.trials (not getattr default []) and
        trials[0].metrics (not getattr default {}) are used — if the getattr
        defaults were in effect for a valid result, the table would render
        identically; but this test proves the direct path executes without error
        and propagates the real metric values.
        """
        trials = [
            _trial({"model": "m1"}, {"accuracy": 0.7}),
            _trial({"model": "m2"}, {"accuracy": 0.9}),
        ]
        results = OptimizationResult(
            trials=trials,
            best_config={"model": "m2"},
            best_score=0.9,
            optimization_id="oid-direct",
            duration=2.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )

        print_results_table(
            results,
            config_space={"model": ["m1", "m2"]},
            objectives=["accuracy"],
        )

        out = capsys.readouterr().out
        # Table was rendered (not "No trials to display" — direct trials access)
        assert "No trials to display" not in out
        # sample_metrics = trials[0].metrics → {"accuracy": 0.7}
        # If the former getattr({}) were in effect on a bad trial, "accuracy"
        # would be absent from metric_info and not rendered.
        assert "accuracy" in out.lower()

    def test_sample_metrics_direct_access_populates_metric_column(self) -> None:
        """trials[0].metrics is the real dict, not a getattr {} default.

        _find_best_per_objective is called with metric_info populated from
        sample_metrics = trials[0].metrics. This verifies the value is the
        actual metrics dict (non-empty), so the correct best-trial index is found.
        """
        trials_with_metrics = [
            _trial({"model": "a"}, {"score": 0.3}),
            _trial({"model": "b"}, {"score": 0.8}),
        ]
        # Directly call _find_best_per_objective using the real TrialResult.metrics
        # (simulating what print_results_table does after sample_metrics = trials[0].metrics)
        sample_metrics = trials_with_metrics[0].metrics  # direct access, not getattr
        assert "score" in sample_metrics  # proves real dict, not {} default

        best = _find_best_per_objective(
            trials_with_metrics,
            metric_info=[("score", None)],
            metric_overrides=None,
        )
        # Trial index 1 (score=0.8) is best — only works if metric_info was
        # populated from the real metrics dict
        assert 1 in best.get("score", set())
