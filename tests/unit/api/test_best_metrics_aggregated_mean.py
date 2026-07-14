"""Regression tests: ``OptimizationResult.best_metrics`` must share ``best_score``'s
basis in aggregated/hybrid runs (finding T4).

In aggregated/hybrid mode the winner is chosen over config-aggregated MEAN
metrics and ``best_score`` is the winning config's mean of the primary
objective. The old ``best_metrics`` returned a SINGLE replicate's raw metrics
(the first trial whose ``config == best_config``), so ``best_config.json``
contradicted itself and ``export_config`` baked a single-replicate value into
the deployable config.

These tests fail on the pre-fix implementation (which returns the first
replicate) and pass on the fixed one (which returns the replicate MEAN). Local
single-trial runs must stay unchanged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.result_selection import select_best_configuration


def _make_trial(
    trial_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    status: TrialStatus = TrialStatus.COMPLETED,
    successful_examples: int = 10,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics=metrics,
        status=status,
        duration=0.1,
        timestamp=datetime.now(),
        metadata={"successful_examples": successful_examples, "examples_attempted": 10},
    )


def _make_result(
    trials: list[TrialResult],
    best_config: dict[str, Any],
    best_score: float | None,
    objectives: list[str],
    *,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config=best_config,
        best_score=best_score,
        optimization_id="opt_test",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives,
        algorithm="random",
        timestamp=datetime.now(),
        metadata=metadata or {},
    )


def _aggregated_metadata(samples: int) -> dict[str, Any]:
    """Mirror the session_summary that result_selection stamps for aggregated runs."""
    return {
        "session_summary": {
            "selection_mode": "aggregated_mean",
            "primary_objective": "accuracy",
            "samples_per_config": {"cfg": samples},
        }
    }


class TestAggregatedBestMetricsAreTheReplicateMean:
    def test_primary_metric_equals_best_score_mean_not_first_replicate(self):
        """Winner has 2 replicates with differing accuracy; best_score is the mean.

        Pre-fix: best_metrics["accuracy"] == 0.80 (first replicate) != best_score.
        Post-fix: best_metrics["accuracy"] == 0.85 (mean) == best_score.
        """
        cfg = {"model": "gpt-4"}
        # accuracy replicates 0.80 and 0.90 -> mean 0.85 (== best_score below)
        rep1 = _make_trial("r1", cfg, {"accuracy": 0.80, "cost": 0.02, "latency": 1.0})
        rep2 = _make_trial("r2", cfg, {"accuracy": 0.90, "cost": 0.04, "latency": 3.0})
        result = _make_result(
            trials=[rep1, rep2],
            best_config=cfg,
            best_score=0.85,
            objectives=["accuracy"],
            metadata=_aggregated_metadata(samples=2),
        )

        bm = result.best_metrics

        assert bm["accuracy"] == pytest.approx(
            0.85
        ), "primary must be the replicate MEAN (0.85), not the first replicate (0.80)"
        assert bm["accuracy"] == pytest.approx(
            result.best_score
        ), "best_metrics[primary] must equal best_score in aggregated runs"

    def test_secondary_metrics_equal_the_replicate_mean(self):
        """Secondary cost/latency must be the mean, matching the session_summary mean."""
        cfg = {"model": "gpt-4"}
        rep1 = _make_trial("r1", cfg, {"accuracy": 0.80, "cost": 0.02, "latency": 1.0})
        rep2 = _make_trial("r2", cfg, {"accuracy": 0.90, "cost": 0.04, "latency": 3.0})
        result = _make_result(
            trials=[rep1, rep2],
            best_config=cfg,
            best_score=0.85,
            objectives=["accuracy"],
            metadata=_aggregated_metadata(samples=2),
        )

        bm = result.best_metrics

        assert bm["cost"] == pytest.approx(0.03), "cost must be mean(0.02, 0.04)"
        assert bm["latency"] == pytest.approx(2.0), "latency must be mean(1.0, 3.0)"

    def test_detects_aggregation_via_samples_per_config_only(self):
        """samples_per_config alone (no explicit selection_mode) still triggers mean."""
        cfg = {"model": "gpt-4"}
        rep1 = _make_trial("r1", cfg, {"accuracy": 0.60})
        rep2 = _make_trial("r2", cfg, {"accuracy": 0.80})
        result = _make_result(
            trials=[rep1, rep2],
            best_config=cfg,
            best_score=0.70,
            objectives=["accuracy"],
            metadata={"session_summary": {"samples_per_config": {"cfg": 2}}},
        )

        assert result.best_metrics["accuracy"] == pytest.approx(0.70)

    def test_three_replicates_mean(self):
        cfg = {"model": "gpt-4"}
        reps = [
            _make_trial("r1", cfg, {"accuracy": 0.60, "cost": 0.10}),
            _make_trial("r2", cfg, {"accuracy": 0.90, "cost": 0.20}),
            _make_trial("r3", cfg, {"accuracy": 0.90, "cost": 0.30}),
        ]
        result = _make_result(
            trials=reps,
            best_config=cfg,
            best_score=0.80,  # mean(0.60, 0.90, 0.90)
            objectives=["accuracy"],
            metadata=_aggregated_metadata(samples=3),
        )

        bm = result.best_metrics
        assert bm["accuracy"] == pytest.approx(0.80)
        assert bm["cost"] == pytest.approx(0.20)


class TestLocalSingleTrialUnchanged:
    def test_local_single_trial_returns_its_own_metrics(self):
        """No aggregation indicator -> single trial's raw metrics, unchanged."""
        cfg = {"model": "gpt-3.5"}
        trial = _make_trial("t1", cfg, {"accuracy": 0.77, "cost": 0.05})
        result = _make_result(
            trials=[trial],
            best_config=cfg,
            best_score=0.77,
            objectives=["accuracy"],
            metadata={},  # no session_summary -> not aggregated
        )

        bm = result.best_metrics
        assert bm["accuracy"] == pytest.approx(0.77)
        assert bm["cost"] == pytest.approx(0.05)

    def test_non_aggregated_multitrial_keeps_winning_trial_metrics(self):
        """Non-aggregated run: best_metrics is the winning trial, not a cross-config mean."""
        cheap = _make_trial("cheap", {"model": "a"}, {"accuracy": 0.70, "cost": 0.01})
        premium = _make_trial(
            "premium", {"model": "b"}, {"accuracy": 0.95, "cost": 0.50}
        )
        result = _make_result(
            trials=[cheap, premium],
            best_config=premium.config,
            best_score=0.95,
            objectives=["accuracy"],
            metadata={},  # not aggregated
        )

        bm = result.best_metrics
        assert bm["accuracy"] == pytest.approx(0.95)
        assert bm["cost"] == pytest.approx(0.50)

    def test_aggregated_single_replicate_matches_that_replicate(self):
        """Aggregated indicator but only one replicate -> equals that replicate."""
        cfg = {"model": "gpt-4"}
        rep = _make_trial("r1", cfg, {"accuracy": 0.88, "cost": 0.07})
        result = _make_result(
            trials=[rep],
            best_config=cfg,
            best_score=0.88,
            objectives=["accuracy"],
            metadata=_aggregated_metadata(samples=1),
        )

        bm = result.best_metrics
        assert bm["accuracy"] == pytest.approx(0.88)
        assert bm["cost"] == pytest.approx(0.07)


class TestEndToEndThroughRealSelection:
    """Drive the real ``select_best_configuration`` aggregated path and wire the
    SelectionResult into an OptimizationResult exactly as the orchestrator does,
    so this guards the session_summary key contract (selection_mode /
    samples_per_config) that ``best_metrics`` relies on."""

    def test_best_metrics_matches_best_score_from_real_aggregated_selection(self):
        win_cfg = {"model": "gpt-4"}
        lose_cfg = {"model": "gpt-3.5"}
        trials = [
            _make_trial("w1", win_cfg, {"accuracy": 0.80, "cost": 0.02}),
            _make_trial("w2", win_cfg, {"accuracy": 0.90, "cost": 0.04}),
            _make_trial("l1", lose_cfg, {"accuracy": 0.50, "cost": 0.01}),
            _make_trial("l2", lose_cfg, {"accuracy": 0.55, "cost": 0.01}),
        ]

        selection = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=True,
            # legacy mode: eligibility needs only a finite primary metric, so we
            # need not fabricate a full comparability payload for this wiring test.
            comparability_mode="legacy",
        )

        # gpt-4 wins on mean accuracy 0.85 > 0.525
        assert selection.best_config == win_cfg
        assert selection.best_score == pytest.approx(0.85)
        assert selection.session_summary is not None
        assert selection.session_summary.get("selection_mode") == "aggregated_mean"

        # Wire into an OptimizationResult the way the orchestrator does.
        metadata: dict[str, Any] = {"session_summary": selection.session_summary}
        if selection.best_trial_id:
            metadata["best_trial_id"] = selection.best_trial_id
        result = _make_result(
            trials=trials,
            best_config=selection.best_config,
            best_score=selection.best_score,
            objectives=["accuracy"],
            metadata=metadata,
        )

        bm = result.best_metrics
        assert bm["accuracy"] == pytest.approx(
            0.85
        ), "best_metrics primary must be the winning config's replicate MEAN"
        assert bm["accuracy"] == pytest.approx(result.best_score)
        assert bm["cost"] == pytest.approx(0.03), "secondary cost must be the mean too"
