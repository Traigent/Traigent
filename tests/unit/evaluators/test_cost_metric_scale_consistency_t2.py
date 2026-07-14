"""Regression tests for finding T2: the ``cost`` metric scale/semantics.

The ``"cost"`` key drives the minimize-cost objective
(:meth:`Orchestrator._extract_objective_values`), the results table, and the
portal. It MUST mean the same thing on every lane:

* local completed lane (:meth:`MetricsTracker.format_for_backend`),
* hybrid lane (:meth:`HybridAPIEvaluator._compute_aggregated_metrics`),
* pruned lane (:func:`build_pruned_result`).

The chosen canonical semantics is **per-trial TOTAL** (sum of per-example
spend), aligning the local lane and the pruned lane to the hybrid lane and to
the authoritative ``total_cost``. Historically the local completed lane emitted
the per-example MEAN, so within one run a completed trial looked ~N× cheaper
than a pruned trial, and the same config looked ~N× cheaper locally than on the
hybrid lane. These tests fail on that old behaviour and pass on the fix.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pytest

from traigent.core.trial_result_factory import (
    build_pruned_result,
    build_success_result,
)
from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
)
from traigent.utils.exceptions import TrialPrunedError

PER_EXAMPLE_COST = 0.01


def _make_tracker(num_examples: int, per_example_cost: float) -> MetricsTracker:
    """Build a tracker with ``num_examples`` successful examples of equal cost."""
    tracker = MetricsTracker()
    tracker.start_tracking()
    for i in range(num_examples):
        tracker.add_example_metrics(
            ExampleMetrics(
                tokens=TokenMetrics(input_tokens=100, output_tokens=50),
                response=ResponseMetrics(response_time_ms=1000 + i),
                # __post_init__ sets total_cost = input_cost + output_cost.
                cost=CostMetrics(input_cost=per_example_cost, output_cost=0.0),
                success=True,
            )
        )
    tracker.end_tracking()
    return tracker


class TestLocalCompletedCostIsPerTrialTotal:
    """Local completed lane emits the per-trial TOTAL, not the per-example mean."""

    def test_cost_is_sum_not_mean(self) -> None:
        n = 4
        formatted = _make_tracker(n, PER_EXAMPLE_COST).format_for_backend()

        # FAIL-ON-OLD: old code returned the per-example MEAN (== PER_EXAMPLE_COST).
        assert formatted["cost"] == pytest.approx(n * PER_EXAMPLE_COST)
        assert formatted["cost"] != pytest.approx(PER_EXAMPLE_COST)

    def test_per_example_mean_preserved_under_distinct_key(self) -> None:
        n = 4
        formatted = _make_tracker(n, PER_EXAMPLE_COST).format_for_backend()

        assert "cost_per_example_mean" in formatted
        assert formatted["cost_per_example_mean"] == pytest.approx(PER_EXAMPLE_COST)
        # The mean must never be overloaded onto ``cost``.
        assert formatted["cost"] != formatted["cost_per_example_mean"]

    def test_empty_tracker_preserves_zero_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            formatted = MetricsTracker().format_for_backend()
        assert formatted["cost"] == 0.0

    def test_empty_tracker_preserves_strict_null(self) -> None:
        with mock.patch.dict(
            os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}, clear=True
        ):
            formatted = MetricsTracker().format_for_backend()
        assert formatted["cost"] is None


class TestCompletedAndPrunedSameScale:
    """Completed and pruned trials in one run rank on the same (total) scale."""

    def test_completed_and_pruned_both_totals(self) -> None:
        n_completed = 5
        k_pruned = 2

        completed = _make_tracker(n_completed, PER_EXAMPLE_COST).format_for_backend()

        pruned = build_pruned_result(
            trial_id="t-pruned",
            evaluation_config={"model": "x"},
            duration=0.1,
            prune_error=TrialPrunedError("pruned at step", step=k_pruned),
            # Cumulative partial spend-so-far == k * per-example cost (a TOTAL).
            progress_state={
                "evaluated": k_pruned,
                "total_examples": n_completed,
                "correct_sum": float(k_pruned),
                "total_cost": k_pruned * PER_EXAMPLE_COST,
            },
            optuna_trial_id=None,
        )

        completed_cost = completed["cost"]
        pruned_cost = pruned.metrics["cost"]

        # Both are TOTALS: dividing by the example count recovers the same
        # per-example cost. On the OLD code the completed cost was already the
        # per-example mean, so completed_cost / n_completed would be
        # PER_EXAMPLE_COST / n_completed -- an ~N× mismatch vs the pruned lane.
        assert completed_cost == pytest.approx(n_completed * PER_EXAMPLE_COST)
        assert pruned_cost == pytest.approx(k_pruned * PER_EXAMPLE_COST)
        assert completed_cost / n_completed == pytest.approx(pruned_cost / k_pruned)
        assert completed_cost / n_completed == pytest.approx(PER_EXAMPLE_COST)


class TestLocalAndHybridAgree:
    """The same config produces the same ``cost`` semantics on both lanes."""

    def test_local_and_hybrid_cost_match(self) -> None:
        n = 3
        local = _make_tracker(n, PER_EXAMPLE_COST).format_for_backend()

        evaluator = HybridAPIEvaluator(transport=MagicMock(), keep_alive=False)
        results = [
            HybridExampleResult(
                example_id=f"ex_{i}", cost_usd=PER_EXAMPLE_COST, metrics={}
            )
            for i in range(n)
        ]
        # The hybrid caller sums per-example spend into ``total_cost``.
        hybrid = evaluator._compute_aggregated_metrics(
            results, total_cost=n * PER_EXAMPLE_COST
        )

        # FAIL-ON-OLD: local would be PER_EXAMPLE_COST (mean) != hybrid total.
        assert local["cost"] == pytest.approx(hybrid["cost"])
        assert local["cost"] == pytest.approx(n * PER_EXAMPLE_COST)


class TestCostReconcilesWithTotalCost:
    """Per-config ``cost`` reconciles with the authoritative ``total_cost``."""

    def test_cost_equals_total_cost_on_completed_trial(self) -> None:
        n = 4
        authoritative_total = n * PER_EXAMPLE_COST
        formatted = _make_tracker(n, PER_EXAMPLE_COST).format_for_backend()

        eval_result = SimpleNamespace(
            metrics=formatted,
            successful_examples=n,
            total_examples=n,
            example_results=None,
            summary_stats=None,
            comparability=None,
        )
        trial = build_success_result(
            trial_id="t-ok",
            evaluation_config={"model": "x"},
            eval_result=eval_result,
            duration=0.1,
            examples_attempted=n,
            # Authoritative per-trial total (same per-example spend, summed).
            total_cost=authoritative_total,
            optuna_trial_id=None,
        )

        # ``total_cost`` is what OptimizationResult.total_cost sums per trial; the
        # minimized ``cost`` now sits on the same scale and reconciles with it.
        assert trial.metrics["total_cost"] == pytest.approx(authoritative_total)
        assert trial.metrics["cost"] == pytest.approx(authoritative_total)
        assert trial.metrics["cost"] == pytest.approx(trial.metrics["total_cost"])
        assert trial.metrics["cost_per_example_mean"] == pytest.approx(PER_EXAMPLE_COST)
