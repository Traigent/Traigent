"""Closing tests for issue #2192: ``hybrid_api.py`` aggregates means
independently and was never checked against the measured-vs-unmeasured
distinction from #1964/#2160.

``test_hybrid_api_measured_zero_averaging_audit_2160.py`` already proved this
lane does not share PR #2160's Blocker 1 defect class for the "missing
output" case, where an errored ``HybridExampleResult`` has an EMPTY
``metrics`` dict (``{}``) -- that row contributes no entry to the mean at
all, which is the audit's whole point.

That earlier audit did not cover a narrower, still-live case: a result can
carry BOTH ``error`` (non-None) AND a NON-EMPTY ``metrics`` dict at the same
time. Concretely:

* ``_process_combined_response`` (hybrid_api.py) normalizes
  ``output_item.get("metrics")`` into ``per_example_metrics`` unconditionally
  whenever it is a dict, independent of whether ``output_item.get("error")``
  is also set on the same output item.
* ``_evaluate_outputs`` normalizes ``result.get("metrics", {})`` from the
  external service's ``/evaluate`` response for a given example_id even when
  that same result also carries ``eval_error``.

So an external agent service that scores an output (perhaps a real 0.0, or a
placeholder value it fills in for schema completeness) and *also* reports an
error for that same example produces exactly the row this test module
targets: ``error is not None`` and ``metrics`` non-empty. Before this fix,
both aggregation paths (``_compute_aggregated_metrics_with_comparability``
and ``_build_summary_stats``) included that row's metrics in the mean
unconditionally -- reproducing PR #2160's Blocker 1 defect class (a
placeholder/untrustworthy value silently depressing a reported average) on
exactly the sub-case the 2160 audit did not exercise.

Fix: both aggregation loops now skip a result's ``.metrics`` entirely when
``result.success`` is ``False`` (i.e. ``error is not None``), regardless of
whether that ``metrics`` dict happens to be empty or populated. This does not
change the already-tested "missing output" behaviour (which relied on the
metrics dict being empty, not on a per-result success check) -- it closes the
gap the 2160 audit itself asked a future pass to check.
"""

from __future__ import annotations

import pytest

from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult


@pytest.fixture
def ev() -> HybridAPIEvaluator:
    return HybridAPIEvaluator(api_endpoint="http://unused.invalid")


def _one_measured_and_one_errored_with_populated_metrics() -> list[HybridExampleResult]:
    """1 real (measured) row + 1 errored row whose ``metrics`` dict is
    NOT empty -- the external agent service scored the example (or filled a
    placeholder) before/alongside reporting an error for it. This is the
    live gap: it is different from a "missing output" row (``metrics={}``),
    which was already proven safe by the #2160 audit.
    """
    return [
        HybridExampleResult(
            example_id="measured",
            metrics={"accuracy": 0.8},
            cost_usd=0.02,
            latency_ms=200.0,
        ),
        HybridExampleResult(
            example_id="errored_with_metrics",
            error="downstream_validation_failed",
            metrics={"accuracy": 0.0},
            cost_usd=0.0,
            latency_ms=0.0,
        ),
    ]


class TestPrimaryAggregationExcludesErroredResultMetrics:
    """``_compute_aggregated_metrics``/``_compute_aggregated_metrics_with_
    comparability`` must not let an errored result's ``metrics`` payload
    into the per-example-mean, even when that payload is non-empty.
    """

    def test_accuracy_mean_ignores_errored_row_with_populated_metrics(
        self, ev: HybridAPIEvaluator
    ) -> None:
        results = _one_measured_and_one_errored_with_populated_metrics()

        agg = ev._compute_aggregated_metrics(results, total_cost=0.02)

        # Buggy (pre-fix): mean over BOTH rows = (0.8 + 0.0) / 2 = 0.4.
        # Fixed: only the successful row's accuracy counts.
        assert agg["accuracy"] == pytest.approx(0.8)
        assert agg["accuracy"] != pytest.approx(0.4)

    def test_success_rate_still_reflects_both_rows(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """success_rate must still see the error -- only the QUALITY metric
        aggregation excludes it, not the failure signal itself."""
        results = _one_measured_and_one_errored_with_populated_metrics()

        agg = ev._compute_aggregated_metrics(results, total_cost=0.02)

        assert agg["success_rate"] == pytest.approx(0.5)


class TestBuildSummaryStatsExcludesErroredResultMetrics:
    """Same closing verdict for the ``summary_stats`` builder."""

    def test_accuracy_stats_ignore_errored_row_with_populated_metrics(
        self, ev: HybridAPIEvaluator
    ) -> None:
        results = _one_measured_and_one_errored_with_populated_metrics()

        stats = ev._build_summary_stats(results, duration=1.0)
        assert stats is not None
        metrics = stats["metrics"]

        assert metrics["accuracy"]["count"] == 1
        assert metrics["accuracy"]["mean"] == pytest.approx(0.8)
        assert metrics["accuracy"]["mean"] != pytest.approx(0.4)

        # success_rate still covers both rows (built from `result.success`
        # unconditionally, independent of the quality-metric exclusion).
        assert metrics["success_rate"]["count"] == 2
        assert metrics["success_rate"]["mean"] == pytest.approx(0.5)
