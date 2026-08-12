"""Audit tests for BLOCKER 2 of the third gpt-5.6-sol DO_NOT_MERGE round on
PR #2160: ``hybrid_api.py`` was listed as "SEPARATE EVALUATOR LANE, NOT
AUDITED" in the prior round. Sol required a definitive verdict -- fixed with
discriminating tests, or proven not to share Blocker 1's defect class -- not
another "flagged open" punt.

BLOCKER 1's defect class (see
``test_measured_flag_excludes_unmeasured_zeros_2160.py``): an example with NO
real measurement (its cost/tokens/response-time are the ABSENCE of a
measurement, not a measured zero) gets blanket-included in a MEAN, silently
dragging the reported average down -- zero is not neutral in a mean.

VERDICT for ``hybrid_api.py`` (established by the tests below, each cited to
the exact aggregation code it exercises):

* ``cost``/``total_cost`` (``_compute_aggregated_metrics_with_comparability``,
  hybrid_api.py ~1421-1429): a per-trial TOTAL, i.e. a SUM of
  ``HybridExampleResult.cost_usd`` accumulated by the caller
  (``evaluate()``'s ``total_cost += batch_cost`` loop, ~756-758). A SUM is
  numerically IMMUNE to Blocker 1's defect by construction -- a
  never-measured row's ``cost_usd=0.0`` contributes nothing whether or not it
  is "in" the sum. There is no per-example-mean COST field on this lane
  (unlike the local lane's ``cost_per_example_mean``): the generic
  ``metric_sums``/``metric_counts`` loop (~1437-1451) never populates a
  "cost"/"total_cost" entry because those values live on the dedicated
  ``cost_usd`` dataclass field, never inside ``HybridExampleResult.metrics``
  (the ``if result.success: for metric_name in ("cost", "total_cost", ...)"``
  block that follows is dead code for exactly this reason -- it increments
  counts for keys that were never added to ``metric_sums``, so it has no
  observable effect; see ``test_dead_success_only_count_increment_has_no_
  observable_effect`` below).
* ``latency`` (hybrid_api.py ~1459-1462): a MEAN, but already computed only
  over ``r.latency_ms > 0`` -- i.e. it ALREADY excludes exactly the
  never-measured (``latency_ms=0.0``, the value ``HybridExampleResult``'s
  default and every "missing output" constructor use) rows from both the
  numerator and denominator. This is the SAFE direction: it does not
  blanket-average unmeasured zeros in. (Pre-existing, deliberately tested
  behaviour: see ``TestComputeAggregatedMetrics.test_latency_averaged`` /
  ``test_no_positive_latency`` in ``test_hybrid_api_evaluator.py``.)
* tokens: NOT TRACKED AT ALL on this lane -- ``HybridExampleResult`` has no
  token field, and no aggregation path here ever produces
  ``input_tokens``/``output_tokens``/``total_tokens``. There is no surface
  for Blocker 1's defect to exist on.
* accuracy (and every other per-example quality metric): a MEAN computed via
  the SAME generic ``metric_sums``/``metric_counts`` loop, but the loop
  iterates ``result.metrics.items()`` -- a row with no measurement
  (``metrics={}``, the value every "missing output"/whole-batch-failure
  constructor uses: ``_batch_error_results``, the ``output_item is None``
  branches in ``_process_combined_response``/``_evaluate_outputs``/
  ``_process_execute_only_response``) contributes NO ENTRY to
  ``metric_sums``/``metric_counts`` at all. It is EXCLUDED from the mean's
  numerator AND denominator -- NOT included as a 0.0. This is the OPPOSITE
  of Blocker 1's defect (which INCLUDED zeros), so it does not share that bug
  class. It IS a pre-existing, intentionally tested design choice (see
  ``TestComputeAggregatedMetrics.test_partial_metrics`` in
  ``test_hybrid_api_evaluator.py``: "Metrics only present in some results are
  averaged over their count") -- NOT a new regression from this PR, and
  changing it to instead count a missing row as 0.0 (local-lane-#1963-style)
  would (a) reverse that currently-passing, deliberately-named test, and (b)
  require deciding accuracy deserves different denominator semantics than
  every OTHER custom quality metric this lane tracks identically -- a
  product/policy call for the owner, not a unilateral fix under a "does this
  share Blocker 1's bug class" audit. Reported as an explicit finding/
  recommendation below, not silently left unexamined and not silently
  "fixed" by reversing tested behaviour.
"""

from __future__ import annotations

import pytest

from traigent.evaluators.hybrid_api import HybridAPIEvaluator, HybridExampleResult


@pytest.fixture
def ev() -> HybridAPIEvaluator:
    # No transport needed: every method under test is a pure function over a
    # ``list[HybridExampleResult]`` the caller already assembled.
    return HybridAPIEvaluator(api_endpoint="http://unused.invalid")


def _mixed_measured_and_missing_output_results() -> list[HybridExampleResult]:
    """1 real (measured) row + 1 "missing output" row -- the hybrid-lane
    analogue of BLOCKER 1's local-lane fixture: a call that never produced
    any output at all (``_batch_error_results``/``output_item is None``),
    so its ``cost_usd``/``latency_ms`` are the SDK's ``0.0`` defaults and its
    ``metrics`` dict is empty -- the ABSENCE of a measurement, not a measured
    zero.
    """
    return [
        HybridExampleResult(
            example_id="measured",
            metrics={"accuracy": 0.8},
            cost_usd=0.02,
            latency_ms=200.0,
        ),
        HybridExampleResult(
            example_id="missing_output",
            error="Execute response did not include output for example_id "
            "'missing_output'",
            metrics={},
            cost_usd=0.0,
            latency_ms=0.0,
        ),
    ]


class TestCostSumImmuneToUnmeasuredRows:
    """``cost``/``total_cost`` are per-trial SUMS -- immune to Blocker 1's
    "averaged-in-as-zero" defect by construction, regardless of how many
    never-measured rows are mixed in.
    """

    def test_cost_sum_reflects_true_total_regardless_of_missing_output_rows(
        self, ev: HybridAPIEvaluator
    ) -> None:
        results = _mixed_measured_and_missing_output_results()
        # total_cost is accumulated by the caller (evaluate()'s own
        # ``total_cost += batch_cost`` loop) from the SAME per-example
        # cost_usd values -- passing the true sum here mirrors that call
        # site exactly.
        true_total = sum(r.cost_usd for r in results)

        agg = ev._compute_aggregated_metrics(results, total_cost=true_total)

        assert agg["cost"] == pytest.approx(0.02)
        assert agg["total_cost"] == pytest.approx(0.02)

    def test_dead_success_only_count_increment_has_no_observable_effect(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """The ``if result.success: for metric_name in ("cost", "total_cost",
        "latency", "response_time_ms"): metric_counts[...] += 1`` block
        (hybrid_api.py ~1444-1451) increments counts for keys that are NEVER
        populated in ``metric_sums`` under normal use (cost/latency live on
        dedicated dataclass fields, not inside ``.metrics``) -- so it has no
        effect on ``aggregated``. Confirmed here so the audit's claim is
        evidence-backed, not just read from the source.
        """
        results = [
            HybridExampleResult(example_id="1", cost_usd=0.01, latency_ms=10.0),
            HybridExampleResult(example_id="2", cost_usd=0.02, latency_ms=20.0),
        ]
        agg = ev._compute_aggregated_metrics(results, total_cost=0.03)

        # "cost"/"total_cost" come ONLY from the `total_cost` sum argument,
        # never from a mean over metric_sums -- if the dead code had any
        # effect, `cost` would be corrupted into some fraction of 0.03.
        assert agg["cost"] == pytest.approx(0.03)
        assert agg["total_cost"] == pytest.approx(0.03)


class TestLatencyAlreadyExcludesUnmeasuredZeros:
    """``latency`` is a MEAN, but already gated on ``latency_ms > 0`` --
    proving it does NOT share Blocker 1's "blanket-include unmeasured
    zeros" defect (it already excludes them, the safe direction).
    """

    def test_latency_mean_reflects_only_the_measured_row(
        self, ev: HybridAPIEvaluator
    ) -> None:
        results = _mixed_measured_and_missing_output_results()

        agg = ev._compute_aggregated_metrics(results, total_cost=0.02)

        # Buggy (Blocker-1-style, if it existed here): mean over BOTH rows
        # = (200.0 + 0.0) / 2 = 100.0. Actual (already correct): only the
        # measured row counts.
        assert agg["latency"] == pytest.approx(200.0)
        assert agg["latency"] != pytest.approx(100.0)
        assert agg["response_time_ms"] == pytest.approx(200.0)


class TestAccuracyExcludesRatherThanZeroAveragesMissingRows:
    """Documents the CURRENT, pre-existing, intentionally tested design for
    accuracy/generic quality metrics: a missing-output row is EXCLUDED from
    the mean (not zero-averaged in). This does NOT share Blocker 1's bug
    class (which INCLUDED zeros) -- it is a DIFFERENT, deliberate design
    choice this audit reports as a finding rather than silently overriding.
    """

    def test_accuracy_mean_excludes_missing_output_row_denominator(
        self, ev: HybridAPIEvaluator
    ) -> None:
        results = _mixed_measured_and_missing_output_results()

        agg = ev._compute_aggregated_metrics(results, total_cost=0.02)

        # If this lane shared Blocker 1's defect, the missing row's absent
        # "accuracy" key would need to be zero-averaged in for the bug to
        # exist. Instead it is excluded entirely: mean is exactly the
        # measured row's own value (0.8), with denominator 1, not 2.
        assert agg["accuracy"] == pytest.approx(0.8)

    def test_comparability_flags_the_missing_row_as_partial_coverage(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """The comparability metadata (a SEPARATE, existing mechanism) is
        how this lane surfaces "not every example was scored" to callers --
        distinct from (and not a substitute for) a local-lane-#1963-style
        denominator fix, but confirms the gap is not silently invisible.
        """
        results = _mixed_measured_and_missing_output_results()

        _agg, comparability = ev._compute_aggregated_metrics_with_comparability(
            results, total_cost=0.02
        )

        assert comparability["total_examples"] == 2
        assert comparability["examples_with_primary_metric"] == 1
        assert comparability["coverage_ratio"] == pytest.approx(0.5)
        assert comparability["ranking_eligible"] is False
        assert "MCI-002" in comparability["warning_codes"]


class TestBuildSummaryStatsExcludesUnmeasuredZeroRows:
    """``_build_summary_stats`` had ZERO existing test coverage before this
    audit. Same verdict as the primary aggregation path: cost/latency
    magnitude-filter out never-measured rows (safe), accuracy excludes
    rather than zero-averages them (different design, not Blocker 1's bug).
    """

    def test_cost_and_latency_stats_exclude_the_missing_output_row(
        self, ev: HybridAPIEvaluator
    ) -> None:
        results = _mixed_measured_and_missing_output_results()

        stats = ev._build_summary_stats(results, duration=1.0)
        assert stats is not None
        metrics = stats["metrics"]

        # Buggy (Blocker-1-style): count=2, mean=(0.02+0.0)/2=0.01. Actual:
        # only the measured row is in the list.
        assert metrics["total_cost"]["count"] == 1
        assert metrics["total_cost"]["mean"] == pytest.approx(0.02)
        assert metrics["total_cost"]["mean"] != pytest.approx(0.01)
        assert metrics["latency"]["count"] == 1
        assert metrics["latency"]["mean"] == pytest.approx(200.0)
        assert metrics["latency"]["mean"] != pytest.approx(100.0)

    def test_accuracy_stats_exclude_the_missing_output_row(
        self, ev: HybridAPIEvaluator
    ) -> None:
        results = _mixed_measured_and_missing_output_results()

        stats = ev._build_summary_stats(results, duration=1.0)
        assert stats is not None
        metrics = stats["metrics"]

        assert metrics["accuracy"]["count"] == 1
        assert metrics["accuracy"]["mean"] == pytest.approx(0.8)

        # success_rate DOES cover both rows (it is built from
        # `result.success`, unconditionally, for every result) -- the
        # missing-output row is not invisible everywhere, just excluded
        # from the quality-metric means.
        assert metrics["success_rate"]["count"] == 2
        assert metrics["success_rate"]["mean"] == pytest.approx(0.5)

    def test_no_measured_examples_omits_cost_latency_accuracy_not_zero_fills(
        self, ev: HybridAPIEvaluator
    ) -> None:
        """When EVERY row is a missing-output/never-measured row, the
        cost/latency/accuracy keys are ABSENT from ``summary_metrics``
        entirely (their value lists were empty, and the ``if values``
        guard, hybrid_api.py ~646-649, skips empty ones) -- NOT
        fabricated as a zero-filled describe structure. ``success_rate`` IS
        still present (built from ``result.success`` unconditionally for
        every result, never gated on whether anything was measured), so the
        payload as a whole is not ``None`` -- but the measurement-derived
        fields correctly fail closed/absent rather than reporting a
        confident zero.
        """
        results = [
            HybridExampleResult(
                example_id="missing_1", error="boom1", metrics={}, cost_usd=0.0
            ),
            HybridExampleResult(
                example_id="missing_2", error="boom2", metrics={}, cost_usd=0.0
            ),
        ]

        stats = ev._build_summary_stats(results, duration=1.0)

        assert stats is not None
        metrics = stats["metrics"]
        assert "total_cost" not in metrics
        assert "latency" not in metrics
        assert "accuracy" not in metrics
        assert metrics["success_rate"]["count"] == 2
        assert metrics["success_rate"]["mean"] == pytest.approx(0.0)
