"""Enhanced metrics tracking for comprehensive evaluation metrics."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Observability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

import math
import os
import re
import statistics
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from traigent._version import get_version

# The backend ``MeasuresDict`` ceiling (≤ 50 total keys) is the single source of
# truth for the per-trial metric cardinality cap. It lives in
# ``traigent.knobs.telemetry`` (which has no evaluators dependency, so importing
# it here is cycle-free) and is reused by ``merge_composite_measures`` for the
# pre-merge path; this aggregation path mirrors the SAME semantics — user keys
# are truncated, evaluator keys are never dropped.
from traigent.knobs.telemetry import TOTAL_MEASURES_CEILING
from traigent.utils.logging import get_logger

# Initialize logger first
logger = get_logger(__name__)

#: Wire contract for user-supplied metric KEY names. This MUST stay byte-for-byte
#: identical to ``traigent.cloud.dtos.MeasuresDict.KEY_PATTERN`` (dtos.py:250):
#: a user key that the evaluator unpacks but the ``MeasuresDict`` wire contract
#: later rejects is the exact Unicode-identifier bypass we are closing — the key
#: would ride into ``result.metrics`` and then either be dropped at submission or
#: smuggled through the backward-compat catch unvalidated. We deliberately do NOT
#: import ``MeasuresDict`` here (the cloud layer must not be a dependency of the
#: evaluators layer); the duplication is pinned by
#: ``test_user_metric_pattern_matches_measuresdict_pattern`` so any drift fails a
#: test rather than silently reopening the bypass. ``re.ASCII`` keeps ``\w`` to
#: ``[A-Za-z0-9_]`` so Unicode identifiers (e.g. ``"π_metric"``) are rejected.
USER_METRIC_KEY_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z_]\w*$", re.ASCII)

# Expose TOKENCOST availability and backward compatibility functions for tests
try:
    from traigent.utils.cost_calculator import (
        TOKENCOST_AVAILABLE as _TOKENCOST_AVAILABLE,
    )
    from traigent.utils.cost_calculator import (
        calculate_completion_cost,
        calculate_prompt_cost,
    )
except Exception:
    _TOKENCOST_AVAILABLE = False
    calculate_prompt_cost = None  # type: ignore[assignment]
    calculate_completion_cost = None  # type: ignore[assignment]

TOKENCOST_AVAILABLE = _TOKENCOST_AVAILABLE

REPORTED_COST_PLAUSIBILITY_FLOOR_RATIO = 0.5

#: Evaluator/tracker-computed metric keys that user-supplied per-example metrics
#: (e.g. the ``(output, metrics_dict)`` tuple channel) MUST NEVER overwrite,
#: at ANY merge or aggregation site, regardless of merge ordering or
#: ``setdefault`` timing. Derived from what actually lands in a trial's metrics:
#:
#: * the metric registry built-ins (``BaseEvaluator._metric_registry``):
#:   ``accuracy``, ``success_rate``, ``error_rate``, ``avg_output_length``,
#:   ``cost``, ``latency`` (plus the per-example ``success`` flag);
#: * ``MetricsTracker.format_for_backend`` outputs: ``score``, ``accuracy``,
#:   ``duration``, ``input_tokens``, ``output_tokens``, ``total_tokens``,
#:   ``response_time_ms``, ``cost`` (per-trial TOTAL),
#:   ``cost_per_example_mean``, ``total_examples``, ``successful_examples``,
#:   ``tokens_per_second``;
#: * the LLM aggregation (``_aggregate_llm_metrics``): ``prompt_tokens``,
#:   ``completion_tokens``, ``total_tokens``, ``input_cost``, ``output_cost``,
#:   ``total_cost``, ``avg_response_time``, ``avg_response_time_ms``;
#: * standard/LLM per-example keys and lifecycle counters: ``input_cost``,
#:   ``output_cost``, ``total_cost``, ``examples_attempted``,
#:   ``examples_consumed``, ``execution_time_ms``.
RESERVED_METRIC_KEYS: frozenset[str] = frozenset(
    {
        # Metric-registry built-ins + per-example success flag.
        "accuracy",
        "success",
        "success_rate",
        "error_rate",
        "avg_output_length",
        "cost",
        # Per-example MEAN cost, surfaced alongside the per-trial TOTAL ``cost``
        # (finding T2). Reserved so a user tuple key cannot overwrite it and it is
        # never dropped under the measures ceiling.
        "cost_per_example_mean",
        "latency",
        "score",
        # Diagnostic: the built-in exact-match scorer recorded alongside a custom
        # scoring_function that owns the objective (issue #1845). Reserved so it
        # is never dropped under the measures ceiling nor overwritten by a
        # user-supplied tuple key of the same name.
        "exact_match_default",
        # Diagnostic: fraction of a config's outputs that are empty or
        # whitespace-only (issue #1851). Computed by the evaluator on EVERY trial
        # (metadata-free complement to the finish_reason guard, #1809). Reserved
        # so a user tuple key named ``empty_output_rate`` cannot overwrite the
        # evaluator's value and it is never sacrificed to the measures ceiling —
        # portals/harnesses gate on it, so it must always survive to the trial.
        "empty_output_rate",
        # format_for_backend / summary outputs.
        "duration",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "response_time_ms",
        "total_examples",
        "successful_examples",
        "tokens_per_second",
        # LLM aggregation + per-example cost/token detail.
        "prompt_tokens",
        "completion_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
        "avg_response_time",
        "avg_response_time_ms",
        # Lifecycle counters / timing.
        "examples_attempted",
        "examples_consumed",
        "execution_time_ms",
        # Surrogate (pre-screen) evaluator: a cheap SECOND scorer injected by the
        # trial lifecycle into per-example + aggregate metrics. Reserving it means
        # (a) a user tuple key named ``surrogate_score`` cannot overwrite it, and
        # (b) it is never sacrificed to the TOTAL_MEASURES_CEILING user-key cap —
        # so a run with 50 user metrics still submits the surrogate score.
        "surrogate_score",
        # TRANSPORT-reserved: the submission lane passes the measures array and
        # summary_stats object THROUGH the metrics dict and extracts them BY NAME
        # in ``trial_operations._extract_measures_from_metrics``; the submission
        # validator ``validators._validate_metrics_no_misplaced_fields`` HARD-
        # rejects either name inside a metrics dict. A user tuple key with one of
        # these names is wire-valid (identifier + numeric) and would otherwise
        # ride into metrics and break submission — and the caps must never drop
        # the system's own transport entries. Reserving them skips user keys with
        # these names (WARNING) at merge.
        "measures",
        "summary_stats",
    }
)


def is_reserved_metric_key(key: str) -> bool:
    """Return ``True`` if ``key`` is an evaluator/tracker-computed metric.

    User-supplied per-example metrics must never overwrite a reserved key; see
    :data:`RESERVED_METRIC_KEYS`.
    """
    return key in RESERVED_METRIC_KEYS


#: Default fraction of empty/whitespace-only outputs a config may have before the
#: evaluator surfaces a run-level warning (issue #1851). An empty output at any
#: meaningful rate signals truncation, output-parsing failure, or refusals — the
#: accuracy comparison is then measuring an artifact, not the config knobs. The
#: warning fires strictly ABOVE this threshold (``rate > threshold``).
EMPTY_OUTPUT_RATE_WARNING_THRESHOLD: float = 0.10


def output_is_empty(output: Any) -> bool:
    """Return ``True`` if ``output`` is effectively empty (issue #1851).

    An output is empty when it is ``None`` or its string form is blank /
    whitespace-only. This is the metadata-free signal that catches truncation,
    output-parsing failures, and refusals even when the user's own function
    makes the LLM call and returns only a string (complement to #1809).
    """
    return output is None or not str(output).strip()


def compute_empty_output_rate(outputs: Sequence[Any]) -> float:
    """Fraction of ``outputs`` that are empty or whitespace-only (issue #1851).

    ``empty_output_rate == mean(output is None or not str(output).strip())`` over
    the trial's per-example outputs. Returns ``0.0`` for an empty sequence (no
    outputs means no empties to report, not a divide-by-zero).
    """
    if not outputs:
        return 0.0
    empty = sum(1 for output in outputs if output_is_empty(output))
    return empty / len(outputs)


def aggregate_user_custom_metrics(
    target: dict[str, Any],
    example_metric_dicts: Sequence[dict[str, Any]],
    *,
    context: str,
    extra_reserved: frozenset[str] = frozenset(),
) -> None:
    """Mean-aggregate non-reserved user keys across examples into ``target``.

    Shared by both the ``SimpleScoringEvaluator`` trial-aggregation path and
    ``MetricsTracker._aggregate_user_custom_metrics`` so the tuple-return
    metrics channel surfaces user keys identically on every public evaluator.

    Semantics (fail-closed):

    * a key in :data:`RESERVED_METRIC_KEYS` OR in ``extra_reserved`` is skipped
      (the evaluator's own computed value wins). ``extra_reserved`` carries the
      runtime-only evaluator-computable names (registered custom metrics and
      RAGAS metrics such as ``context_precision``) that the static frozenset
      cannot enumerate, so a user tuple key with one of those names cannot
      overwrite the evaluator's value;
    * a key already present in ``target`` is left untouched (never clobbered);
    * non-numeric / non-finite / ``bool`` values are skipped (the wire channel
      carries finite numbers only);
    * the resulting ``target`` is capped at :data:`TOTAL_MEASURES_CEILING`
      TOTAL keys — only USER keys are truncated, deterministically in sorted
      order, with the drop warning-logged; evaluator keys are never dropped,
      mirroring ``merge_composite_measures``.

    Logging split: reserved-key SKIPS here log at ``DEBUG``. This shared pass is
    fed evaluator-INTERNAL dicts (e.g. the local lane's per-example
    ``custom_metrics``, which carry injected ``input_cost`` / ``total_cost`` /
    token keys), so a reserved-key collision here is dominated by the evaluator's
    own keys, not a user mistake — warning on it would lie. The per-example
    ``BaseEvaluator._merge_user_metrics`` (which sees the GENUINELY user-supplied
    tuple dict) keeps the ``WARNING`` for the same condition.
    """
    values_by_key: dict[str, list[float]] = {}
    for metric in example_metric_dicts:
        if not metric:
            continue
        for key, value in metric.items():
            if key in target:
                continue
            if is_reserved_metric_key(key) or key in extra_reserved:
                logger.debug(
                    "Skipping reserved metric %r (%s): evaluator-computed key, "
                    "cannot be overwritten",
                    key,
                    context,
                )
                continue
            if isinstance(value, bool) or value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            values_by_key.setdefault(key, []).append(numeric)

    means = {
        key: sum(values) / len(values)
        for key, values in values_by_key.items()
        if values
    }
    _merge_capped_user_means(target, means, context=context)


def _merge_capped_user_means(
    target: dict[str, Any],
    user_means: dict[str, float],
    *,
    context: str,
) -> None:
    """Merge ``user_means`` into ``target`` under the total-key ceiling.

    Only user keys are truncated (deterministic sorted order, warning-logged);
    evaluator keys already in ``target`` are never dropped.
    """
    if not user_means:
        return
    budget = TOTAL_MEASURES_CEILING - len(target)
    ordered_keys = sorted(user_means)
    if budget <= 0:
        logger.warning(
            "%s: trial metrics already at the %d-key ceiling; ALL %d "
            "user metric(s) dropped: %s",
            context,
            TOTAL_MEASURES_CEILING,
            len(ordered_keys),
            ordered_keys[:10],
        )
        return
    if len(ordered_keys) > budget:
        kept = ordered_keys[:budget]
        dropped = ordered_keys[budget:]
        logger.warning(
            "%s: user metrics would exceed the %d-key ceiling; %d user key(s) "
            "dropped deterministically: %s",
            context,
            TOTAL_MEASURES_CEILING,
            len(dropped),
            dropped[:10],
        )
        ordered_keys = kept
    for key in ordered_keys:
        target[key] = user_means[key]


def enforce_user_metric_ceiling(
    target: dict[str, Any],
    *,
    context: str,
    extra_reserved: frozenset[str] = frozenset(),
) -> None:
    """Clamp ``target`` to :data:`TOTAL_MEASURES_CEILING` TOTAL keys in place.

    The cap mirrors ``merge_composite_measures``: only NON-reserved (user) keys
    are dropped — deterministically, in sorted order, warning-logged — so the
    reserved evaluator-computed keys are never sacrificed to fit the ceiling.
    ``extra_reserved`` extends the static :data:`RESERVED_METRIC_KEYS` with the
    evaluator's runtime-only computable names (registry + RAGAS), so an
    evaluator-computed key like ``context_precision`` is never mistaken for a
    droppable user key. This is the authoritative cap for a lane whose user keys
    arrive via an intermediate dict (e.g. the local lane's
    ``comprehensive_metrics``), where capping the intermediate dict alone cannot
    bound the final union.
    """
    if len(target) <= TOTAL_MEASURES_CEILING:
        return
    user_keys = sorted(
        k for k in target if not is_reserved_metric_key(k) and k not in extra_reserved
    )
    overflow = len(target) - TOTAL_MEASURES_CEILING
    if overflow <= 0:
        return
    # Drop the LAST `overflow` user keys (deterministic sorted order); reserved
    # keys are never eligible for dropping.
    droppable = user_keys[len(user_keys) - min(overflow, len(user_keys)) :]
    for key in droppable:
        del target[key]
    if droppable:
        logger.warning(
            "%s: trial metrics exceeded the %d-key ceiling; %d user key(s) "
            "dropped deterministically: %s",
            context,
            TOTAL_MEASURES_CEILING,
            len(droppable),
            droppable[:10],
        )
    if len(target) > TOTAL_MEASURES_CEILING:
        # All remaining overflow is reserved keys (never dropped): the evaluator
        # itself produced more than the ceiling. Surface loudly; do not silently
        # exceed the contract by dropping computed keys.
        logger.warning(
            "%s: %d reserved evaluator key(s) leave the trial metrics above the "
            "%d-key ceiling; reserved keys are never dropped",
            context,
            len(target) - TOTAL_MEASURES_CEILING,
            TOTAL_MEASURES_CEILING,
        )


@dataclass
class TokenMetrics:
    """Token usage metrics for a single evaluation."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        # Ensure non-negative values and handle None
        self.input_tokens = max(0, self.input_tokens or 0)
        self.output_tokens = max(0, self.output_tokens or 0)
        self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class ResponseMetrics:
    """Response time metrics for a single evaluation."""

    response_time_ms: float = 0.0
    first_token_ms: float | None = None
    tokens_per_second: float | None = None

    def __post_init__(self) -> None:
        # Ensure non-negative response time
        self.response_time_ms = max(0.0, self.response_time_ms or 0.0)
        if self.first_token_ms is not None:
            self.first_token_ms = max(0.0, self.first_token_ms)
        if self.tokens_per_second is not None:
            self.tokens_per_second = max(0.0, self.tokens_per_second)


@dataclass
class CostMetrics:
    """Cost metrics for a single evaluation."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    # True when non-strict cost accounting could not price the model at all
    # (litellm + custom pricing + Traigent's builtin fallback all missed) and
    # ``total_cost`` was recorded as 0.0 despite non-zero token usage (#1597).
    # This is distinct from a model that is legitimately free/self-hosted —
    # those resolve to a real (known) 0.0 price and never set this flag.
    # Consumers (trial/result aggregation) must treat this as "unknown spend",
    # not "verified free".
    unpriced: bool = False

    def __post_init__(self) -> None:
        # Ensure non-negative costs and handle None
        self.input_cost = max(0.0, self.input_cost or 0.0)
        self.output_cost = max(0.0, self.output_cost or 0.0)
        self.total_cost = self.input_cost + self.output_cost


@dataclass
class ExampleMetrics:
    """Complete metrics for a single example evaluation."""

    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    response: ResponseMetrics = field(default_factory=ResponseMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    success: bool = True
    error: str | None = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
    """Tracks and aggregates metrics across multiple evaluations."""

    def __init__(self) -> None:
        self.example_metrics: list[ExampleMetrics] = []
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start_tracking(self) -> None:
        """Start tracking metrics."""
        self.start_time = time.time()
        self.example_metrics = []

    def add_example_metrics(self, metrics: ExampleMetrics) -> None:
        """Add metrics for a single example."""
        if metrics is None:
            logger.warning(
                "Attempted to add None metrics, creating empty metrics instead"
            )
            metrics = ExampleMetrics()
        elif not isinstance(metrics, ExampleMetrics):
            logger.error(
                f"Invalid metrics type: {type(metrics)}. Expected ExampleMetrics"
            )
            return
        self.example_metrics.append(metrics)

    def end_tracking(self) -> None:
        """End tracking and calculate duration."""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get total duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def calculate_statistics(self, values: Sequence[int | float]) -> dict[str, float]:
        """Calculate mean, median, and std for a list of values."""
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}

        # Filter out None values and ensure all are float
        clean_values = []
        for v in values:
            if v is not None:
                try:
                    clean_values.append(float(v))
                except (TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid value {v}: {e}")

        if not clean_values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}

        try:
            mean_val = statistics.mean(clean_values)
            median_val = statistics.median(clean_values)
            std_val = statistics.stdev(clean_values) if len(clean_values) > 1 else 0.0
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {"mean": 0.0, "median": 0.0, "std": 0.0}

        return {
            "mean": round(mean_val, 6),
            "median": round(median_val, 6),
            "std": round(std_val, 6),
        }

    def aggregate_metrics(self) -> dict[str, Any]:
        """Aggregate all tracked metrics into summary statistics."""
        if not self.example_metrics:
            return self._empty_aggregated_metrics()

        # Filter successful examples for metrics calculation
        successful_metrics = [m for m in self.example_metrics if m.success]

        if not successful_metrics:
            return self._empty_aggregated_metrics()

        # Extract values for each metric type
        input_tokens = [m.tokens.input_tokens for m in successful_metrics]
        output_tokens = [m.tokens.output_tokens for m in successful_metrics]
        total_tokens = [m.tokens.total_tokens for m in successful_metrics]

        response_times = [m.response.response_time_ms for m in successful_metrics]

        input_costs = [m.cost.input_cost for m in successful_metrics]
        output_costs = [m.cost.output_cost for m in successful_metrics]
        total_costs = [m.cost.total_cost for m in successful_metrics]

        # Calculate statistics for each metric
        aggregated = {
            # Token metrics
            "input_tokens": self.calculate_statistics(input_tokens),
            "output_tokens": self.calculate_statistics(output_tokens),
            "total_tokens": self.calculate_statistics(total_tokens),
            # Response time metrics
            "response_time_ms": self.calculate_statistics(response_times),
            # Cost metrics
            "input_cost": self.calculate_statistics(input_costs),
            "output_cost": self.calculate_statistics(output_costs),
            "total_cost": self.calculate_statistics(total_costs),
            # Summary metrics
            "total_examples": len(self.example_metrics),
            "successful_examples": len(successful_metrics),
            "success_rate": (
                len(successful_metrics) / len(self.example_metrics)
                if len(self.example_metrics) > 0
                else 0.0
            ),
            "duration": self.get_duration(),
        }

        # Add tokens per second if available
        tps_values = [
            m.response.tokens_per_second
            for m in successful_metrics
            if m.response.tokens_per_second is not None
        ]
        if tps_values:
            aggregated["tokens_per_second"] = self.calculate_statistics(tps_values)

        return aggregated

    def _empty_aggregated_metrics(self) -> dict[str, Any]:
        """Return empty aggregated metrics structure."""
        # Check for strict metrics nulls mode
        strict_nulls = os.environ.get("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        )
        missing_default = None if strict_nulls else 0.0

        empty_stats = {
            "mean": missing_default,
            "median": missing_default,
            "std": missing_default,
        }
        return {
            "input_tokens": empty_stats.copy(),
            "output_tokens": empty_stats.copy(),
            "total_tokens": empty_stats.copy(),
            "response_time_ms": empty_stats.copy(),
            "input_cost": empty_stats.copy(),
            "output_cost": empty_stats.copy(),
            "total_cost": empty_stats.copy(),
            "total_examples": 0,
            "successful_examples": 0,
            "success_rate": missing_default,
            "duration": missing_default,
        }

    def format_for_backend(
        self, extra_reserved: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Format aggregated metrics for backend submission.

        ``extra_reserved`` carries the owning evaluator's runtime-only
        computable metric names (registered custom metrics + RAGAS names) so
        user-supplied tuple keys cannot overwrite an evaluator-computed value
        during the user-metric aggregation pass below.
        """
        try:
            aggregated = self.aggregate_metrics()
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            return self._empty_backend_format()

        strict_nulls = os.environ.get("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        )

        # Safe getter with defaults
        def safe_get(
            d: dict[str, Any], key: str, subkey: str | None = None, default: float = 0.0
        ) -> float | None:
            """Safely get value from nested dict with default."""

            actual_default = None if strict_nulls else default

            try:
                if subkey:
                    return cast(
                        float | None, d.get(key, {}).get(subkey, actual_default)
                    )
                return cast(float | None, d.get(key, actual_default))
            except (AttributeError, TypeError):
                return actual_default

        # Format metrics in a cleaner format without _mean/_median/_std suffixes
        # Since measures now contain per-example data and summary_stats contain
        # the statistical aggregations, we only need the primary metrics here

        # Calculate accuracy from custom metrics if available
        accuracy_value: float | None = None
        if self.example_metrics:
            accuracy_scores = [
                m.custom_metrics["accuracy"]
                for m in self.example_metrics
                if "accuracy" in m.custom_metrics
                and m.custom_metrics["accuracy"] is not None
            ]
            if accuracy_scores:
                accuracy_value = sum(accuracy_scores) / len(accuracy_scores)
        else:
            # Fallback to success rate if no example metrics
            accuracy_value = safe_get(aggregated, "success_rate")

        if accuracy_value is None and not strict_nulls:
            accuracy_value = 0.0

        duration_value = safe_get(aggregated, "duration")

        # ``cost`` MUST be the per-trial TOTAL (sum of per-example spend), NOT the
        # per-example mean (finding T2). The minimize-cost objective
        # (``orchestrator._extract_objective_values``), the results table, and the
        # portal all read this single ``cost`` key, yet the other two lanes emit a
        # TOTAL: the hybrid lane sets ``cost = total_cost`` (hybrid_api.py) and the
        # pruned lane sets ``cost`` to a cumulative partial sum
        # (trial_result_factory.py). Emitting the MEAN here made a completed trial
        # look ~N× cheaper than a pruned trial in the SAME run (biasing selection
        # and the "best cost") and ~N× cheaper than the hybrid lane for the same
        # config. Summing the per-example ``total_cost`` yields the same value the
        # authoritative ``total_cost`` is built from (see
        # ``orchestrator_helpers.extract_cost_from_results``), so per-config
        # ``cost`` now reconciles with ``total_cost`` instead of diverging by ~N.
        # The per-example mean is still useful and is preserved verbatim under the
        # distinct ``cost_per_example_mean`` key — ``cost`` is never overloaded.
        cost_per_example_mean = safe_get(aggregated, "total_cost", "mean")
        successful_metrics = [m for m in self.example_metrics if m.success]
        cost_total: float | None
        if successful_metrics:
            cost_total = sum(float(m.cost.total_cost) for m in successful_metrics)
        else:
            # Nothing to sum: mirror the mean's null/zero default so the strict-
            # nulls contract (None) and the normal contract (0.0) are preserved.
            cost_total = cost_per_example_mean

        formatted = {
            # Core metrics (single values)
            "score": accuracy_value,  # Use actual accuracy for score
            "accuracy": accuracy_value,  # Use actual accuracy
            "duration": duration_value,
            "execution_time_ms": (
                duration_value * 1000.0
                if isinstance(duration_value, (int, float))
                else duration_value
            ),
            # Use mean values as the primary metrics (without suffix)
            "input_tokens": safe_get(aggregated, "input_tokens", "mean"),
            "output_tokens": safe_get(aggregated, "output_tokens", "mean"),
            "total_tokens": safe_get(aggregated, "total_tokens", "mean"),
            "response_time_ms": safe_get(aggregated, "response_time_ms", "mean"),
            # Per-trial TOTAL cost (consistent scale across all three lanes).
            "cost": cost_total,
            # Per-example MEAN cost, preserved under a distinct key.
            "cost_per_example_mean": cost_per_example_mean,
            # Additional useful metrics
            "total_examples": aggregated["total_examples"],
            "successful_examples": aggregated["successful_examples"],
        }

        # Add tokens per second if available
        if "tokens_per_second" in aggregated:
            formatted["tokens_per_second"] = safe_get(
                aggregated, "tokens_per_second", "mean"
            )

        # Mean-aggregate user-supplied per-example custom metrics (e.g. the
        # composite_* telemetry keys from a (output, metrics_dict) return) the
        # same way accuracy aggregates above. These are rates/means/counts, so
        # mean across examples is the correct trial-level value. Standard keys
        # already produced above are excluded so we never double-count or
        # clobber the canonical aggregation.
        self._aggregate_user_custom_metrics(formatted, extra_reserved)

        return formatted

    def _aggregate_user_custom_metrics(
        self, formatted: dict[str, Any], extra_reserved: frozenset[str] = frozenset()
    ) -> None:
        """Mean-aggregate non-reserved custom metrics into ``formatted``.

        Delegates to the shared :func:`aggregate_user_custom_metrics` helper so
        the tuple-return metrics channel behaves identically on every public
        evaluator path: reserved evaluator-computed keys (the static
        :data:`RESERVED_METRIC_KEYS` plus the evaluator's runtime-only
        ``extra_reserved`` names) are skipped (never overwritten), already-present
        keys are left untouched, non-finite values are dropped, and the total key
        count is capped at :data:`TOTAL_MEASURES_CEILING` by truncating user keys
        only. Reserved-key skips here log at ``DEBUG``: this pass is fed the
        evaluator's own per-example ``custom_metrics`` (incl. injected cost/token
        keys), so a collision is evaluator-internal, not a user error.
        """
        if not self.example_metrics:
            return
        aggregate_user_custom_metrics(
            formatted,
            [m.custom_metrics for m in self.example_metrics if m.success],
            context="format_for_backend user metrics",
            extra_reserved=extra_reserved,
        )

    def _empty_backend_format(self) -> dict[str, Any]:
        """Return empty backend format structure when error occurs."""
        # Check for strict metrics nulls mode
        strict_nulls = os.environ.get("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
            "true",
            "1",
            "yes",
        )
        missing_default = None if strict_nulls else 0.0

        return {
            "score": missing_default,
            "accuracy": missing_default,
            "duration": missing_default,
            "input_tokens": missing_default,
            "output_tokens": missing_default,
            "total_tokens": missing_default,
            "response_time_ms": missing_default,
            "cost": missing_default,
            "total_examples": 0,
            "successful_examples": 0,
        }

    def format_as_summary_stats(self) -> dict[str, Any]:
        """Format metrics as pandas.describe()-compatible summary statistics.

        This format is used for privacy-preserving mode where individual
        results are not transmitted, only aggregated statistics.

        Returns:
            Dictionary with summary_stats structure matching pandas.describe()
        """
        if not self.example_metrics:
            return self._empty_summary_stats()

        # Filter successful examples for metrics calculation
        successful_metrics = [m for m in self.example_metrics if m.success]

        if not successful_metrics:
            return self._empty_summary_stats()

        # Extract values for each metric type
        # For accuracy, use custom metrics when available; otherwise derive from success flags
        accuracy_values = []
        for metric in self.example_metrics:
            if "accuracy" in metric.custom_metrics:
                accuracy_values.append(metric.custom_metrics["accuracy"] or 0.0)
            else:
                accuracy_values.append(1.0 if metric.success else 0.0)

        metrics_data: dict[str, list[int | float]] = {
            "accuracy": accuracy_values,
            "input_tokens": [m.tokens.input_tokens for m in successful_metrics],
            "output_tokens": [m.tokens.output_tokens for m in successful_metrics],
            "total_tokens": [m.tokens.total_tokens for m in successful_metrics],
            "response_time_ms": [
                m.response.response_time_ms for m in successful_metrics
            ],
            "input_cost": [m.cost.input_cost for m in successful_metrics],
            "output_cost": [m.cost.output_cost for m in successful_metrics],
            "total_cost": [m.cost.total_cost for m in successful_metrics],
        }

        # Add tokens per second if available
        tps_values = [
            m.response.tokens_per_second
            for m in successful_metrics
            if m.response.tokens_per_second is not None
        ]
        if tps_values:
            metrics_data["tokens_per_second"] = tps_values

        # Add any other custom metrics that appear in all examples
        if self.example_metrics:
            # Find common custom metrics across all examples
            all_custom_keys: set[str] = set()
            for m in self.example_metrics:
                all_custom_keys.update(m.custom_metrics.keys())

            # Add custom metrics that we haven't already handled
            for key in all_custom_keys:
                if (
                    key not in metrics_data and key != "accuracy"
                ):  # Skip accuracy as we've already handled it
                    custom_values = []
                    for m in self.example_metrics:
                        if key in m.custom_metrics:
                            custom_values.append(m.custom_metrics[key])
                        else:
                            custom_values.append(
                                0.0
                            )  # Default value if metric not present
                    if custom_values:
                        metrics_data[key] = custom_values

        # Generate pandas.describe()-compatible statistics for each metric
        summary_metrics = {}
        for metric_name, values in metrics_data.items():
            if values:
                summary_metrics[metric_name] = self._calculate_describe_stats(values)

        # Build the complete summary_stats structure
        summary_stats = {
            "metrics": summary_metrics,
            "execution_time": self.get_duration(),
            "total_examples": len(self.example_metrics),
            "metadata": {
                "sdk_version": get_version(),
                "aggregation_method": "pandas.describe",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }

        return summary_stats

    def _empty_summary_stats(self) -> dict[str, Any]:
        """Return empty summary stats structure."""
        empty_describe = {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "25%": 0.0,
            "50%": 0.0,
            "75%": 0.0,
            "max": 0.0,
        }

        return {
            "metrics": {
                "accuracy": empty_describe.copy(),
                "input_tokens": empty_describe.copy(),
                "output_tokens": empty_describe.copy(),
                "total_tokens": empty_describe.copy(),
                "response_time_ms": empty_describe.copy(),
                "total_cost": empty_describe.copy(),
            },
            "execution_time": 0.0,
            "total_examples": 0,
            "metadata": {
                "sdk_version": get_version(),
                "aggregation_method": "pandas.describe",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }

    def _calculate_describe_stats(
        self, values: Sequence[int | float]
    ) -> dict[str, float]:
        """Calculate pandas.describe()-compatible statistics.

        Returns statistics in the exact format of pandas.DataFrame.describe():
        count, mean, std, min, 25%, 50%, 75%, max
        """
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "25%": 0.0,
                "50%": 0.0,
                "75%": 0.0,
                "max": 0.0,
            }

        sorted_values = sorted(values)
        n = len(values)

        # Calculate percentiles
        def percentile(data, p):
            """Calculate percentile using linear interpolation (same as pandas)."""
            if not data:
                return 0.0
            try:
                k = (len(data) - 1) * p
                f = int(k)
                c = k - f
                if f + 1 < len(data):
                    return float(data[f] * (1 - c) + data[f + 1] * c)
                return float(data[f])
            except (IndexError, TypeError, ValueError) as e:
                logger.warning(f"Error calculating percentile: {e}")
                return 0.0

        return {
            "count": n,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if n > 1 else 0.0,
            "min": min(values),
            "25%": percentile(sorted_values, 0.25),
            "50%": percentile(sorted_values, 0.50),  # median
            "75%": percentile(sorted_values, 0.75),
            "max": max(values),
        }


# Response Handler Hierarchy


class ResponseHandler(ABC):
    """Base class for handling different LLM provider response formats."""

    def __init__(self, next_handler: Optional["ResponseHandler"] = None) -> None:
        self._next_handler = next_handler

    @abstractmethod
    def can_handle(self, response: Any) -> bool:
        """Check if this handler can process the response."""
        pass

    @abstractmethod
    def extract_tokens(self, response: Any) -> TokenMetrics:
        """Extract token metrics from response."""
        pass

    @abstractmethod
    def extract_response_time(self, response: Any) -> float:
        """Extract response time in milliseconds."""
        pass

    def extract_metadata_cost(self, response: Any) -> CostMetrics:
        """Extract cost information from response metadata if available.

        Checks the following sources in order:
        1. ``response.cost`` — generic cost attribute (dict or scalar).
        2. ``response._hidden_params['response_cost']`` — LiteLLM sets this for
           OpenRouter and other providers that return per-call cost directly.
           OpenRouter models are often missing from LiteLLM's pricing table, so
           their ``_hidden_params['response_cost']`` is the only reliable source.
        3. ``response.usage.cost`` — LiteLLM's Usage object exposes ``cost`` when
           the provider (e.g. OpenRouter) includes it in the usage block.
        """
        cost_metrics = CostMetrics()

        # 1. Generic ``response.cost`` attribute (dict or scalar).
        if hasattr(response, "cost"):
            try:
                if isinstance(response.cost, dict):
                    cost_metrics.input_cost = response.cost.get("input", 0.0)
                    cost_metrics.output_cost = response.cost.get("output", 0.0)
                    cost_metrics.total_cost = response.cost.get("total", 0.0)
                else:
                    cost_metrics.total_cost = float(response.cost)
            except (TypeError, ValueError) as e:
                logger.debug(f"Failed to parse cost from response: {e}")

        if cost_metrics.total_cost > 0.0:
            return cost_metrics

        # 2. LiteLLM hidden params — OpenRouter and other providers that report
        #    per-call cost populate ``_hidden_params['response_cost']``.
        hidden_params = getattr(response, "_hidden_params", None)
        if hidden_params is not None:
            try:
                response_cost = (
                    hidden_params.get("response_cost")
                    if hasattr(hidden_params, "get")
                    else getattr(hidden_params, "response_cost", None)
                )
                if isinstance(response_cost, (int, float)) and response_cost > 0:
                    cost_metrics.total_cost = float(response_cost)
                    logger.debug(
                        "Extracted cost $%.6f from _hidden_params.response_cost "
                        "(OpenRouter/LiteLLM provider-reported cost).",
                        cost_metrics.total_cost,
                    )
                    return cost_metrics
            except Exception as e:  # pragma: no cover
                logger.debug(f"Failed to parse cost from _hidden_params: {e}")

        # 3. LiteLLM Usage.cost field — set for some provider responses.
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                usage_cost = getattr(usage, "cost", None)
                if isinstance(usage_cost, (int, float)) and usage_cost > 0:
                    cost_metrics.total_cost = float(usage_cost)
                    logger.debug(
                        "Extracted cost $%.6f from usage.cost "
                        "(LiteLLM provider-reported cost).",
                        cost_metrics.total_cost,
                    )
                    return cost_metrics
            except Exception as e:  # pragma: no cover
                logger.debug(f"Failed to parse cost from usage.cost: {e}")

        return cost_metrics

    def extract_metadata_info(self, response: Any, metrics: ExampleMetrics) -> None:
        """Extract additional metrics from metadata."""
        if not hasattr(response, "metadata") or not isinstance(response.metadata, dict):
            return

        metadata = response.metadata

        # Extract token information
        if "tokens" in metadata and isinstance(metadata["tokens"], dict):
            tokens = metadata["tokens"]
            metrics.tokens.input_tokens = tokens.get(
                "input", metrics.tokens.input_tokens
            )
            metrics.tokens.output_tokens = tokens.get(
                "output", metrics.tokens.output_tokens
            )
            metrics.tokens.total_tokens = (
                metrics.tokens.input_tokens + metrics.tokens.output_tokens
            )

        # Extract cost information
        if "cost" in metadata:
            cost = metadata["cost"]
            if isinstance(cost, dict):
                metrics.cost.input_cost = cost.get("input", metrics.cost.input_cost)
                metrics.cost.output_cost = cost.get("output", metrics.cost.output_cost)
                metrics.cost.total_cost = cost.get("total", metrics.cost.total_cost)
            else:
                try:
                    metrics.cost.total_cost = float(cost)
                except (TypeError, ValueError):
                    pass

        # Extract response time
        if "response_time_ms" in metadata:
            try:
                metrics.response.response_time_ms = float(metadata["response_time_ms"])
            except (TypeError, ValueError):
                pass

    def handle(self, response: Any) -> ExampleMetrics | None:
        """Handle the response or pass to next handler."""
        if self.can_handle(response):
            metrics = ExampleMetrics()
            metrics.tokens = self.extract_tokens(response)
            metrics.response.response_time_ms = self.extract_response_time(response)

            # Extract cost from response if available
            response_cost = self.extract_metadata_cost(response)
            if response_cost.total_cost > 0:
                metrics.cost = response_cost

            # Extract additional metadata
            self.extract_metadata_info(response, metrics)

            return metrics
        elif self._next_handler:
            return self._next_handler.handle(response)
        else:
            return None


class OpenAIResponseHandler(ResponseHandler):
    """Handler for OpenAI ChatCompletion responses."""

    def can_handle(self, response: Any) -> bool:
        # More specific check for OpenAI - look for prompt_tokens and completion_tokens
        if not hasattr(response, "usage"):
            return False
        usage = response.usage
        return hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens")

    def extract_tokens(self, response: Any) -> TokenMetrics:
        tokens = TokenMetrics()
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens") and isinstance(
                usage.prompt_tokens, (int, float)
            ):
                tokens.input_tokens = int(usage.prompt_tokens)
            if hasattr(usage, "completion_tokens") and isinstance(
                usage.completion_tokens, (int, float)
            ):
                tokens.output_tokens = int(usage.completion_tokens)
            if hasattr(usage, "total_tokens") and isinstance(
                usage.total_tokens, (int, float)
            ):
                tokens.total_tokens = int(usage.total_tokens)
            else:
                tokens.total_tokens = tokens.input_tokens + tokens.output_tokens
        return tokens

    def extract_response_time(self, response: Any) -> float:
        if hasattr(response, "response_time_ms") and isinstance(
            response.response_time_ms, (int, float)
        ):
            return float(response.response_time_ms)
        elif hasattr(response, "_response_time"):
            return cast(float, response._response_time)
        return 0.0


class AnthropicResponseHandler(ResponseHandler):
    """Handler for Anthropic API responses."""

    @staticmethod
    def _has_usage_attr(usage: Any, name: str) -> bool:
        if isinstance(usage, dict):
            return name in usage
        return hasattr(usage, name)

    def _is_anthropic_message(self, response: Any) -> bool:
        """Check for explicit Anthropic Message class with claude model."""
        if type(response).__name__ != "Message" or not hasattr(response, "content"):
            return False
        return (
            hasattr(response, "model")
            and "claude" in str(getattr(response, "model", "")).lower()
        )

    def _has_anthropic_usage_pattern(self, response: Any) -> bool:
        """Check for Anthropic-specific usage token keys (not OpenAI)."""
        if not hasattr(response, "usage"):
            return False
        usage = response.usage
        _has = self._has_usage_attr
        has_anthropic_tokens = (
            (_has(usage, "input_tokens") and _has(usage, "output_tokens"))
            or (_has(usage, "num_input_tokens") and _has(usage, "num_output_tokens"))
            or (_has(usage, "inputTokens") and _has(usage, "outputTokens"))
        )
        has_openai_tokens = _has(usage, "prompt_tokens") and _has(
            usage, "completion_tokens"
        )
        return has_anthropic_tokens and not has_openai_tokens

    def _is_claude_model_without_choices(self, response: Any) -> bool:
        """Check for model name containing claude without OpenAI choices attr."""
        if not hasattr(response, "model"):
            return False
        if "claude" not in str(getattr(response, "model", "")).lower():
            return False
        return not hasattr(response, "choices")

    def can_handle(self, response: Any) -> bool:
        # Check for Anthropic-specific attributes via three strategies
        return (
            self._is_anthropic_message(response)
            or self._has_anthropic_usage_pattern(response)
            or self._is_claude_model_without_choices(response)
        )

    def extract_tokens(self, response: Any) -> TokenMetrics:
        tokens = TokenMetrics()
        if hasattr(response, "usage"):
            usage = response.usage

            # Support both object-style and dict-style usage
            def _get(u: Any, name: str) -> Any:
                if isinstance(u, dict):
                    return u.get(name)
                return getattr(u, name, None)

            input_val = (
                _get(usage, "input_tokens")
                or _get(usage, "num_input_tokens")
                or _get(usage, "inputTokens")
                or _get(usage, "prompt_tokens")
            )
            output_val = (
                _get(usage, "output_tokens")
                or _get(usage, "num_output_tokens")
                or _get(usage, "outputTokens")
                or _get(usage, "completion_tokens")
            )
            total_val = _get(usage, "total_tokens")

            if isinstance(input_val, (int, float)):
                tokens.input_tokens = int(input_val)
            if isinstance(output_val, (int, float)):
                tokens.output_tokens = int(output_val)
            if isinstance(total_val, (int, float)):
                tokens.total_tokens = int(total_val)
            else:
                tokens.total_tokens = tokens.input_tokens + tokens.output_tokens
        return tokens

    def extract_response_time(self, response: Any) -> float:
        if hasattr(response, "response_time_ms") and isinstance(
            response.response_time_ms, (int, float)
        ):
            return float(response.response_time_ms)
        elif hasattr(response, "latency") and isinstance(
            response.latency, (int, float)
        ):
            return float(response.latency) * 1000  # Convert to ms
        return 0.0


class LangChainResponseHandler(ResponseHandler):
    """Handler for LangChain response objects."""

    def can_handle(self, response: Any) -> bool:
        # Check for LangChain-specific patterns
        return (
            hasattr(response, "llm_output")
            or hasattr(response, "generations")
            or hasattr(response, "usage_metadata")  # LangChain AIMessage
            or hasattr(response, "response_metadata")  # LangChain ChatOpenAI responses
            or str(type(response).__module__).startswith("langchain")
        )

    def extract_tokens(self, response: Any) -> TokenMetrics:
        tokens = TokenMetrics()

        # Check usage_metadata for token usage (langchain_core AIMessage)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            tokens.input_tokens = usage.get("input_tokens", 0)
            tokens.output_tokens = usage.get("output_tokens", 0)
            tokens.total_tokens = usage.get(
                "total_tokens", tokens.input_tokens + tokens.output_tokens
            )
            if tokens.total_tokens > 0:
                return tokens

        # Check response_metadata for token usage (alternative format)
        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata

            # Check for token_usage field (ChatOpenAI format)
            if "token_usage" in metadata and isinstance(metadata["token_usage"], dict):
                usage = metadata["token_usage"]
                tokens.input_tokens = usage.get("prompt_tokens", 0)
                tokens.output_tokens = usage.get("completion_tokens", 0)
                tokens.total_tokens = usage.get(
                    "total_tokens", tokens.input_tokens + tokens.output_tokens
                )
                if tokens.total_tokens > 0:
                    return tokens

            # Check for usage field (alternative format)
            if "usage" in metadata and isinstance(metadata["usage"], dict):
                usage = metadata["usage"]
                input_val = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                output_val = usage.get(
                    "output_tokens", usage.get("completion_tokens", 0)
                )
                tokens.input_tokens = int(input_val) if input_val is not None else 0
                tokens.output_tokens = int(output_val) if output_val is not None else 0
                total_val = usage.get(
                    "total_tokens", tokens.input_tokens + tokens.output_tokens
                )
                tokens.total_tokens = (
                    int(total_val)
                    if total_val is not None
                    else tokens.input_tokens + tokens.output_tokens
                )
                if tokens.total_tokens > 0:
                    return tokens

        # Check llm_output for token usage (older format)
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            llm_output = response.llm_output
            if "token_usage" in llm_output:
                usage = llm_output["token_usage"]
                tokens.input_tokens = usage.get("prompt_tokens", 0)
                tokens.output_tokens = usage.get("completion_tokens", 0)
                tokens.total_tokens = usage.get(
                    "total_tokens", tokens.input_tokens + tokens.output_tokens
                )

        return tokens

    def extract_response_time(self, response: Any) -> float:
        # First check response_metadata for injected timing
        if hasattr(response, "response_metadata") and isinstance(
            response.response_metadata, dict
        ):
            if "response_time_ms" in response.response_metadata:
                try:
                    return float(response.response_metadata["response_time_ms"])
                except (TypeError, ValueError):
                    pass

        # Fall back to direct attribute
        if hasattr(response, "response_time_ms"):
            return float(response.response_time_ms)

        return 0.0


class DictResponseHandler(ResponseHandler):
    """Handler for dictionary responses with token counts."""

    def can_handle(self, response: Any) -> bool:
        if not isinstance(response, dict):
            return False
        # Check if it has token-related keys
        return any(
            key in response
            for key in ["input_tokens", "output_tokens", "total_tokens", "usage"]
        )

    def extract_tokens(self, response: Any) -> TokenMetrics:
        """Extract tokens from dict response."""
        if not isinstance(response, dict):
            return TokenMetrics()

        # Direct token fields
        input_tokens = response.get("input_tokens", 0)
        output_tokens = response.get("output_tokens", 0)
        total_tokens = response.get("total_tokens", 0)

        # Check for usage field (OpenAI style)
        if "usage" in response and isinstance(response["usage"], dict):
            usage = response["usage"]
            input_tokens = usage.get(
                "prompt_tokens", usage.get("input_tokens", input_tokens)
            )
            output_tokens = usage.get(
                "completion_tokens", usage.get("output_tokens", output_tokens)
            )
            total_tokens = usage.get("total_tokens", total_tokens)

        # If total_tokens is not provided, calculate it
        if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
            total_tokens = input_tokens + output_tokens

        return TokenMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def extract_response_time(self, response: Any) -> float:
        """Extract response time from dict."""
        if not isinstance(response, dict):
            return 0.0
        # Check common keys for response time
        for key in ["response_time_ms", "response_time", "latency", "duration"]:
            if key in response:
                value = response[key]
                if isinstance(value, (int, float)):
                    # Convert to ms if needed
                    if key == "latency" or key == "duration":
                        return float(value * 1000)
                    return float(value)
        return 0.0


class GenericResponseHandler(ResponseHandler):
    """Fallback handler for any response format."""

    def can_handle(self, response: Any) -> bool:
        return True  # Always can handle as fallback

    def extract_tokens(self, response: Any) -> TokenMetrics:
        metadata = getattr(response, "metadata", None)
        if isinstance(metadata, dict):
            usage = metadata.get("usage") or metadata.get("token_usage")
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                output_tokens = usage.get(
                    "output_tokens", usage.get("completion_tokens", 0)
                )
                total_tokens = usage.get(
                    "total_tokens", (input_tokens or 0) + (output_tokens or 0)
                )
                return TokenMetrics(
                    input_tokens=int(input_tokens or 0),
                    output_tokens=int(output_tokens or 0),
                    total_tokens=int(total_tokens or 0),
                )
        return TokenMetrics()  # Return empty metrics

    def extract_response_time(self, response: Any) -> float:
        # Try common response time attributes
        for attr in ["response_time_ms", "latency", "_response_time"]:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if isinstance(value, (int, float)):
                    return float(value * 1000 if attr == "latency" else value)
        return 0.0


class ResponseHandlerFactory:
    """Factory for creating response handler chains."""

    @staticmethod
    def create_handler_chain() -> ResponseHandler:
        """Create a chain of response handlers in order of specificity."""
        generic = GenericResponseHandler()
        dict_handler = DictResponseHandler(generic)
        langchain = LangChainResponseHandler(dict_handler)
        anthropic = AnthropicResponseHandler(langchain)
        openai = OpenAIResponseHandler(anthropic)
        return openai


def _infer_model_name_from_response(response: Any) -> str | None:
    """Infer a model name from an LLM response when the config supplies none.

    Multi-model/multi-step agents often have no single ``model`` key in the
    optimization config, so the central cost-calculation chokepoint must be
    able to recover the model from the response itself (the SDK-side
    equivalent of ``LocalEvaluator._infer_model_name_from_output`` /
    ``CustomEvaluatorWrapper``'s response fallback, but shared by every path
    that funnels through ``extract_llm_metrics``, e.g. #1599).

    Must never raise: a response whose ``model``/``response_metadata``/
    ``llm_output`` accessor is a property that raises on access would
    otherwise propagate out of ``extract_llm_metrics`` and abort cost
    calculation entirely, instead of the intended "skip cost, log a
    warning" degradation.
    """
    try:
        # 1. Direct attributes (e.g. OpenAI ChatCompletion, Anthropic Message)
        for attr in ("model", "model_name"):
            val = getattr(response, attr, None)
            if isinstance(val, str) and val:
                return val

        # 2. Dict-shaped response
        if isinstance(response, dict):
            for key in ("model", "model_name"):
                val = response.get(key)
                if isinstance(val, str) and val:
                    return val

        # 3. LangChain response_metadata
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            for key in ("model", "model_name"):
                val = response_metadata.get(key)
                if isinstance(val, str) and val:
                    return val

        # 4. LangChain llm_output
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            for key in ("model", "model_name"):
                val = llm_output.get(key)
                if isinstance(val, str) and val:
                    return val

        return None
    except Exception:
        return None


def _calculate_cost_for_metrics(
    metrics: ExampleMetrics,
    model_name: str | None,
    original_prompt: Any,
    response_text: str | None,
    prompt_length: int | None = None,
    response_length: int | None = None,
) -> None:
    """Calculate and update cost metrics.

    Uses cost_from_tokens() as the canonical cost path when token counts are
    available, falling back to deprecated text-based functions otherwise.
    """
    from traigent.utils.env_config import is_strict_cost_accounting

    strict_cost_accounting = is_strict_cost_accounting()
    # NOTE: ``TRAIGENT_MOCK_LLM`` no longer suppresses cost calculation here
    # (S2-B retirement of the mock flag). Cost is always computed from the
    # real token counts so that production traces reflect real spend even if
    # a stale env var leaks into a deployed environment. ``TRAIGENT_GENERATE_MOCKS``
    # is preserved because it is an internal fixture-recording knob, not a
    # user-facing mock-LLM toggle.
    generate_mocks_env = os.environ.get("TRAIGENT_GENERATE_MOCKS", "").lower()

    if generate_mocks_env == "true":
        _handle_mock_mode(metrics, prompt_length, response_length)
        return

    if metrics.cost.total_cost > 0.0:
        _reconcile_reported_cost_with_tokens(metrics, model_name)
        return

    if not model_name:
        logger.warning(
            "Cost calculation skipped: model_name is None/empty. "
            "Ensure the optimization config includes a 'model' key or "
            "the LLM response exposes the model name. "
            "tokens=(in=%d, out=%d)",
            metrics.tokens.input_tokens,
            metrics.tokens.output_tokens,
        )
        return

    try:
        _compute_cost(
            metrics,
            model_name,
            original_prompt,
            response_text,
            strict_cost_accounting=strict_cost_accounting,
        )

        # Log when cost ends up at zero despite having tokens — this is the
        # most actionable diagnostic for the "$0 cost" issue (#325).
        if math.isclose(metrics.cost.total_cost, 0.0, abs_tol=1e-12) and (
            metrics.tokens.input_tokens > 0 or metrics.tokens.output_tokens > 0
        ):
            logger.warning(
                "Cost is $0.00 despite non-zero tokens for model %r "
                "(in=%d, out=%d). Model may be missing from pricing tables.",
                model_name,
                metrics.tokens.input_tokens,
                metrics.tokens.output_tokens,
            )
    except Exception as e:
        logger.error(
            "Cost calculation failed for model %s: %s (tokens: in=%d, out=%d)",
            model_name,
            e,
            metrics.tokens.input_tokens,
            metrics.tokens.output_tokens,
            exc_info=True,
        )
        if (
            strict_cost_accounting
            or os.environ.get("TRAIGENT_DEBUG", "").lower() == "true"
        ):
            raise


def _token_derived_cost(
    metrics: ExampleMetrics, model_name: str | None
) -> tuple[float, float, float] | None:
    if not model_name:
        return None
    in_tokens = metrics.tokens.input_tokens
    out_tokens = metrics.tokens.output_tokens
    if in_tokens <= 0 and out_tokens <= 0:
        return None

    from traigent.utils.cost_calculator import cost_from_tokens

    try:
        input_cost, output_cost = cost_from_tokens(
            in_tokens, out_tokens, model_name, strict=False
        )
    except Exception:  # pragma: no cover - defensive; never break the trial
        logger.debug(
            "Cost breakdown backfill failed for model %r", model_name, exc_info=True
        )
        return None

    derived_total = input_cost + output_cost
    if derived_total <= 0.0:
        return None
    return input_cost, output_cost, derived_total


def _reconcile_reported_cost_with_tokens(
    metrics: ExampleMetrics, model_name: str | None
) -> None:
    """Backfill plausible reported costs and clamp implausible under-reports.

    Provider-reported and user-injected cost paths set ``total_cost`` directly
    and often leave the per-trial breakdown at 0.0. When real token counts and
    known model pricing are available, derive a canonical token-cost estimate.
    Reported costs remain authoritative unless they are implausibly below that
    estimate. If token data or pricing is unavailable, this is the residual BYOK
    trust boundary: accept the caller/provider-reported value as-is.
    """
    derived = _token_derived_cost(metrics, model_name)
    if derived is None:
        return

    input_cost, output_cost, derived_total = derived
    authoritative_total = metrics.cost.total_cost
    if authoritative_total < (derived_total * REPORTED_COST_PLAUSIBILITY_FLOOR_RATIO):
        logger.warning(
            "Reported LLM cost $%.8f for model %r is implausibly below token-derived "
            "estimate $%.8f (tokens: in=%d, out=%d); using token-derived cost "
            "for runtime enforcement.",
            authoritative_total,
            model_name,
            derived_total,
            metrics.tokens.input_tokens,
            metrics.tokens.output_tokens,
        )
        metrics.cost.input_cost = input_cost
        metrics.cost.output_cost = output_cost
        metrics.cost.total_cost = derived_total
        return

    _backfill_cost_breakdown(metrics, input_cost, output_cost, derived_total)


def _backfill_cost_breakdown(
    metrics: ExampleMetrics,
    input_cost: float,
    output_cost: float,
    derived_total: float,
) -> None:
    """Populate input_cost/output_cost when only total_cost is known (#1423)."""
    if metrics.cost.input_cost > 0.0 or metrics.cost.output_cost > 0.0:
        return

    authoritative_total = metrics.cost.total_cost
    # Scale the derived split to match the authoritative total so the two stay
    # consistent (provider-reported totals can differ slightly from token math).
    scale = authoritative_total / derived_total
    metrics.cost.input_cost = input_cost * scale
    metrics.cost.output_cost = output_cost * scale
    metrics.cost.total_cost = authoritative_total
    logger.debug(
        "Backfilled cost breakdown: in=$%.6f out=$%.6f (total=$%.6f)",
        metrics.cost.input_cost,
        metrics.cost.output_cost,
        metrics.cost.total_cost,
    )


def _handle_mock_mode(
    metrics: ExampleMetrics,
    prompt_length: int | None,
    response_length: int | None,
) -> None:
    """Set zero costs in mock mode, estimating tokens from text length."""
    if prompt_length is not None and metrics.tokens.input_tokens == 0:
        metrics.tokens.input_tokens = max(1, prompt_length // 4)
    if response_length is not None and metrics.tokens.output_tokens == 0:
        metrics.tokens.output_tokens = max(1, response_length // 4)
    if metrics.tokens.input_tokens > 0 or metrics.tokens.output_tokens > 0:
        metrics.tokens.total_tokens = (
            metrics.tokens.input_tokens + metrics.tokens.output_tokens
        )
    metrics.cost.input_cost = 0.0
    metrics.cost.output_cost = 0.0
    metrics.cost.total_cost = 0.0


def _compute_cost(
    metrics: ExampleMetrics,
    model_name: str,
    original_prompt: Any,
    response_text: str | None,
    *,
    strict_cost_accounting: bool,
) -> None:
    """Compute cost using cost_from_tokens (preferred) or legacy text path."""
    from traigent.utils.cost_calculator import cost_from_tokens

    # Ensure total tokens computed
    if metrics.tokens.total_tokens == 0:
        metrics.tokens.total_tokens = (
            metrics.tokens.input_tokens + metrics.tokens.output_tokens
        )

    if metrics.tokens.input_tokens > 0 or metrics.tokens.output_tokens > 0:
        # Preferred path: use canonical cost_from_tokens directly
        input_cost, output_cost = cost_from_tokens(
            metrics.tokens.input_tokens,
            metrics.tokens.output_tokens,
            model_name,
            strict=strict_cost_accounting,
        )
        metrics.cost.input_cost = input_cost
        metrics.cost.output_cost = output_cost
        metrics.cost.total_cost = input_cost + output_cost
        if metrics.cost.total_cost > 0:
            logger.debug(
                "Cost calculated for %s: $%.6f via cost_from_tokens",
                model_name,
                metrics.cost.total_cost,
            )
        else:
            # Non-strict path: the model priced to $0 despite non-zero tokens
            # (litellm + custom + built-in tables all miss). Record it so the
            # optimizer can surface a result-level warning to the USER instead
            # of leaving only a buried log (#1407). Strict accounting never
            # reaches here — ``cost_from_tokens(strict=True)`` raises above.
            from traigent.utils.cost_calculator import record_unpriced_runtime_model

            record_unpriced_runtime_model(model_name)
            # Mark this example's cost as "unknown", not "verified free", so
            # trial/result-level aggregation (#1597) does not silently treat
            # the unpriced usage as zero-cost when reporting total spend.
            metrics.cost.unpriced = True
    else:
        logger.debug(
            "No token usage extracted for model %s; "
            "falling back to deprecated text-based cost calculation",
            model_name,
        )
        # Legacy fallback: text-based cost (deprecated)
        _try_legacy_cost_calculation(
            metrics,
            model_name,
            original_prompt,
            response_text,
        )


def _try_legacy_cost_calculation(
    metrics: ExampleMetrics,
    model_name: str,
    original_prompt: Any,
    response_text: str | None,
) -> None:
    """Legacy cost calculation using deprecated text-based functions."""
    backward_compatible = False

    if calculate_prompt_cost is not None and original_prompt is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            metrics.cost.input_cost = calculate_prompt_cost(original_prompt, model_name)
        backward_compatible = True

    if calculate_completion_cost is not None and response_text is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            metrics.cost.output_cost = calculate_completion_cost(
                response_text, model_name
            )
        backward_compatible = True

    if backward_compatible:
        metrics.cost.total_cost = metrics.cost.input_cost + metrics.cost.output_cost


class CostCalculator:
    """Backward-compat shim — delegates to module-level functions.

    .. deprecated::
        Use ``_calculate_cost_for_metrics()`` or ``cost_from_tokens()`` directly.
    """

    def __init__(self) -> None:
        self.logger = logger

    def calculate_cost(
        self,
        metrics: ExampleMetrics,
        model_name: str | None,
        original_prompt: Any,
        response_text: str | None,
        prompt_length: int | None = None,
        response_length: int | None = None,
    ) -> None:
        _calculate_cost_for_metrics(
            metrics,
            model_name,
            original_prompt,
            response_text,
            prompt_length=prompt_length,
            response_length=response_length,
        )

    def _try_unified_cost_calculation(
        self,
        metrics: ExampleMetrics,
        model_name: str,
        original_prompt: Any,
        response_text: str | None,
        prompt_length: int | None = None,
        response_length: int | None = None,
    ) -> None:
        del prompt_length, response_length
        from traigent.utils.env_config import is_strict_cost_accounting

        _compute_cost(
            metrics,
            model_name,
            original_prompt,
            response_text,
            strict_cost_accounting=is_strict_cost_accounting(),
        )


class MetricsCalculator:
    """Helper for calculating derived metrics."""

    @staticmethod
    def calculate_tokens_per_second(metrics: ExampleMetrics) -> None:
        """Calculate and set tokens per second if possible."""
        try:
            if (
                isinstance(metrics.tokens.total_tokens, (int, float))
                and isinstance(metrics.response.response_time_ms, (int, float))
                and metrics.tokens.total_tokens > 0
                and metrics.response.response_time_ms
                > 0.001  # Minimum 1 microsecond to avoid division by zero
            ):
                metrics.response.tokens_per_second = metrics.tokens.total_tokens / (
                    metrics.response.response_time_ms / 1000
                )
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"Failed to calculate tokens per second: {e}")


def extract_llm_metrics(
    response: Any,
    model_name: str | None = None,
    original_prompt: Any | None = None,
    response_text: str | None = None,
    prompt_length: int | None = None,
    response_length: int | None = None,
) -> ExampleMetrics:
    """Extract metrics from LLM response objects with cost calculation.

    This function uses a handler pattern to extract metrics from various LLM response formats:
    - OpenAI ChatCompletion responses
    - Anthropic responses
    - LangChain responses
    - Custom response formats

    Args:
        response: The response object from an LLM call
        model_name: The model name used for the LLM call (for cost calculation)
        original_prompt: The original prompt sent to the LLM (for accurate cost calculation)
        response_text: The response text from the LLM (for accurate cost calculation)
        prompt_length: Length of prompt in privacy mode (alternative to original_prompt)
        response_length: Length of response in privacy mode (alternative to response_text)

    Returns:
        ExampleMetrics with extracted token, cost, and response metrics
    """
    # Create handler chain and extract basic metrics
    handler_chain = ResponseHandlerFactory.create_handler_chain()
    metrics = handler_chain.handle(response)

    if metrics is None:
        metrics = ExampleMetrics()
        logger.warning("No handler could process the response, using empty metrics")

    # Calculate cost using canonical cost_from_tokens path. Fall back to the
    # model name carried on the response itself when the config supplies none
    # (multi-model/multi-step agents have no single config 'model' key; #1599).
    effective_model_name = model_name or _infer_model_name_from_response(response)
    _calculate_cost_for_metrics(
        metrics,
        effective_model_name,
        original_prompt,
        response_text,
        prompt_length=prompt_length,
        response_length=response_length,
    )

    # Calculate derived metrics
    MetricsCalculator.calculate_tokens_per_second(metrics)

    return metrics
