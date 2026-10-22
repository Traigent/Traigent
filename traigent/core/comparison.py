"""Fixed-configuration comparison (``OptimizedFunction.compare``, issue #1952).

``compare()`` evaluates a list of *fully specified* configurations on the same
evaluation dataset, ``repeats`` times each, and returns ONE
:class:`ComparisonResult`. It is pure evaluation: there is no search, and no
parallel scoring path -- every (config, repeat) is one single-trial run of the
normal ``optimize()`` pipeline over a one-cell configuration space, so the same
evaluator, metric capture and honest-cost rules apply as for an optimization
run. This module holds the pure pieces (validation, dataset fingerprints,
aggregation, result types); the orchestration lives on ``OptimizedFunction``.

Honest-cost invariant: a config whose cost could not be measured in at least
one repeat reports ``total_cost=None`` plus ``cost_lower_bound`` (the known
spend, a floor) -- never ``0.0``.

Holdout rule: the result records how the evaluation dataset relates to the
decorated (tuning) dataset, by content fingerprint, so a "holdout" that
actually shares examples with the tuning data is flagged instead of trusted.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from traigent.utils.discrete_domains import (
    discrete_values_for_config_param,
    stepped_int_contains,
)

__all__ = [
    "ComparisonResult",
    "ConfigComparison",
    "HoldoutCheck",
    "example_fingerprint",
    "validate_comparison_configs",
]

#: Holdout statuses (``HoldoutCheck.status``).
HOLDOUT_TUNING_DATASET = "tuning_dataset"
HOLDOUT_DISJOINT = "holdout"
HOLDOUT_OVERLAPS = "overlaps_tuning_dataset"
HOLDOUT_UNKNOWN = "unknown"


def example_fingerprint(example: Any) -> str:
    """Stable content digest of one evaluation example (input + expected)."""
    payload = {
        "input": getattr(example, "input_data", None),
        "expected": getattr(example, "expected_output", None),
    }
    blob = json.dumps(payload, sort_keys=True, default=repr, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _in_numeric_range(
    value: Any, low: Any, high: Any, *, step: Any = None, integer: bool = False
) -> bool:
    """Bounds, integer-ness and step membership for one numeric domain."""
    if integer:
        if not _is_int(value) or not _is_int(low):
            return False
    elif not _is_number(value):
        return False
    if not (_is_number(low) and _is_number(high)) or not low <= value <= high:
        return False
    if step is None:
        return True
    if not _is_number(step) or step <= 0:
        return False
    if integer:
        return _is_int(step) and stepped_int_contains(low, high, step, value)
    k = round((value - low) / step)
    return k >= 0 and math.isclose(low + k * step, value, rel_tol=1e-9, abs_tol=1e-12)


def _value_in_domain(spec: Any, value: Any) -> bool:
    """True when ``value`` lies inside one normalized configuration-space entry.

    Finite domains (choice lists, categorical/fixed dicts) are resolved with the
    SDK's own ``discrete_values_for_config_param``; numeric ranges are checked
    arithmetically (integer-ness, bounds and ``step``) so a huge stepped range
    is never expanded.
    """
    if isinstance(spec, tuple) and len(spec) == 2 and all(_is_number(b) for b in spec):
        return _in_numeric_range(
            value, spec[0], spec[1], integer=all(_is_int(b) for b in spec)
        )
    if isinstance(spec, Mapping) and "low" in spec and "high" in spec:
        kind = str(spec.get("type") or "").lower()
        if kind in {"int", "integer"}:
            step = spec.get("step")
            return _in_numeric_range(
                value,
                spec["low"],
                spec["high"],
                step=1 if step is None else step,
                integer=True,
            )
        return _in_numeric_range(
            value, spec["low"], spec["high"], step=spec.get("step")
        )
    values = discrete_values_for_config_param(spec)
    if values is None:
        return False
    return any(
        value == option and isinstance(value, bool) == isinstance(option, bool)
        for option in values
    )


def validate_comparison_configs(
    configs: Sequence[Mapping[str, Any]],
    configuration_space: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate that each config is fully specified and inside the space.

    Errors are value-free: they name the config's position and the *declared*
    parameter names, never a client-supplied key or value.
    """
    if isinstance(configs, (str, bytes)) or not isinstance(configs, Sequence):
        raise ValueError("compare() requires configs to be a list of dicts.")
    if len(configs) == 0:
        raise ValueError("compare() requires at least one configuration.")
    if not configuration_space:
        raise ValueError(
            "compare() requires a declared configuration space to validate "
            "configurations against."
        )
    declared = sorted(configuration_space)
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, config in enumerate(configs):
        if not isinstance(config, Mapping):
            raise ValueError(f"Configuration #{index} must be a dict.")
        extra = [key for key in config if key not in configuration_space]
        if extra:
            raise ValueError(
                f"Configuration #{index} has {len(extra)} parameter(s) outside "
                f"the declared configuration space (declared: {declared})."
            )
        missing = [name for name in declared if name not in config]
        if missing:
            raise ValueError(
                f"Configuration #{index} is not fully specified; missing "
                f"declared parameter(s): {missing}."
            )
        bad = [
            name
            for name in declared
            if not _value_in_domain(configuration_space[name], config[name])
        ]
        if bad:
            raise ValueError(
                f"Configuration #{index} has value(s) outside the declared "
                f"configuration space for parameter(s): {bad}."
            )
        key = json.dumps({n: config[n] for n in declared}, sort_keys=True, default=repr)
        if key in seen:
            raise ValueError(f"Configuration #{index} duplicates an earlier one.")
        seen.add(key)
        cleaned.append({name: config[name] for name in declared})
    return cleaned


@dataclass(frozen=True)
class HoldoutCheck:
    """How the comparison dataset relates to the decorated (tuning) dataset."""

    status: str
    n_examples: int
    n_overlapping_examples: int | None
    holdout_valid: bool | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_examples": self.n_examples,
            "n_overlapping_examples": self.n_overlapping_examples,
            "holdout_valid": self.holdout_valid,
            "note": self.note,
        }


def build_holdout_check(
    eval_fingerprints: Sequence[str],
    tuning_fingerprints: Sequence[str] | None,
    *,
    eval_dataset_supplied: bool,
) -> HoldoutCheck:
    """Classify the evaluation dataset against the tuning dataset."""
    n = len(eval_fingerprints)
    if not eval_dataset_supplied:
        return HoldoutCheck(
            HOLDOUT_TUNING_DATASET,
            n,
            n,
            False,
            "Evaluated on the decorated (tuning) dataset: this is not a holdout. "
            "Pass eval_dataset=<held-out data> to check generalization.",
        )
    if tuning_fingerprints is None:
        return HoldoutCheck(
            HOLDOUT_UNKNOWN,
            n,
            None,
            None,
            "The tuning dataset could not be read, so holdout disjointness is "
            "unverified.",
        )
    overlap = len(set(eval_fingerprints) & set(tuning_fingerprints))
    if overlap:
        return HoldoutCheck(
            HOLDOUT_OVERLAPS,
            n,
            overlap,
            False,
            f"{overlap} of {n} comparison examples also appear in the tuning "
            "dataset: results are not a clean holdout.",
        )
    return HoldoutCheck(
        HOLDOUT_DISJOINT,
        n,
        0,
        True,
        "No comparison example appears in the tuning dataset.",
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _spread(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


@dataclass
class ConfigComparison:
    """Per-config aggregate over ``repeats`` evaluations."""

    config: dict[str, Any]
    repeats: int
    scores: list[float | None]
    mean_score: float | None
    score_stdev: float | None
    score_min: float | None
    score_max: float | None
    metrics_mean: dict[str, float]
    #: Spend over all repeats; ``None`` when any repeat's cost is unmeasured.
    total_cost: float | None
    #: Known spend (a floor) when ``total_cost`` is ``None``; else ``None``.
    cost_lower_bound: float | None
    cost_unmeasured_repeats: int
    failed_repeats: int
    example_ids: tuple[str, ...]
    trials: list[Any] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "repeats": self.repeats,
            "scores": list(self.scores),
            "mean_score": self.mean_score,
            "score_stdev": self.score_stdev,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "metrics_mean": dict(self.metrics_mean),
            "total_cost": self.total_cost,
            "cost_lower_bound": self.cost_lower_bound,
            "cost_unmeasured_repeats": self.cost_unmeasured_repeats,
            "failed_repeats": self.failed_repeats,
            "example_ids": list(self.example_ids),
        }


def aggregate_config(
    config: dict[str, Any],
    run_results: Sequence[Any],
    primary_metric: str,
    example_ids: tuple[str, ...],
) -> ConfigComparison:
    """Fold the single-trial ``OptimizationResult`` of each repeat into one row."""
    scores: list[float | None] = []
    metric_values: dict[str, list[float]] = {}
    known_cost = 0.0
    floor = 0.0
    unmeasured = 0
    failed = 0
    trials: list[Any] = []
    for result in run_results:
        trial = result.trials[0] if getattr(result, "trials", None) else None
        if trial is None:
            failed += 1
            unmeasured += 1
            scores.append(None)
            continue
        trials.append(trial)
        score = getattr(trial, "score", None)
        if score is None:
            score = (trial.metrics or {}).get(primary_metric)
        if not _is_number(score) or not trial.is_successful:
            failed += 1
            scores.append(None)
        else:
            scores.append(float(cast(float, score)))
        for name, value in (trial.metrics or {}).items():
            if _is_number(value):
                metric_values.setdefault(name, []).append(float(value))
        stats = result.experiment_stats
        if stats.total_cost is None or stats.cost_unmeasured_trials:
            unmeasured += 1
            floor += float(stats.cost_lower_bound or 0.0)
        else:
            known_cost += float(stats.total_cost)
    valid = [s for s in scores if s is not None]
    cost_unknown = unmeasured > 0
    return ConfigComparison(
        config=dict(config),
        repeats=len(run_results),
        scores=scores,
        mean_score=_mean(valid),
        score_stdev=_spread(valid),
        score_min=min(valid) if valid else None,
        score_max=max(valid) if valid else None,
        metrics_mean={
            name: statistics.fmean(vals)
            for name, vals in sorted(metric_values.items())
            if vals
        },
        total_cost=None if cost_unknown else known_cost,
        cost_lower_bound=(known_cost + floor) if cost_unknown else None,
        cost_unmeasured_repeats=unmeasured,
        failed_repeats=failed,
        example_ids=example_ids,
        trials=trials,
    )


@dataclass
class ComparisonResult:
    """ONE experiment: every fixed config evaluated on the same examples."""

    comparison_id: str
    primary_metric: str
    #: Declared orientation of the primary objective: maximize/minimize/band.
    orientation: str
    #: ``(low, high)`` target band when ``orientation == "band"``.
    band: tuple[Any, Any] | None
    objectives: list[str]
    repeats: int
    #: Examples each evaluation was asked to run (after any ``max_examples`` cap).
    n_examples: int
    #: Examples in the dataset before the cap.
    n_source_examples: int
    #: The cap that truncated the dataset, or ``None`` when none applied.
    max_examples_applied: int | None
    #: Example ids the evaluator actually returned (first evaluation).
    example_ids: tuple[str, ...]
    #: True when EVERY evaluation returned the identical, non-empty id list.
    same_examples_across_configs: bool
    #: True when every evaluation returned a row for every requested example.
    coverage_complete: bool
    holdout: HoldoutCheck
    configs: list[ConfigComparison]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: Why the comparison stopped early (``cost_limit``, ``timeout``,
    #: ``max_samples_reached``, ``execution_budget``, ...); ``None`` = finished.
    stop_reason: str | None = None
    n_configs_requested: int = 0
    n_runs_completed: int = 0
    n_runs_requested: int = 0
    #: Mock-LLM provenance, as for optimization runs (#2516).
    mock_llm_run: bool = False
    mock_llm_mixed: bool = False
    strict_cost_accounting: bool = False
    strict_cost_accounting_origin: str | None = None

    @property
    def completed(self) -> bool:
        """True when every requested (config, repeat) was evaluated."""
        return self.stop_reason is None and (
            self.n_runs_completed == self.n_runs_requested
        )

    def _rank_key(self, mean: float) -> float:
        """Higher is better, per the objective's declared orientation."""
        if self.orientation == "minimize":
            return -mean
        if self.orientation == "band":
            low, high = self.band if self.band is not None else (None, None)
            if low is None or high is None:
                raise ValueError("A banded objective requires a complete band.")
            return float(-max(low - mean, mean - high, 0.0))
        return mean

    @property
    def best(self) -> ConfigComparison | None:
        """Best config by mean score under the declared orientation.

        ``None`` when no config scored. Ties go to the earlier config.
        """
        scored = [c for c in self.configs if c.mean_score is not None]
        if not scored:
            return None
        return max(scored, key=lambda c: self._rank_key(c.mean_score or 0.0))

    @property
    def best_config(self) -> dict[str, Any] | None:
        best = self.best
        return dict(best.config) if best is not None else None

    @property
    def total_cost(self) -> float | None:
        """Spend over the whole comparison; ``None`` if any config is unmeasured."""
        if any(c.total_cost is None for c in self.configs):
            return None
        return sum(c.total_cost or 0.0 for c in self.configs)

    @property
    def cost_lower_bound(self) -> float | None:
        if self.total_cost is not None:
            return None
        return sum(
            c.total_cost if c.total_cost is not None else (c.cost_lower_bound or 0.0)
            for c in self.configs
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "primary_metric": self.primary_metric,
            "orientation": self.orientation,
            "band": list(self.band) if self.band is not None else None,
            "objectives": list(self.objectives),
            "repeats": self.repeats,
            "n_examples": self.n_examples,
            "n_source_examples": self.n_source_examples,
            "max_examples_applied": self.max_examples_applied,
            "example_ids": list(self.example_ids),
            "same_examples_across_configs": self.same_examples_across_configs,
            "coverage_complete": self.coverage_complete,
            "holdout": self.holdout.to_dict(),
            "configs": [c.to_dict() for c in self.configs],
            "total_cost": self.total_cost,
            "cost_lower_bound": self.cost_lower_bound,
            "timestamp": self.timestamp.isoformat(),
            "stop_reason": self.stop_reason,
            "completed": self.completed,
            "n_configs_requested": self.n_configs_requested,
            "n_runs_completed": self.n_runs_completed,
            "n_runs_requested": self.n_runs_requested,
            "mock_llm_run": self.mock_llm_run,
            "mock_llm_mixed": self.mock_llm_mixed,
            "strict_cost_accounting": self.strict_cost_accounting,
            "strict_cost_accounting_origin": self.strict_cost_accounting_origin,
        }


def new_comparison_id() -> str:
    return f"compare_{uuid.uuid4().hex[:12]}"
