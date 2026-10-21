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
from typing import Any

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


def _value_in_domain(spec: Any, value: Any) -> bool:
    """True when ``value`` lies inside one normalized configuration-space entry."""
    if isinstance(spec, list):
        return any(
            value == option and isinstance(value, bool) == isinstance(option, bool)
            for option in spec
        )
    if isinstance(spec, tuple) and len(spec) == 2 and all(_is_number(b) for b in spec):
        return _is_number(value) and spec[0] <= value <= spec[1]
    if isinstance(spec, Mapping) and "low" in spec and "high" in spec:
        low, high = spec["low"], spec["high"]
        return (
            _is_number(value)
            and _is_number(low)
            and _is_number(high)
            and low <= value <= high
        )
    return bool(value == spec)


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
            scores.append(float(score))
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
    objectives: list[str]
    repeats: int
    n_examples: int
    example_ids: tuple[str, ...]
    #: True when every config's evaluated example ids equal the dataset's.
    same_examples_across_configs: bool
    holdout: HoldoutCheck
    configs: list[ConfigComparison]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def best(self) -> ConfigComparison | None:
        """Config with the highest mean score (``None`` if none scored)."""
        scored = [c for c in self.configs if c.mean_score is not None]
        return max(scored, key=lambda c: c.mean_score or 0.0) if scored else None

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
            "objectives": list(self.objectives),
            "repeats": self.repeats,
            "n_examples": self.n_examples,
            "example_ids": list(self.example_ids),
            "same_examples_across_configs": self.same_examples_across_configs,
            "holdout": self.holdout.to_dict(),
            "configs": [c.to_dict() for c in self.configs],
            "total_cost": self.total_cost,
            "cost_lower_bound": self.cost_lower_bound,
            "timestamp": self.timestamp.isoformat(),
        }


def new_comparison_id() -> str:
    return f"compare_{uuid.uuid4().hex[:12]}"
