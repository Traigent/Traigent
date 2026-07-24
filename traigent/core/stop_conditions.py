"""Reusable stop conditions for optimization orchestration."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import copy
import json
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from traigent.api.types import TrialResult, TrialStatus
from traigent.config.feature_flags import _coerce_bool as _coerce_config_bool_value
from traigent.core.objectives import ObjectiveSchema
from traigent.core.utils import extract_examples_attempted
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.core.cost_enforcement import CostEnforcer
    from traigent.core.execution_budget import ExecutionBudget

logger = get_logger(__name__)


_AUTO_SELECTOR = "auto"
_SEMANTIC_SATURATION_ALLOWED_KEYS = frozenset(
    {
        "enabled",
        "window",
        "min_trials",
        "score_tolerance",
        "per_example_tol",
        "churn_threshold",
        "min_overlap",
        "relative_improvement_epsilon",
        "absolute_improvement_epsilon",
        "objectives",
        "example_score_metrics",
        "accuracy_metrics",
        "continuous_objectives",
        "include_example_ids",
        "max_example_ids",
        "max_ids_in_diagnostic",
    }
)
_QUALITY_NAME_MARKERS = (
    "accuracy",
    "score",
    "success",
    "correct",
    "pass",
    "passed",
    "match",
    "quality",
    "f1",
    "precision",
    "recall",
)
_CONTINUOUS_NAME_MARKERS = (
    "cost",
    "latency",
    "token",
    "duration",
    "time",
    "ms",
    "seconds",
    "error",
    "loss",
    "price",
    "memory",
)
_MINIMIZE_NAME_MARKERS = (
    "cost",
    "latency",
    "token",
    "duration",
    "time",
    "ms",
    "seconds",
    "error",
    "loss",
    "price",
    "memory",
)
_SEMANTIC_DEFAULT_WINDOW = 4
_SEMANTIC_DEFAULT_MIN_TRIALS = 4
_SEMANTIC_DEFAULT_SCORE_TOLERANCE = 1e-9
_SEMANTIC_DEFAULT_CHURN_THRESHOLD = 0.0
_SEMANTIC_DEFAULT_MIN_OVERLAP = 0.8
_SEMANTIC_DEFAULT_RELATIVE_IMPROVEMENT_EPSILON = 0.01
_SEMANTIC_DEFAULT_ABSOLUTE_IMPROVEMENT_EPSILON = 0.0
_SEMANTIC_DEFAULT_MAX_EXAMPLE_IDS = 50


@dataclass(frozen=True, slots=True)
class _SemanticSaturationConfig:
    enabled: bool = True
    window: int = _SEMANTIC_DEFAULT_WINDOW
    min_trials: int = _SEMANTIC_DEFAULT_MIN_TRIALS
    score_tolerance: float = _SEMANTIC_DEFAULT_SCORE_TOLERANCE
    churn_threshold: float = _SEMANTIC_DEFAULT_CHURN_THRESHOLD
    min_overlap: float = _SEMANTIC_DEFAULT_MIN_OVERLAP
    relative_improvement_epsilon: float = _SEMANTIC_DEFAULT_RELATIVE_IMPROVEMENT_EPSILON
    absolute_improvement_epsilon: float = _SEMANTIC_DEFAULT_ABSOLUTE_IMPROVEMENT_EPSILON
    example_score_metrics: tuple[str, ...] | None = None
    continuous_objectives: tuple[str, ...] | None = None
    include_example_ids: bool = True
    max_example_ids: int = _SEMANTIC_DEFAULT_MAX_EXAMPLE_IDS

    @classmethod
    def from_raw(
        cls, raw: bool | Mapping[str, Any] | None
    ) -> _SemanticSaturationConfig:
        if raw is None:
            raw = {}
        if isinstance(raw, bool):
            return cls(enabled=raw)
        if not isinstance(raw, Mapping):
            raise ValueError("semantic_saturation must be a boolean or config object")

        unknown = set(raw) - _SEMANTIC_SATURATION_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"Unknown semantic_saturation option(s): {sorted(unknown)}"
            )

        score_tolerance = raw.get(
            "score_tolerance",
            raw.get("per_example_tol", _SEMANTIC_DEFAULT_SCORE_TOLERANCE),
        )
        max_example_ids = raw.get(
            "max_example_ids",
            raw.get("max_ids_in_diagnostic", _SEMANTIC_DEFAULT_MAX_EXAMPLE_IDS),
        )
        example_score_metrics = raw.get(
            "example_score_metrics", raw.get("accuracy_metrics")
        )
        continuous_objectives = raw.get("continuous_objectives", raw.get("objectives"))

        return cls(
            enabled=_coerce_config_bool(raw.get("enabled", True), "enabled"),
            window=_coerce_positive_int(
                raw.get("window", _SEMANTIC_DEFAULT_WINDOW), "window"
            ),
            min_trials=_coerce_positive_int(
                raw.get("min_trials", _SEMANTIC_DEFAULT_MIN_TRIALS), "min_trials"
            ),
            score_tolerance=_coerce_non_negative_float(
                score_tolerance, "score_tolerance"
            ),
            churn_threshold=_coerce_bounded_float(
                raw.get("churn_threshold", _SEMANTIC_DEFAULT_CHURN_THRESHOLD),
                "churn_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            min_overlap=_coerce_bounded_float(
                raw.get("min_overlap", _SEMANTIC_DEFAULT_MIN_OVERLAP),
                "min_overlap",
                minimum=0.0,
                maximum=1.0,
            ),
            relative_improvement_epsilon=_coerce_non_negative_float(
                raw.get(
                    "relative_improvement_epsilon",
                    _SEMANTIC_DEFAULT_RELATIVE_IMPROVEMENT_EPSILON,
                ),
                "relative_improvement_epsilon",
            ),
            absolute_improvement_epsilon=_coerce_non_negative_float(
                raw.get(
                    "absolute_improvement_epsilon",
                    _SEMANTIC_DEFAULT_ABSOLUTE_IMPROVEMENT_EPSILON,
                ),
                "absolute_improvement_epsilon",
            ),
            example_score_metrics=_normalize_metric_selector(
                example_score_metrics,
                "example_score_metrics",
            ),
            continuous_objectives=_normalize_metric_selector(
                continuous_objectives,
                "continuous_objectives",
            ),
            include_example_ids=_coerce_config_bool(
                raw.get("include_example_ids", True), "include_example_ids"
            ),
            max_example_ids=_coerce_positive_int(
                max_example_ids,
                "max_example_ids",
            ),
        )


def _coerce_config_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        coerced = _coerce_config_bool_value(value)
        if coerced is not None:
            return coerced
    if isinstance(value, str):
        coerced = _coerce_config_bool_value(value)
        if coerced is not None:
            return coerced
    raise ValueError(f"{name} must be a boolean")


def _coerce_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _coerce_non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return number


def _coerce_bounded_float(
    value: Any, name: str, *, minimum: float, maximum: float
) -> float:
    number = _coerce_non_negative_float(value, name)
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _normalize_metric_selector(value: Any, name: str) -> tuple[str, ...] | None:
    if value is None or value == _AUTO_SELECTOR:
        return None
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} entries must be non-empty strings")
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValueError(f"{name} entries must be non-empty strings")
            if item not in normalized:
                normalized.append(item)
        return tuple(normalized)
    raise ValueError(f"{name} must be 'auto', a string, or a list of strings")


def _coerce_numeric_or_bool(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _looks_like_quality_metric(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _QUALITY_NAME_MARKERS)


def _looks_like_continuous_metric(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _CONTINUOUS_NAME_MARKERS)


def _infer_direction_from_name(name: str) -> str:
    lowered = name.lower()
    if any(marker in lowered for marker in _MINIMIZE_NAME_MARKERS):
        return "minimize"
    return "maximize"


class StopCondition(ABC):
    """Interface for reusable stop criteria."""

    reason: str = "condition"

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal tracking state."""

    @abstractmethod
    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        """Return ``True`` when the condition dictates stopping."""


class MaxTrialsStopCondition(StopCondition):
    """Stop once a maximum number of executed trials has been reached.

    Counts all trials except those explicitly marked as abandoned in metadata.
    This keeps cache/cap-abandoned trials from reducing the execution budget,
    while still counting pruned trials produced during execution.
    """

    reason = "max_trials"

    _ABANDONED_METADATA_KEY = "abandoned"

    def __init__(self, max_trials: int | None) -> None:
        if max_trials is None:
            self._max_trials = None
            return

        if not isinstance(max_trials, int) or max_trials <= 0:
            raise ValueError("max_trials must be a positive integer")

        self._max_trials = max_trials

    def reset(self) -> None:  # noqa: D401 - interface requirement
        return

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if self._max_trials is None:
            return False

        def is_abandoned(trial: TrialResult) -> bool:
            metadata = getattr(trial, "metadata", None) or {}
            return bool(metadata.get(self._ABANDONED_METADATA_KEY, False))

        count = sum(1 for trial in trials if not is_abandoned(trial))
        return count >= self._max_trials


class PlateauAfterNStopCondition(StopCondition):
    """Stop when the best weighted score plateaus for ``window_size`` trials."""

    reason = "plateau"

    def __init__(
        self,
        *,
        window_size: int,
        epsilon: float,
        objective_schema: ObjectiveSchema | None,
    ) -> None:
        if objective_schema is None:
            raise ValueError("objective_schema is required for plateau detection")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        self._window_size = window_size
        self._epsilon = float(epsilon)
        self._schema = objective_schema
        self._history: deque[float] = deque(maxlen=self._window_size)
        self._best_score: float | None = None
        self._last_index = 0

    def reset(self) -> None:
        self._history.clear()
        self._best_score = None
        self._last_index = 0

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        new_trials = trial_seq[self._last_index :]
        if not new_trials:
            return False

        for trial in new_trials:
            score = self._schema.compute_weighted_score(trial.metrics)
            if score is None:
                continue

            if self._best_score is None or score > self._best_score + self._epsilon:
                self._best_score = score

            if self._best_score is not None:
                self._history.append(self._best_score)

        self._last_index = len(trial_seq)

        if len(self._history) < self._window_size:
            return False

        delta = self._history[-1] - self._history[0]
        return abs(delta) <= self._epsilon


class MetricLimitStopCondition(StopCondition):
    """Stop when the cumulative value of a named metric reaches a limit.

    This is a soft, post-trial stop condition. It sums completed trial metrics and
    stops before the next trial once the configured metric has reached the limit.
    For hard money-spend control, use ``cost_limit`` and ``CostLimitStopCondition``.
    """

    reason = "metric_limit"

    def __init__(
        self,
        *,
        limit: float,
        metric_name: str,
        include_pruned: bool = True,
    ) -> None:
        if limit is None or float(limit) <= 0:
            raise ValueError("metric limit must be a positive number")
        if not metric_name:
            raise ValueError("metric_name must be provided")

        self._limit = float(limit)
        self._metric = metric_name
        self._include_pruned = include_pruned
        self._running_total = 0.0
        self._last_index = 0

    def reset(self) -> None:  # noqa: D401 - interface requirement
        self._running_total = 0.0
        self._last_index = 0

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        new_trials = trial_seq[self._last_index :]
        if not new_trials:
            return self._running_total >= self._limit

        for trial in new_trials:
            if not self._include_pruned and trial.status == TrialStatus.PRUNED:
                continue

            metrics = trial.metrics or {}
            value = metrics.get(self._metric)

            if value is None and self._metric == "total_cost":
                # Compatibility with older evaluators that reported cost under
                # different names before metric_limit required an explicit metric.
                value = metrics.get("cost")
                if value is None:
                    value = (trial.metadata or {}).get("total_example_cost")

            if value is None:
                raise ValueError(
                    f"Mandatory metric '{self._metric}' missing for trial {trial.trial_id}"
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Metric '{self._metric}' for trial {trial.trial_id} is not numeric: {value!r}"
                ) from exc

            self._running_total += numeric_value

            if self._running_total >= self._limit:
                self._last_index = len(trial_seq)
                return True

        self._last_index = len(trial_seq)
        return self._running_total >= self._limit


class MaxSamplesStopCondition(StopCondition):
    """Stop once cumulative ``examples_attempted`` reaches a threshold."""

    reason = "max_samples"

    def __init__(
        self,
        *,
        max_samples: int | None,
        include_pruned: bool = True,
    ) -> None:
        if max_samples is None:
            self._max_samples = None
        else:
            if not isinstance(max_samples, int) or max_samples <= 0:
                raise ValueError("max_samples must be a positive integer or None")
            self._max_samples = max_samples
        self._include_pruned = include_pruned
        self._total_attempted = 0
        self._last_index = 0

    def reset(self) -> None:  # noqa: D401
        self._total_attempted = 0
        self._last_index = 0

    def update_limit(self, value: int | None) -> None:
        if value is None:
            self._max_samples = None
        else:
            if not isinstance(value, int) or value <= 0:
                raise ValueError("max_samples must be a positive integer or None")
            self._max_samples = value
        self._total_attempted = 0
        self._last_index = 0

    def set_include_pruned(self, include_pruned: bool) -> None:
        self._include_pruned = include_pruned
        self._total_attempted = 0
        self._last_index = 0

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if self._max_samples is None:
            return False

        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        new_trials = trial_seq[self._last_index :]
        if not new_trials:
            return self._total_attempted >= self._max_samples

        for trial in new_trials:
            if not self._include_pruned and trial.status == TrialStatus.PRUNED:
                continue

            attempted = extract_examples_attempted(
                trial, default=None, check_example_results=True
            )
            if attempted is not None:
                self._total_attempted += attempted

            if self._total_attempted >= self._max_samples:
                self._last_index = len(trial_seq)
                return True

        self._last_index = len(trial_seq)
        return self._total_attempted >= self._max_samples


class SemanticSaturationStopCondition(StopCondition):
    """Stop when per-example quality vectors and continuous objectives saturate.

    The privacy boundary is deliberately narrow: this condition reads only
    ``example_id`` and numeric or boolean values from per-example ``metrics``.
    It never reads raw inputs, expected outputs, actual outputs, model text, or
    document content, and diagnostics emit IDs only under the configured cap.
    """

    reason = "semantic_saturation"

    def __init__(
        self,
        semantic_saturation: bool | Mapping[str, Any] | None = None,
        *,
        objective_schema: ObjectiveSchema | None = None,
    ) -> None:
        self._config = _SemanticSaturationConfig.from_raw(semantic_saturation)
        self._schema = objective_schema
        self._last_diagnostics: dict[str, Any] | None = None

    def reset(self) -> None:
        self._last_diagnostics = None

    @property
    def diagnostics(self) -> dict[str, Any] | None:
        if self._last_diagnostics is None:
            return None
        return copy.deepcopy(self._last_diagnostics)

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        diagnostics = self._evaluate(trial_seq)
        self._last_diagnostics = diagnostics
        return bool(diagnostics["decision"] == "stop")

    def _evaluate(self, trials: Sequence[TrialResult]) -> dict[str, Any]:
        cfg = self._config
        distinct_trials = self._latest_distinct_completed_trials(trials)
        base = self._base_diagnostics(distinct_trials)

        if not cfg.enabled:
            return {**base, "decision": "continue", "reason_detail": "disabled"}

        required_trials = max(cfg.window, cfg.min_trials)
        if len(distinct_trials) < required_trials:
            return {**base, "decision": "continue", "reason_detail": "warmup"}

        window_trials = distinct_trials[-cfg.window :]
        base = self._base_diagnostics(window_trials)
        quality_metrics = self._resolve_quality_metrics(window_trials)
        objectives: dict[str, dict[str, Any]] = {}

        if not quality_metrics:
            return {
                **base,
                "decision": "continue",
                "reason_detail": "insufficient_example_scores",
            }

        for metric in quality_metrics:
            objectives[metric] = self._evaluate_quality_metric(metric, window_trials)

        quality_saturated = all(diag["saturated"] for diag in objectives.values())
        if not quality_saturated:
            reason_detail = (
                "accuracy_churning"
                if any(
                    diag["reason_detail"] == "accuracy_churning"
                    for diag in objectives.values()
                )
                else "insufficient_example_scores"
            )
            return {
                **base,
                "decision": "continue",
                "reason_detail": reason_detail,
                "objectives": objectives,
            }

        continuous_objectives = self._resolve_continuous_objectives(
            window_trials,
            quality_metrics,
        )
        for metric in continuous_objectives:
            objectives[metric] = self._evaluate_continuous_objective(
                metric,
                window_trials,
            )

        all_saturated = all(diag["saturated"] for diag in objectives.values())
        if all_saturated:
            return {
                **base,
                "decision": "stop",
                "reason_detail": "all_objectives_saturated",
                "objectives": objectives,
            }

        return {
            **base,
            "decision": "continue",
            "reason_detail": "quality_saturated_efficiency_improving",
            "objectives": objectives,
        }

    def _base_diagnostics(self, trials: Sequence[TrialResult]) -> dict[str, Any]:
        return {
            "condition": "semantic_saturation",
            "decision": "continue",
            "reason_detail": "warmup",
            "window": self._config.window,
            "min_trials": self._config.min_trials,
            "trials_considered": [str(trial.trial_id) for trial in trials],
            "objectives": {},
        }

    def _latest_distinct_completed_trials(
        self, trials: Sequence[TrialResult]
    ) -> list[TrialResult]:
        required_trials = max(self._config.window, self._config.min_trials)
        seen_signatures: set[str] = set()
        selected_reversed: list[TrialResult] = []

        for trial in reversed(trials):
            if trial.status != TrialStatus.COMPLETED:
                continue
            signature = self._config_signature(trial.config)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            selected_reversed.append(trial)
            if len(selected_reversed) >= required_trials:
                break

        return list(reversed(selected_reversed))

    @staticmethod
    def _config_signature(config: Mapping[str, Any] | None) -> str:
        try:
            return json.dumps(
                config or {},
                sort_keys=True,
                separators=(",", ":"),
                default=repr,
            )
        except (TypeError, ValueError):
            return repr(config)

    def _resolve_quality_metrics(
        self, trials: Sequence[TrialResult]
    ) -> tuple[str, ...]:
        if self._config.example_score_metrics is not None:
            return self._config.example_score_metrics

        values_by_metric: dict[str, list[float]] = {}
        for trial in trials:
            for result in self._example_results_for_trial(trial):
                metrics = self._read_example_metrics(result)
                if not isinstance(metrics, Mapping):
                    continue
                for metric_name, raw_value in metrics.items():
                    if not isinstance(metric_name, str):
                        continue
                    value = _coerce_numeric_or_bool(raw_value)
                    if value is None:
                        continue
                    values_by_metric.setdefault(metric_name, []).append(value)

        selected: list[str] = []
        for metric_name in sorted(values_by_metric):
            values = values_by_metric[metric_name]
            if _looks_like_continuous_metric(metric_name):
                continue
            if _looks_like_quality_metric(metric_name) or self._all_binary(values):
                selected.append(metric_name)
        return tuple(selected)

    @staticmethod
    def _all_binary(values: Sequence[float]) -> bool:
        return bool(values) and all(value in (0.0, 1.0) for value in values)

    def _resolve_continuous_objectives(
        self,
        trials: Sequence[TrialResult],
        quality_metrics: Sequence[str],
    ) -> tuple[str, ...]:
        quality_metric_set = set(quality_metrics)
        if self._config.continuous_objectives is not None:
            return tuple(
                metric
                for metric in self._config.continuous_objectives
                if metric not in quality_metric_set
            )

        objective_names: list[str] = []
        if self._schema is not None:
            for objective in self._schema.objectives:
                name = objective.name
                if name in quality_metric_set:
                    continue
                if _looks_like_quality_metric(
                    name
                ) and not _looks_like_continuous_metric(name):
                    continue
                objective_names.append(name)
            if objective_names:
                return tuple(objective_names)

        aggregate_metric_names: set[str] = set()
        for trial in trials:
            for metric_name, raw_value in (trial.metrics or {}).items():
                if (
                    not isinstance(metric_name, str)
                    or metric_name in quality_metric_set
                ):
                    continue
                if _coerce_numeric_or_bool(raw_value) is None:
                    continue
                if _looks_like_continuous_metric(metric_name):
                    aggregate_metric_names.add(metric_name)
        return tuple(sorted(aggregate_metric_names))

    def _evaluate_quality_metric(
        self,
        metric: str,
        trials: Sequence[TrialResult],
    ) -> dict[str, Any]:
        vectors = [self._example_score_vector(trial, metric) for trial in trials]
        if any(not vector for vector in vectors):
            return {
                "regime": "example_vector",
                "saturated": False,
                "reason_detail": "insufficient_example_scores",
                "examples_compared": 0,
            }

        common_ids = set(vectors[0])
        union_ids = set(vectors[0])
        for vector in vectors[1:]:
            common_ids.intersection_update(vector)
            union_ids.update(vector)

        if not union_ids:
            overlap_ratio = 0.0
        else:
            overlap_ratio = len(common_ids) / len(union_ids)

        if not common_ids or overlap_ratio < self._config.min_overlap:
            return {
                "regime": "example_vector",
                "saturated": False,
                "reason_detail": "insufficient_overlap",
                "overlap_ratio": overlap_ratio,
                "examples_compared": len(common_ids),
            }

        changed_ids: list[str] = []
        stable_ids: list[str] = []
        for example_id in sorted(common_ids):
            values = [vector[example_id] for vector in vectors]
            if max(values) - min(values) > self._config.score_tolerance:
                changed_ids.append(example_id)
            else:
                stable_ids.append(example_id)

        changed_count = len(changed_ids)
        stable_count = len(stable_ids)
        churn = changed_count / len(common_ids)
        saturated = churn <= self._config.churn_threshold
        diagnostic: dict[str, Any] = {
            "regime": "example_vector",
            "saturated": saturated,
            "reason_detail": "quality_saturated" if saturated else "accuracy_churning",
            "max_churn": churn,
            "churn_threshold": self._config.churn_threshold,
            "overlap_ratio": overlap_ratio,
            "examples_compared": len(common_ids),
            "stable_example_count": stable_count,
            "changed_example_count": changed_count,
        }
        if self._config.include_example_ids:
            self._add_capped_ids(diagnostic, "changed", changed_ids)
            self._add_capped_ids(diagnostic, "stable", stable_ids)
        return diagnostic

    def _example_score_vector(
        self,
        trial: TrialResult,
        metric: str,
    ) -> dict[str, float]:
        vector: dict[str, float] = {}
        for result in self._example_results_for_trial(trial):
            example_id = self._read_example_id(result)
            if example_id is None:
                continue
            metrics = self._read_example_metrics(result)
            if not isinstance(metrics, Mapping):
                continue
            value = _coerce_numeric_or_bool(metrics.get(metric))
            if value is not None:
                vector[str(example_id)] = value
        return vector

    @staticmethod
    def _example_results_for_trial(trial: TrialResult) -> list[Any]:
        metadata = getattr(trial, "metadata", None)
        if isinstance(metadata, Mapping):
            results = metadata.get("example_results")
            if isinstance(results, list):
                return results
        results = getattr(trial, "example_results", None)
        if isinstance(results, list):
            return results
        return []

    @staticmethod
    def _read_example_id(result: Any) -> Any | None:
        if isinstance(result, Mapping):
            return result.get("example_id")
        try:
            return getattr(result, "example_id", None)
        except Exception:
            return None

    @staticmethod
    def _read_example_metrics(result: Any) -> Any | None:
        if isinstance(result, Mapping):
            return result.get("metrics")
        try:
            return getattr(result, "metrics", None)
        except Exception:
            return None

    def _add_capped_ids(
        self,
        diagnostic: dict[str, Any],
        prefix: str,
        example_ids: Sequence[str],
    ) -> None:
        cap = self._config.max_example_ids
        diagnostic[f"{prefix}_example_ids"] = list(example_ids[:cap])
        diagnostic[f"{prefix}_example_ids_truncated_count"] = max(
            len(example_ids) - cap,
            0,
        )

    def _evaluate_continuous_objective(
        self,
        metric: str,
        trials: Sequence[TrialResult],
    ) -> dict[str, Any]:
        direction = self._direction_for_metric(metric)
        if direction == "band":
            return {
                "regime": "continuous",
                "saturated": False,
                "reason_detail": "unsupported_band_objective",
                "direction": "band",
            }

        values: list[float] = []
        for trial in trials:
            value = _coerce_numeric_or_bool((trial.metrics or {}).get(metric))
            if value is None:
                return {
                    "regime": "continuous",
                    "saturated": False,
                    "reason_detail": "insufficient_continuous_metric",
                    "direction": direction,
                }
            values.append(value)

        if not values:
            return {
                "regime": "continuous",
                "saturated": False,
                "reason_detail": "insufficient_continuous_metric",
                "direction": direction,
            }

        initial_best = values[0]
        if direction == "minimize":
            best = min(values)
            improvement = max(0.0, initial_best - best)
        else:
            best = max(values)
            improvement = max(0.0, best - initial_best)

        threshold = max(
            self._config.absolute_improvement_epsilon,
            abs(initial_best) * self._config.relative_improvement_epsilon,
        )
        saturated = improvement <= threshold
        return {
            "regime": "continuous",
            "saturated": saturated,
            "reason_detail": (
                "continuous_saturated"
                if saturated
                else "continuous_objective_improving"
            ),
            "direction": direction,
            "best": best,
            "improvement": improvement,
            "improvement_threshold": threshold,
        }

    def _direction_for_metric(self, metric: str) -> str:
        if self._schema is not None:
            for objective in self._schema.objectives:
                if objective.name == metric:
                    return str(objective.orientation)
        return _infer_direction_from_name(metric)


class CostLimitStopCondition(StopCondition):
    """Stop when cost limit reached using shared CostEnforcer.

    This stop condition uses a shared CostEnforcer instance to avoid double
    counting costs. The enforcer tracks actual costs; this condition just
    checks the enforcer's state.

    Note:
        The CostEnforcer must be passed in and is expected to be tracking
        costs already (via track_cost() calls from the orchestrator).
    """

    reason = "cost_limit"

    def __init__(self, cost_enforcer: CostEnforcer) -> None:
        """Initialize with shared cost enforcer.

        Args:
            cost_enforcer: Shared CostEnforcer instance that tracks costs.
        """
        self._cost_enforcer = cost_enforcer

    def reset(self) -> None:
        """Reset is handled by the shared CostEnforcer, not here."""
        # Note: We don't reset the enforcer here because it's shared
        # and may be used across multiple stop condition checks
        pass

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        """Check if cost limit has been reached.

        Args:
            trials: Trial results (not used - we check the enforcer directly).

        Returns:
            True if the cost limit has been reached.
        """
        # Cost is already tracked by the orchestrator - just check status
        return self._cost_enforcer.is_limit_reached

    def get_reason(self) -> str:
        """Get a descriptive reason for stopping.

        Returns:
            Human-readable description of why optimization stopped.
        """
        status = self._cost_enforcer.get_status()
        if status.unknown_cost_mode:
            return (
                f"Trial limit reached: {status.trial_count} trials "
                f"(cost unknown, fallback mode)"
            )
        return (
            f"Cost limit reached: ${status.accumulated_cost_usd:.2f} "
            f">= ${status.limit_usd:.2f} USD"
        )


class ExecutionBudgetStopCondition(StopCondition):
    """Stop when a shared cumulative ExecutionBudget is exhausted (issue #1980).

    Modeled on :class:`CostLimitStopCondition`: it holds a reference to the shared
    :class:`~traigent.core.execution_budget.ExecutionBudget` (attached per run) and
    checks its live state rather than re-deriving anything from ``trials``.

    Registration is FRONT of the condition list (``insert(0, ...)``) via
    ``StopConditionManager.register_execution_budget_condition``. This ordering is
    mandatory: the user's own per-run ``CostLimitStopCondition`` can fire on the
    same iteration as the cumulative budget (e.g. a run whose ``cost_limit`` and
    shared ``max_cost_usd`` are both spent). ``should_stop`` is first-match-wins, so
    the ExecutionBudget condition must be evaluated first — otherwise a stop that
    was really the shared cumulative cap would be mislabeled ``"cost_limit"``,
    masking the true ``"execution_budget"`` reason. (The per-run cost enforcer's
    limit is NOT clamped to the budget's remaining — that clamp was removed with
    issue #1980's F1 fix; the cumulative cost is enforced by this condition plus the
    orchestrator's pre-batch admission gate, both reading the budget directly.)
    """

    reason = "execution_budget"

    def __init__(self, budget: ExecutionBudget) -> None:
        """Initialize with the shared execution budget.

        Args:
            budget: Shared ExecutionBudget instance debited by the orchestrator.
        """
        self._budget = budget

    def reset(self) -> None:
        """Reset is a no-op — the budget is shared/external, not owned here."""
        # Mirror CostLimitStopCondition: never reset a shared, cross-run object.

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        """Return True when any attached-budget dimension is exhausted.

        Args:
            trials: Trial results (unused — the budget is authoritative).
        """
        return self._budget.exhausted_dimension is not None

    def get_reason(self) -> str:
        """Get a descriptive reason naming the exhausted dimension."""
        snapshot = self._budget.snapshot()
        dimension = snapshot.exhausted_dimension or "unknown"
        if dimension == "cost":
            return (
                f"Execution budget cost exhausted: "
                f"${snapshot.consumed_cost:.2f} of ${snapshot.max_cost_usd:.2f} USD"
            )
        if dimension == "examples":
            return (
                "Execution budget examples exhausted: "
                f"{snapshot.consumed_examples} of {snapshot.max_examples}"
            )
        if dimension == "deadline":
            return (
                "Execution budget deadline reached: "
                f"{snapshot.elapsed_seconds:.1f}s of {snapshot.deadline_seconds:.1f}s"
            )
        if dimension == "untracked_cost":
            return (
                "Execution budget stopped: cost became unobservable and "
                "enforce_untracked_cost is set (fail-closed)"
            )
        return "Execution budget exhausted"


class HypervolumeConvergenceStopCondition(StopCondition):
    """Stop when hypervolume improvement falls below threshold.

    This stop condition implements convergence detection based on hypervolume
    improvement as specified in TVL 0.9. It monitors the hypervolume indicator
    over a sliding window and triggers early stopping when improvement stagnates.

    The hypervolume indicator measures the volume of objective space dominated
    by the Pareto front. When this improvement falls below a threshold over
    multiple consecutive trials, optimization is considered converged.

    Example:
        ```python
        from traigent.core.stop_conditions import HypervolumeConvergenceStopCondition

        condition = HypervolumeConvergenceStopCondition(
            window=10,
            threshold=0.001,
            objective_names=["accuracy", "latency"],
            directions=["maximize", "minimize"],
            reference_point=[0.0, 1000.0],  # Worst case for each objective
        )
        ```

    Note:
        This is typically configured via TVL spec's exploration.convergence section:
        ```yaml
        exploration:
          convergence:
            metric: hypervolume_improvement
            window: 20
            threshold: 0.001
        ```
    """

    reason = "convergence"

    def __init__(
        self,
        *,
        window: int,
        threshold: float,
        objective_names: list[str],
        directions: list[str],
        reference_point: list[float] | None = None,
    ) -> None:
        """Initialize hypervolume convergence stop condition.

        Args:
            window: Number of trials to consider for convergence detection.
            threshold: Minimum hypervolume improvement required to continue.
                If improvement falls below this for `window` consecutive trials,
                optimization stops.
            objective_names: Names of the objectives being optimized.
            directions: Optimization direction for each objective ("maximize" or "minimize").
            reference_point: Reference point for hypervolume calculation. If None,
                uses worst observed values with a margin.

        Raises:
            ValueError: If arguments are invalid.
        """
        if window <= 0:
            raise ValueError("window must be a positive integer")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if len(objective_names) != len(directions):
            raise ValueError("objective_names and directions must have same length")
        if not objective_names:
            raise ValueError("At least one objective is required")

        self._window = window
        self._threshold = threshold
        self._objective_names = objective_names
        self._directions = directions
        self._reference_point = reference_point

        # Internal state
        self._hypervolume_history: deque[float] = deque(maxlen=window)
        self._pareto_front: list[list[float]] = []
        self._last_index = 0
        self._computed_reference: list[float] | None = None

    def reset(self) -> None:
        """Reset convergence tracking state."""
        self._hypervolume_history.clear()
        self._pareto_front = []
        self._last_index = 0
        self._computed_reference = None

    def should_stop(self, trials: Iterable[TrialResult]) -> bool:
        """Check if hypervolume has converged.

        Args:
            trials: All trial results so far.

        Returns:
            True if hypervolume improvement has fallen below threshold
            for the entire sliding window.
        """
        if isinstance(trials, Sequence):
            trial_seq = trials
        else:
            trial_seq = list(trials)

        # Process new trials
        new_trials = trial_seq[self._last_index :]
        if not new_trials:
            return self._check_convergence()

        for trial in new_trials:
            if trial.status != TrialStatus.COMPLETED:
                continue

            # Extract objective values
            point = self._extract_point(trial)
            if point is None:
                continue

            # Update reference point if needed
            self._update_reference_point(point)

            # Calculate hypervolume improvement
            improvement = self._calculate_improvement(point)
            self._hypervolume_history.append(improvement)

            # Update Pareto front
            self._update_pareto_front(point)

        self._last_index = len(trial_seq)
        return self._check_convergence()

    def _extract_point(self, trial: TrialResult) -> list[float] | None:
        """Extract objective values from trial metrics."""
        point = []
        for name in self._objective_names:
            value = trial.metrics.get(name)
            if value is None:
                return None
            point.append(float(value))
        return point

    def _update_reference_point(self, point: list[float]) -> None:
        """Update reference point based on observed values."""
        if self._reference_point is not None:
            self._computed_reference = list(self._reference_point)
            return

        if self._computed_reference is None:
            # Initialize with first point + margin
            self._computed_reference = []
            for val, direction in zip(point, self._directions, strict=True):
                if direction == "maximize":
                    # For maximize, reference should be below all observed values
                    self._computed_reference.append(val - abs(val) * 0.1 - 0.1)
                else:
                    # For minimize, reference should be above all observed values
                    self._computed_reference.append(val + abs(val) * 0.1 + 0.1)
        else:
            # Update to ensure reference is dominated by all observed points
            for i, (val, direction) in enumerate(
                zip(point, self._directions, strict=True)
            ):
                if direction == "maximize":
                    self._computed_reference[i] = min(
                        self._computed_reference[i], val - abs(val) * 0.1 - 0.1
                    )
                else:
                    self._computed_reference[i] = max(
                        self._computed_reference[i], val + abs(val) * 0.1 + 0.1
                    )

    def _calculate_improvement(self, point: list[float]) -> float:
        """Calculate hypervolume improvement from adding a point."""
        if self._computed_reference is None:
            return 0.0

        # Normalize points for hypervolume calculation
        # (flip minimize objectives so all are maximize)
        def normalize(p: list[float]) -> list[float]:
            result = []
            for val, direction in zip(p, self._directions, strict=True):
                if direction == "minimize":
                    result.append(-val)
                else:
                    result.append(val)
            return result

        normalized_point = normalize(point)
        normalized_front = [normalize(p) for p in self._pareto_front]
        normalized_ref = normalize(self._computed_reference)

        # Calculate current hypervolume
        current_hv = self._simple_hypervolume(normalized_front, normalized_ref)

        # Check if point is dominated
        for front_point in normalized_front:
            if self._dominates(front_point, normalized_point):
                return 0.0

        # Add point and calculate new hypervolume
        new_front = [
            p for p in normalized_front if not self._dominates(normalized_point, p)
        ]
        new_front.append(normalized_point)
        new_hv = self._simple_hypervolume(new_front, normalized_ref)

        return max(0.0, new_hv - current_hv)

    def _dominates(self, a: list[float], b: list[float]) -> bool:
        """Check if point a dominates point b (all maximize)."""
        at_least_one_better = False
        for av, bv in zip(a, b, strict=True):
            if av < bv:
                return False
            if av > bv:
                at_least_one_better = True
        return at_least_one_better

    def _simple_hypervolume(
        self, front: list[list[float]], reference: list[float]
    ) -> float:
        """Calculate hypervolume using simple 2D algorithm or approximation."""
        if not front:
            return 0.0

        n_obj = len(reference)

        if n_obj == 1:
            # 1D: just the range
            max_val = max(p[0] for p in front)
            return max(0.0, max_val - reference[0])

        if n_obj == 2:
            # 2D: exact algorithm
            return self._hypervolume_2d(front, reference)

        # Higher dimensions: use maximum individual contribution as lower bound
        # This is monotonic - adding non-dominated points cannot decrease the value
        # We use max contribution rather than sum to avoid double-counting overlap
        max_contribution = 0.0
        for point in front:
            volume = 1.0
            for pv, rv in zip(point, reference, strict=True):
                volume *= max(0.0, pv - rv)
            max_contribution = max(max_contribution, volume)
        return max_contribution

    def _hypervolume_2d(
        self, front: list[list[float]], reference: list[float]
    ) -> float:
        """Calculate exact 2D hypervolume."""
        if not front:
            return 0.0

        # Filter points dominated by reference
        valid = [p for p in front if p[0] > reference[0] and p[1] > reference[1]]
        if not valid:
            return 0.0

        # Sort by first objective descending
        sorted_front = sorted(valid, key=lambda p: -p[0])

        hv = 0.0
        prev_y = reference[1]

        for point in sorted_front:
            if point[1] > prev_y:
                hv += (point[0] - reference[0]) * (point[1] - prev_y)
                prev_y = point[1]

        return hv

    def _update_pareto_front(self, point: list[float]) -> None:
        """Update the Pareto front with a new point."""

        # Normalize for comparison (flip minimize objectives)
        def normalize(p: list[float]) -> list[float]:
            result = []
            for val, direction in zip(p, self._directions, strict=True):
                if direction == "minimize":
                    result.append(-val)
                else:
                    result.append(val)
            return result

        normalized_point = normalize(point)

        # Check if dominated by existing front
        for front_point in self._pareto_front:
            normalized_front = normalize(front_point)
            if self._dominates(normalized_front, normalized_point):
                return  # Point is dominated, don't add

        # Remove points dominated by new point
        new_front = []
        for front_point in self._pareto_front:
            normalized_front = normalize(front_point)
            if not self._dominates(normalized_point, normalized_front):
                new_front.append(front_point)

        new_front.append(point)
        self._pareto_front = new_front

    def _check_convergence(self) -> bool:
        """Check if convergence criterion is met."""
        if len(self._hypervolume_history) < self._window:
            return False

        # Check if all improvements in window are below threshold
        return all(imp <= self._threshold for imp in self._hypervolume_history)


__all__ = [
    "CostLimitStopCondition",
    "HypervolumeConvergenceStopCondition",
    "MaxSamplesStopCondition",
    "MaxTrialsStopCondition",
    "MetricLimitStopCondition",
    "PlateauAfterNStopCondition",
    "SemanticSaturationStopCondition",
    "StopCondition",
]
