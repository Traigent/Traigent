"""Strategy preset registry and advisory selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from traigent.api.types import PresetSelection, TrialResult

PresetName = Literal[
    "max_accuracy_then_cheapest_within_epsilon",
    "quality_floor_min_cost",
    "pareto_frontier",
]

SELECTION_GRADE = "advisory"
ADVISORY_SELECTION_NOTICE = (
    "advisory selection — no statistical certificate; results are task-local"
)
MAX_ACCURACY_THEN_CHEAPEST = "max_accuracy_then_cheapest_within_epsilon"
QUALITY_FLOOR_MIN_COST = "quality_floor_min_cost"
PARETO_FRONTIER = "pareto_frontier"
VALID_PRESET_NAMES: tuple[str, ...] = (
    MAX_ACCURACY_THEN_CHEAPEST,
    QUALITY_FLOOR_MIN_COST,
    PARETO_FRONTIER,
)

_OBJECTIVES = ("accuracy", "cost")
_MAX_ACCURACY_RATIONALE = (
    "Selected the lowest-cost completed trial within the preset accuracy band."
)
_QUALITY_FLOOR_RATIONALE = (
    "Selected the lowest-cost completed trial satisfying the preset quality floor."
)
_PARETO_FRONTIER_RATIONALE = (
    "Selected all completed trials on the advisory accuracy-cost Pareto frontier."
)
_FAILED_EPSILON_RATIONALE = (
    "No completed trial had both accuracy and cost metrics for this preset."
)
_FAILED_FLOOR_RATIONALE = (
    "No completed trial satisfied the preset quality floor with a cost metric."
)
_FAILED_PARETO_RATIONALE = (
    "No completed trial had both accuracy and cost metrics for the frontier."
)


class StrategyPresetError(ValueError):
    """Base class for strategy preset validation errors."""


class UnknownStrategyPresetError(StrategyPresetError):
    """Raised when a strategy preset name is not registered."""

    def __init__(self, preset_name: str):
        valid = ", ".join(VALID_PRESET_NAMES)
        super().__init__(
            f"Unknown strategy preset {preset_name!r}. Valid presets: {valid}."
        )


class StrategyPresetValidationError(StrategyPresetError):
    """Raised when strategy preset params do not match the schema contract."""


@dataclass(frozen=True, slots=True)
class NormalizedStrategyPreset:
    """Normalized registry entry for an advisory strategy preset."""

    preset_name: str
    params: dict[str, Any]
    objectives: list[str]
    constraints: list[Callable[..., bool]] = field(default_factory=list)
    selection_rule: str = ""
    selection_rationale: str = ""

    def to_metadata(self, *, selection_rationale: str | None = None) -> dict[str, Any]:
        """Return the schema-shaped advisory metadata payload."""
        rationale = selection_rationale or self.selection_rationale
        metadata: dict[str, Any] = {
            "preset_name": self.preset_name,
            "params": dict(self.params),
            "selection_grade": SELECTION_GRADE,
        }
        if rationale:
            metadata["selection_rationale"] = rationale[:280]
        return metadata


def is_strategy_preset_name(value: str | None) -> bool:
    """Return True when ``value`` is a registered strategy preset name."""
    return value in VALID_PRESET_NAMES


def _coerce_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise StrategyPresetValidationError("strategy_params must be a mapping.")
    return dict(params)


def _validate_only_keys(
    preset_name: str, params: Mapping[str, Any], required_key: str | None
) -> None:
    expected = {required_key} if required_key is not None else set()
    provided = set(params)
    missing = expected - provided
    extra = provided - expected
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise StrategyPresetValidationError(
            f"{preset_name} requires strategy_params: {missing_list}."
        )
    if extra:
        extra_list = ", ".join(sorted(extra))
        raise StrategyPresetValidationError(
            f"{preset_name} does not accept strategy_params: {extra_list}."
        )


def _coerce_number(value: Any, *, field_name: str, preset_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyPresetValidationError(
            f"{preset_name} strategy_params.{field_name} must be a number."
        )
    return float(value)


def _quality_floor_constraint(
    floor: float,
) -> Callable[[dict[str, Any], dict[str, Any]], bool]:
    def constraint(_config: dict[str, Any], metrics: dict[str, Any]) -> bool:
        value = _metric_value(metrics, "accuracy")
        return value is not None and value >= floor

    constraint.__name__ = "strategy_quality_floor_min_cost"
    constraint.__dict__["__tvl_constraint__"] = {
        "id": "strategy_preset.quality_floor_min_cost.floor",
        "message": "Strategy preset quality_floor_min_cost quality floor unmet",
        "requires_metrics": True,
    }
    return constraint


def normalize_strategy_preset(
    preset_name: str | None,
    params: Mapping[str, Any] | None = None,
) -> NormalizedStrategyPreset:
    """Normalize a public strategy preset request into objectives and constraints."""
    if preset_name is None:
        if params:
            raise StrategyPresetValidationError(
                "strategy_params requires a strategy preset name."
            )
        raise StrategyPresetValidationError("strategy preset name is required.")

    if preset_name not in VALID_PRESET_NAMES:
        raise UnknownStrategyPresetError(preset_name)

    normalized_params = _coerce_params(params)
    if preset_name == MAX_ACCURACY_THEN_CHEAPEST:
        _validate_only_keys(preset_name, normalized_params, "epsilon")
        epsilon = _coerce_number(
            normalized_params["epsilon"],
            field_name="epsilon",
            preset_name=preset_name,
        )
        if epsilon <= 0 or epsilon > 1:
            raise StrategyPresetValidationError(
                f"{preset_name} strategy_params.epsilon must be > 0 and <= 1."
            )
        return NormalizedStrategyPreset(
            preset_name=preset_name,
            params={"epsilon": epsilon},
            objectives=list(_OBJECTIVES),
            selection_rule=preset_name,
            selection_rationale=_MAX_ACCURACY_RATIONALE,
        )

    if preset_name == QUALITY_FLOOR_MIN_COST:
        _validate_only_keys(preset_name, normalized_params, "floor")
        floor = _coerce_number(
            normalized_params["floor"],
            field_name="floor",
            preset_name=preset_name,
        )
        if floor < 0 or floor > 1:
            raise StrategyPresetValidationError(
                f"{preset_name} strategy_params.floor must be >= 0 and <= 1."
            )
        return NormalizedStrategyPreset(
            preset_name=preset_name,
            params={"floor": floor},
            objectives=list(_OBJECTIVES),
            constraints=[_quality_floor_constraint(floor)],
            selection_rule=preset_name,
            selection_rationale=_QUALITY_FLOOR_RATIONALE,
        )

    _validate_only_keys(preset_name, normalized_params, None)
    return NormalizedStrategyPreset(
        preset_name=preset_name,
        params={},
        objectives=list(_OBJECTIVES),
        selection_rule=preset_name,
        selection_rationale=_PARETO_FRONTIER_RATIONALE,
    )


def normalize(
    preset_name: str | None,
    params: Mapping[str, Any] | None = None,
) -> NormalizedStrategyPreset:
    """Compatibility alias matching the registry contract wording."""
    return normalize_strategy_preset(preset_name, params)


def _metric_value(metrics: Mapping[str, Any] | None, name: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _completed_trials(
    trials: Iterable[TrialResult],
) -> list[tuple[int, TrialResult]]:
    def is_completed(trial: TrialResult) -> bool:
        is_successful = getattr(trial, "is_successful", None)
        if isinstance(is_successful, bool):
            return is_successful
        status = getattr(trial, "status", None)
        status_value = getattr(status, "value", status)
        return status_value == "completed"

    return [(index, trial) for index, trial in enumerate(trials) if is_completed(trial)]


def _empty_selection(
    preset: NormalizedStrategyPreset,
    rationale: str,
) -> PresetSelection:
    return PresetSelection(
        preset_name=preset.preset_name,
        params=dict(preset.params),
        selection_grade=SELECTION_GRADE,
        selection_rationale=rationale,
        status="failed",
    )


def _selected_single(
    preset: NormalizedStrategyPreset,
    *,
    rationale: str,
    index: int,
    trial: TrialResult,
) -> PresetSelection:
    config = dict(trial.config or {})
    return PresetSelection(
        preset_name=preset.preset_name,
        params=dict(preset.params),
        selection_grade=SELECTION_GRADE,
        selection_rationale=rationale,
        status="selected",
        selected_config=config,
        selected_configs=[config],
        selected_trial_indices=[index],
    )


def _select_max_accuracy_then_cheapest(
    preset: NormalizedStrategyPreset,
    trials: Iterable[TrialResult],
) -> PresetSelection:
    epsilon = float(preset.params["epsilon"])
    completed = _completed_trials(trials)
    accuracy_rows = [
        (index, trial, accuracy)
        for index, trial in completed
        if (accuracy := _metric_value(trial.metrics, "accuracy")) is not None
    ]
    if not accuracy_rows:
        return _empty_selection(preset, _FAILED_EPSILON_RATIONALE)

    best_accuracy = max(accuracy for _, _, accuracy in accuracy_rows)
    boundary = best_accuracy - epsilon
    candidates = [
        (index, trial, cost)
        for index, trial, accuracy in accuracy_rows
        if accuracy >= boundary
        if (cost := _metric_value(trial.metrics, "cost")) is not None
    ]
    if not candidates:
        return _empty_selection(preset, _FAILED_EPSILON_RATIONALE)

    index, trial, _cost = min(candidates, key=lambda row: (row[2], row[0]))
    return _selected_single(
        preset,
        rationale=_MAX_ACCURACY_RATIONALE,
        index=index,
        trial=trial,
    )


def _select_quality_floor_min_cost(
    preset: NormalizedStrategyPreset,
    trials: Iterable[TrialResult],
) -> PresetSelection:
    floor = float(preset.params["floor"])
    candidates = [
        (index, trial, cost)
        for index, trial in _completed_trials(trials)
        if (accuracy := _metric_value(trial.metrics, "accuracy")) is not None
        and accuracy >= floor
        if (cost := _metric_value(trial.metrics, "cost")) is not None
    ]
    if not candidates:
        return _empty_selection(preset, _FAILED_FLOOR_RATIONALE)

    index, trial, _cost = min(candidates, key=lambda row: (row[2], row[0]))
    return _selected_single(
        preset,
        rationale=_QUALITY_FLOOR_RATIONALE,
        index=index,
        trial=trial,
    )


def _dominates(left: tuple[float, float], right: tuple[float, float]) -> bool:
    left_accuracy, left_cost = left
    right_accuracy, right_cost = right
    return (
        left_accuracy >= right_accuracy
        and left_cost <= right_cost
        and (left_accuracy > right_accuracy or left_cost < right_cost)
    )


def _select_pareto_frontier(
    preset: NormalizedStrategyPreset,
    trials: Iterable[TrialResult],
) -> PresetSelection:
    rows = [
        (index, trial, (accuracy, cost))
        for index, trial in _completed_trials(trials)
        if (accuracy := _metric_value(trial.metrics, "accuracy")) is not None
        if (cost := _metric_value(trial.metrics, "cost")) is not None
    ]
    if not rows:
        return _empty_selection(preset, _FAILED_PARETO_RATIONALE)

    frontier: list[tuple[int, TrialResult]] = []
    for index, trial, metrics_pair in rows:
        if any(
            other_index != index and _dominates(other_pair, metrics_pair)
            for other_index, _other_trial, other_pair in rows
        ):
            continue
        frontier.append((index, trial))

    selected_configs = [dict(trial.config or {}) for _, trial in frontier]
    return PresetSelection(
        preset_name=preset.preset_name,
        params=dict(preset.params),
        selection_grade=SELECTION_GRADE,
        selection_rationale=_PARETO_FRONTIER_RATIONALE,
        status="selected",
        selected_config=None,
        selected_configs=selected_configs,
        selected_trial_indices=[index for index, _trial in frontier],
    )


def select_strategy_preset(
    preset: NormalizedStrategyPreset,
    trials: Iterable[TrialResult],
) -> PresetSelection:
    """Apply a normalized advisory preset selection rule to completed trials."""
    if preset.preset_name == MAX_ACCURACY_THEN_CHEAPEST:
        return _select_max_accuracy_then_cheapest(preset, trials)
    if preset.preset_name == QUALITY_FLOOR_MIN_COST:
        return _select_quality_floor_min_cost(preset, trials)
    if preset.preset_name == PARETO_FRONTIER:
        return _select_pareto_frontier(preset, trials)
    raise UnknownStrategyPresetError(preset.preset_name)


def calculate_weighted_trial_score(
    trial: TrialResult,
    weight_dict: Mapping[str, float],
) -> float | None:
    """Calculate a weighted metric score for CLI/result reranking."""
    if not trial.metrics:
        return None

    score = 0.0
    has_metrics = False
    for metric, weight in weight_dict.items():
        metric_value = trial.metrics.get(metric)
        if isinstance(metric_value, (int, float)) and not isinstance(
            metric_value, bool
        ):
            score += weight * float(metric_value)
            has_metrics = True

    return score if has_metrics else None
