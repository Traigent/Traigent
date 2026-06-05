from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.strategy_presets import (
    ADVISORY_SELECTION_NOTICE,
    MAX_ACCURACY_THEN_CHEAPEST,
    PARETO_FRONTIER,
    QUALITY_FLOOR_MIN_COST,
    StrategyPresetValidationError,
    UnknownStrategyPresetError,
    normalize_strategy_preset,
    select_strategy_preset,
)
from traigent.api.types import TrialResult, TrialStatus


def _trial(index: int, accuracy: float, cost: float, *, completed: bool = True):
    return TrialResult(
        trial_id=f"trial_{index}",
        config={"candidate": index},
        metrics={"accuracy": accuracy, "cost": cost},
        status=TrialStatus.COMPLETED if completed else TrialStatus.FAILED,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )


def test_normalize_max_accuracy_then_cheapest_metadata_shape():
    preset = normalize_strategy_preset(
        MAX_ACCURACY_THEN_CHEAPEST,
        {"epsilon": 0.02},
    )

    assert preset.objectives == ["accuracy", "cost"]
    assert preset.constraints == []
    assert preset.selection_rule == MAX_ACCURACY_THEN_CHEAPEST
    assert preset.to_metadata() == {
        "preset_name": MAX_ACCURACY_THEN_CHEAPEST,
        "params": {"epsilon": 0.02},
        "selection_grade": "advisory",
        "selection_rationale": (
            "Selected the lowest-cost completed trial within the preset accuracy band."
        ),
    }


@pytest.mark.parametrize(
    ("preset_name", "params"),
    [
        (MAX_ACCURACY_THEN_CHEAPEST, {"epsilon": 0}),
        (MAX_ACCURACY_THEN_CHEAPEST, {"epsilon": 1.01}),
        (MAX_ACCURACY_THEN_CHEAPEST, {}),
        (MAX_ACCURACY_THEN_CHEAPEST, {"epsilon": 0.1, "extra": 1}),
        (QUALITY_FLOOR_MIN_COST, {"floor": -0.01}),
        (QUALITY_FLOOR_MIN_COST, {"floor": 1.01}),
        (QUALITY_FLOOR_MIN_COST, {}),
        (PARETO_FRONTIER, {"epsilon": 0.1}),
    ],
)
def test_normalize_rejects_schema_bound_violations(preset_name, params):
    with pytest.raises(StrategyPresetValidationError):
        normalize_strategy_preset(preset_name, params)


def test_normalize_unknown_lists_valid_names():
    with pytest.raises(UnknownStrategyPresetError) as exc_info:
        normalize_strategy_preset("unknown", {})

    message = str(exc_info.value)
    assert MAX_ACCURACY_THEN_CHEAPEST in message
    assert QUALITY_FLOOR_MIN_COST in message
    assert PARETO_FRONTIER in message


def test_max_accuracy_epsilon_boundary_is_inclusive_and_tie_breaks_by_index():
    preset = normalize_strategy_preset(
        MAX_ACCURACY_THEN_CHEAPEST,
        {"epsilon": 0.02},
    )
    trials = [
        _trial(0, 0.98, 0.02),
        _trial(1, 0.96, 0.01),
        _trial(2, 0.96, 0.01),
        _trial(3, 0.95, 0.001),
    ]

    selection = select_strategy_preset(preset, trials)

    assert selection.status == "selected"
    assert selection.selection_grade == "advisory"
    assert selection.selected_config == {"candidate": 1}
    assert selection.selected_trial_indices == [1]


def test_quality_floor_unmet_fails_closed():
    preset = normalize_strategy_preset(QUALITY_FLOOR_MIN_COST, {"floor": 0.9})
    selection = select_strategy_preset(preset, [_trial(0, 0.89, 0.001)])

    assert selection.status == "failed"
    assert selection.selected_config is None
    assert selection.selected_configs == []
    assert selection.selection_grade == "advisory"


def test_quality_floor_constraint_requires_metrics():
    preset = normalize_strategy_preset(QUALITY_FLOOR_MIN_COST, {"floor": 0.8})
    [constraint] = preset.constraints

    assert constraint({}, {"accuracy": 0.8}) is True
    assert constraint({}, {"accuracy": 0.79}) is False
    assert constraint.__dict__["__tvl_constraint__"]["requires_metrics"] is True


def test_pareto_frontier_returns_frontier_set():
    preset = normalize_strategy_preset(PARETO_FRONTIER, {})
    selection = select_strategy_preset(
        preset,
        [
            _trial(0, 0.90, 0.03),
            _trial(1, 0.88, 0.01),
            _trial(2, 0.85, 0.02),
        ],
    )

    assert selection.status == "selected"
    assert selection.selected_trial_indices == [0, 1]
    assert selection.selected_configs == [{"candidate": 0}, {"candidate": 1}]
    assert "statistical certificate" in ADVISORY_SELECTION_NOTICE
