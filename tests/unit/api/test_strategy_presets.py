from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

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
from traigent.api.types import PresetSelection, TrialResult, TrialStatus


def _trial(
    index: int,
    accuracy: float,
    cost: float | None,
    *,
    completed: bool = True,
):
    metrics = {"accuracy": accuracy}
    if cost is not None:
        metrics["cost"] = cost
    return TrialResult(
        trial_id=f"trial_{index}",
        config={"candidate": index},
        metrics=metrics,
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


def test_max_accuracy_epsilon_boundary_is_float_tolerant():
    preset = normalize_strategy_preset(
        MAX_ACCURACY_THEN_CHEAPEST,
        {"epsilon": 0.7},
    )
    trials = [
        _trial(0, 0.9, 0.5),
        _trial(1, 0.2, 0.01),
    ]

    selection = select_strategy_preset(preset, trials)

    assert selection.status == "selected"
    assert selection.selected_config == {"candidate": 1}
    assert selection.selected_trial_indices == [1]


def test_quality_floor_unmet_fails_closed():
    preset = normalize_strategy_preset(QUALITY_FLOOR_MIN_COST, {"floor": 0.9})
    selection = select_strategy_preset(preset, [_trial(0, 0.89, 0.001)])

    assert selection.status == "failed"
    assert selection.selected_config is None
    assert selection.selected_configs == []
    assert selection.selection_grade == "advisory"


def test_quality_floor_empty_trials_fails_closed():
    preset = normalize_strategy_preset(QUALITY_FLOOR_MIN_COST, {"floor": 0.9})
    selection = select_strategy_preset(preset, [])

    assert selection.status == "failed"
    assert selection.selected_config is None
    assert selection.selected_configs == []
    assert selection.selection_grade == "advisory"


def test_quality_floor_all_failed_trials_fails_closed():
    preset = normalize_strategy_preset(QUALITY_FLOOR_MIN_COST, {"floor": 0.9})
    selection = select_strategy_preset(
        preset,
        [
            _trial(0, 0.95, 0.01, completed=False),
            _trial(1, 0.97, 0.02, completed=False),
        ],
    )

    assert selection.status == "failed"
    assert selection.selected_config is None
    assert selection.selected_configs == []
    assert selection.selection_grade == "advisory"


def test_quality_floor_missing_cost_metric_returns_failed_selection():
    preset = normalize_strategy_preset(QUALITY_FLOOR_MIN_COST, {"floor": 0.8})
    selection = select_strategy_preset(preset, [_trial(0, 0.9, None)])

    assert selection.status == "failed"
    assert selection.selected_config is None
    assert selection.selected_configs == []
    assert selection.selection_grade == "advisory"


def test_quality_floor_preset_does_not_create_search_constraints():
    preset = normalize_strategy_preset(QUALITY_FLOOR_MIN_COST, {"floor": 0.8})

    assert preset.constraints == []


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


def test_strategy_preset_metadata_shape_matches_schema_component():
    repo_root = Path(__file__).resolve().parents[3]
    schema_path = (
        repo_root.parent
        / "TraigentSchema"
        / "traigent_schema"
        / "schemas"
        / "optimization"
        / "strategy_preset_schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    required_keys = {"preset_name", "params", "selection_grade"}
    assert set(schema["required"]) == required_keys
    assert set(schema["properties"]) == {
        "preset_name",
        "params",
        "selection_grade",
        "selection_rationale",
    }
    assert schema["definitions"]["SelectionGrade"]["enum"] == ["advisory"]

    metadata = normalize_strategy_preset(
        QUALITY_FLOOR_MIN_COST,
        {"floor": 0.8},
    ).to_metadata()
    assert required_keys <= set(metadata)
    assert set(metadata) <= set(schema["properties"])
    assert metadata["selection_grade"] == "advisory"


def test_persisted_non_advisory_selection_grade_is_rejected():
    selection = PresetSelection.from_dict(
        {
            "preset_name": PARETO_FRONTIER,
            "params": {},
            "selection_grade": "certified",
            "selection_rationale": "should not render",
            "status": "selected",
            "selected_configs": [{"candidate": 0}],
        }
    )

    assert selection is None
