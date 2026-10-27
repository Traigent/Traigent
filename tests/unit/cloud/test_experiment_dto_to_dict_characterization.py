"""Characterization of ExperimentDTO.to_dict dataset-alias resolution."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from traigent.cloud.dtos import ExperimentDTO, ExperimentListRunSummaryDTO

DEFAULT_EVALSET = "local-evalset-001"

FIELDS = (
    "id",
    "name",
    "description",
    "configurations",
    "measures",
    "agent_id",
    "evaluation_set_id",
    "model_parameters_id",
    "dataset_id",
    "benchmark_id",
    "experiment_parameters",
    "metadata",
    "status",
    "created_at",
    "updated_at",
    "configuration_runs_count",
    "total_examples",
    "optimization_runs_count",
    "experiment_run",
)


def _dto(**kwargs: Any) -> ExperimentDTO:
    base: dict[str, Any] = {
        "id": "e1",
        "name": "n",
        "description": "d",
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-02T00:00:00+00:00",
    }
    base.update(kwargs)
    return ExperimentDTO(**base)


def _dataset_view(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("dataset_id", "evaluation_set_id", "eval_dataset_id", "benchmark_id")
    return {k: result[k] for k in keys if k in result}


def test_defaults_full_result() -> None:
    result = _dto().to_dict()
    assert list(result) == [
        "id",
        "name",
        "description",
        "configurations",
        "measures",
        "experiment_parameters",
        "metadata",
        "created_at",
        "updated_at",
        "agent_id",
        "dataset_id",
        "evaluation_set_id",
        "eval_dataset_id",
        "model_parameters_id",
    ]
    assert set(result["configurations"]) == {"infrastructure", "parameters"}
    assert result["configurations"]["parameters"] == {}
    assert result == {
        "id": "e1",
        "name": "n",
        "description": "d",
        "configurations": result["configurations"],
        "measures": ["score", "cost", "latency"],
        "experiment_parameters": {},
        "metadata": {},
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-02T00:00:00+00:00",
        "agent_id": "local-agent-001",
        "dataset_id": DEFAULT_EVALSET,
        "evaluation_set_id": DEFAULT_EVALSET,
        "eval_dataset_id": DEFAULT_EVALSET,
        "model_parameters_id": "local-model-params-001",
    }


@pytest.mark.parametrize(
    ("kwargs", "resolved"),
    [
        ({"dataset_id": "ds"}, "ds"),
        ({"benchmark_id": "bm"}, "bm"),
        ({"evaluation_set_id": "es"}, "es"),
        ({"dataset_id": "x", "benchmark_id": "x"}, "x"),
        ({"dataset_id": "x", "evaluation_set_id": "x"}, "x"),
        ({"benchmark_id": "x", "evaluation_set_id": "x"}, "x"),
        (
            {"dataset_id": "x", "benchmark_id": "x", "evaluation_set_id": "x"},
            "x",
        ),
        ({"dataset_id": "ds", "evaluation_set_id": DEFAULT_EVALSET}, "ds"),
        ({"benchmark_id": "bm", "evaluation_set_id": DEFAULT_EVALSET}, "bm"),
    ],
)
def test_alias_resolution_and_legacy_benchmark_field(
    kwargs: dict[str, Any], resolved: str
) -> None:
    result = _dto(**kwargs).to_dict()
    expected = {
        "dataset_id": resolved,
        "evaluation_set_id": resolved,
        "eval_dataset_id": resolved,
        "benchmark_id": resolved,
    }
    assert _dataset_view(result) == expected


def test_default_evalset_has_no_legacy_benchmark_field() -> None:
    result = _dto().to_dict()
    assert "benchmark_id" not in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_id": "a", "benchmark_id": "b"},
        {"dataset_id": "a", "evaluation_set_id": "b"},
        {"benchmark_id": "a", "evaluation_set_id": "b"},
        {"dataset_id": "a", "benchmark_id": "b", "evaluation_set_id": "a"},
    ],
)
def test_conflicting_aliases_raise_exact_message(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError) as exc:
        _dto(**kwargs).to_dict()
    assert str(exc.value) == (
        "Conflicting dataset aliases supplied for ExperimentDTO; "
        "dataset_id, evaluation_set_id, and benchmark_id must match"
    )


def test_default_evalset_never_conflicts_with_explicit_alias() -> None:
    result = _dto(dataset_id="a", benchmark_id="a").to_dict()
    assert result["dataset_id"] == "a"


def test_empty_and_nonempty_aliases_conflict() -> None:
    with pytest.raises(ValueError):
        _dto(dataset_id="", benchmark_id="b").to_dict()


def test_empty_dataset_alone_resolves_to_evaluation_set_with_legacy_field() -> None:
    result = _dto(dataset_id="").to_dict()
    assert _dataset_view(result) == {
        "dataset_id": DEFAULT_EVALSET,
        "evaluation_set_id": DEFAULT_EVALSET,
        "eval_dataset_id": DEFAULT_EVALSET,
        "benchmark_id": DEFAULT_EVALSET,
    }


def test_empty_evaluation_set_id_is_explicit_and_resolves_to_empty() -> None:
    result = _dto(evaluation_set_id="").to_dict()
    assert result["dataset_id"] == ""
    assert result["evaluation_set_id"] == ""
    assert result["eval_dataset_id"] == ""
    assert result["benchmark_id"] == ""


def test_empty_evaluation_set_id_with_dataset_id_conflicts() -> None:
    with pytest.raises(ValueError):
        _dto(dataset_id="ds", evaluation_set_id="").to_dict()


def test_unhashable_alias_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _dto(dataset_id=["a"], benchmark_id=["a"]).to_dict()


def test_optional_fields_only_when_set_and_ints_coerced() -> None:
    result = _dto(
        status="running",
        configuration_runs_count=True,
        total_examples="5",
        optimization_runs_count=2.9,
    ).to_dict()
    assert result["status"] == "running"
    assert result["configuration_runs_count"] == 1
    assert type(result["configuration_runs_count"]) is int
    assert result["total_examples"] == 5
    assert result["optimization_runs_count"] == 2
    assert "experiment_run" not in result
    bare = _dto().to_dict()
    for key in (
        "status",
        "configuration_runs_count",
        "total_examples",
        "optimization_runs_count",
        "experiment_run",
    ):
        assert key not in bare


def test_zero_counts_are_kept() -> None:
    result = _dto(
        configuration_runs_count=0, total_examples=0, optimization_runs_count=0
    ).to_dict()
    assert result["configuration_runs_count"] == 0
    assert result["total_examples"] == 0
    assert result["optimization_runs_count"] == 0


def test_empty_status_string_is_kept() -> None:
    assert _dto(status="").to_dict()["status"] == ""


def test_bad_count_value_raises() -> None:
    with pytest.raises(ValueError):
        _dto(total_examples="abc").to_dict()
    with pytest.raises(TypeError):
        _dto(configuration_runs_count=[1]).to_dict()


def test_option_order_in_result() -> None:
    run = {"k": "v"}
    result = _dto(
        status="s",
        configuration_runs_count=1,
        total_examples=2,
        optimization_runs_count=3,
        experiment_run=run,
        benchmark_id="b",
    ).to_dict()
    assert list(result)[-6:] == [
        "benchmark_id",
        "status",
        "configuration_runs_count",
        "total_examples",
        "optimization_runs_count",
        "experiment_run",
    ]


def test_experiment_run_dict_is_passed_through_same_object() -> None:
    run: dict[str, Any] = {"k": "v"}
    result = _dto(experiment_run=run).to_dict()
    assert result["experiment_run"] is run


def test_experiment_run_empty_dict_is_kept() -> None:
    assert _dto(experiment_run={}).to_dict()["experiment_run"] == {}


def test_experiment_run_dto_is_serialized() -> None:
    summary = ExperimentListRunSummaryDTO(
        id="r", experiment_id="e1", status="completed"
    )
    result = _dto(experiment_run=summary).to_dict()
    assert result["experiment_run"] == summary.to_dict()
    assert isinstance(result["experiment_run"], dict)


def test_generated_timestamps_when_unset() -> None:
    result = _dto(created_at=None, updated_at=None).to_dict()
    for key in ("created_at", "updated_at"):
        parsed = datetime.fromisoformat(result[key])
        assert parsed.tzinfo is not None
        offset = parsed.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0


def test_empty_timestamp_strings_are_replaced() -> None:
    result = _dto(created_at="", updated_at="").to_dict()
    assert result["created_at"] != ""
    datetime.fromisoformat(result["created_at"])


class _Traced(ExperimentDTO):
    events: list[str] = []

    def __getattribute__(self, name: str) -> Any:
        if name in FIELDS:
            _Traced.events.append(name)
        return super().__getattribute__(name)


def _traced(**kwargs: Any) -> _Traced:
    base: dict[str, Any] = {
        "id": "e1",
        "name": "n",
        "description": "d",
        "created_at": "c",
        "updated_at": "u",
    }
    base.update(kwargs)
    _Traced.events = []
    return _Traced(**base)


def test_read_trace_defaults() -> None:
    dto = _traced()
    dto.to_dict()
    assert _Traced.events == [
        "dataset_id",
        "benchmark_id",
        "evaluation_set_id",
        "dataset_id",
        "benchmark_id",
        "evaluation_set_id",
        "evaluation_set_id",
        "dataset_id",
        "benchmark_id",
        "evaluation_set_id",
        "id",
        "name",
        "description",
        "configurations",
        "measures",
        "experiment_parameters",
        "metadata",
        "created_at",
        "updated_at",
        "agent_id",
        "model_parameters_id",
        "status",
        "configuration_runs_count",
        "total_examples",
        "optimization_runs_count",
        "experiment_run",
    ]


def test_read_trace_explicit_dataset_and_options() -> None:
    dto = _traced(
        dataset_id="ds",
        status="s",
        configuration_runs_count=1,
        total_examples=2,
        optimization_runs_count=3,
        experiment_run={},
    )
    dto.to_dict()
    assert _Traced.events == [
        "dataset_id",
        "benchmark_id",
        "evaluation_set_id",
        "dataset_id",
        "dataset_id",
        "id",
        "name",
        "description",
        "configurations",
        "measures",
        "experiment_parameters",
        "metadata",
        "created_at",
        "updated_at",
        "agent_id",
        "model_parameters_id",
        "status",
        "status",
        "configuration_runs_count",
        "configuration_runs_count",
        "total_examples",
        "total_examples",
        "optimization_runs_count",
        "optimization_runs_count",
        "experiment_run",
        "experiment_run",
        "experiment_run",
    ]


def test_conflict_raises_before_any_result_field_is_read() -> None:
    dto = _traced(dataset_id="a", benchmark_id="b")
    with pytest.raises(ValueError):
        dto.to_dict()
    assert _Traced.events == ["dataset_id", "benchmark_id", "evaluation_set_id"]
