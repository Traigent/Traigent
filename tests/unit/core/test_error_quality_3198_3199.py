"""Error quality for ObjectiveSchema.from_dict (#3198) and datasets (#3199)."""

from __future__ import annotations

import asyncio
import json
import pathlib

import pytest

import traigent
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.evaluators.base import Dataset
from traigent.utils.exceptions import ValidationError

CANARY = "CANARY_ROW_SECRET_7731"


def _good_schema() -> dict:
    return ObjectiveSchema.from_objectives(
        [ObjectiveDefinition(name="accuracy", orientation="maximize", weight=1.0)]
    ).to_dict()


def test_schema_round_trip_control() -> None:
    d = _good_schema()
    assert ObjectiveSchema.from_dict(d).to_dict() == d


@pytest.mark.parametrize(
    "data, needle",
    [
        ({}, "objectives is required"),
        (None, "must be a mapping"),
        ({"objectives": "x"}, "objectives must be a non-empty list"),
        ({"objectives": []}, "objectives must be a non-empty list"),
        ({"objectives": [{}]}, "objectives[0].name is required"),
        ({"objectives": [{"name": 5}]}, "objectives[0].name must be"),
        ({"objectives": ["x"]}, "objectives[0] must be a mapping"),
        (
            {"objectives": [{"name": "a", "weight": CANARY}]},
            "objectives[0].weight must be a number",
        ),
    ],
)
def test_schema_from_dict_bad_input_is_named_value_error(data, needle) -> None:
    with pytest.raises(ValueError) as info:
        ObjectiveSchema.from_dict(data)
    assert needle in str(info.value)
    assert CANARY not in str(info.value)


@pytest.mark.parametrize("drop", ["weights_sum", "weights_normalized"])
def test_schema_missing_top_level_key(drop) -> None:
    d = _good_schema()
    del d[drop]
    with pytest.raises(ValueError, match=f"{drop} is required"):
        ObjectiveSchema.from_dict(d)


def test_definition_from_dict_bad_input() -> None:
    with pytest.raises(ValueError, match="objective.name is required"):
        ObjectiveDefinition.from_dict({})
    with pytest.raises(ValueError, match="must be a mapping"):
        ObjectiveDefinition.from_dict(None)  # type: ignore[arg-type]
    ok = ObjectiveDefinition.from_dict({"name": "accuracy", "orientation": "maximize"})
    assert ok.name == "accuracy"


@pytest.fixture(autouse=True)
def _dataset_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))


def _rows(n: int = 3) -> bytes:
    return "\n".join(
        json.dumps({"input": {"x": i}, "output": str(i)}) for i in range(n)
    ).encode()


def _optimize(path):
    @traigent.optimize(
        eval_dataset=path,
        objectives=["accuracy"],
        configuration_space={"a": [1, 2]},
        injection_mode="parameter",
        config_param="config",
        offline=True,
        scoring_function=lambda output, expected, **k: float(output == expected),
    )
    def f(x, config):
        return str(x)

    return asyncio.run(f.optimize(algorithm="grid", max_trials=2))


def test_bom_file_runs_through_optimize(tmp_path) -> None:
    p = tmp_path / "bom.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + _rows())
    assert len(_optimize(str(p)).trials) == 2
    assert len(Dataset.from_jsonl(str(p)).examples) == 3


def test_plain_file_control(tmp_path) -> None:
    p = tmp_path / "plain.jsonl"
    p.write_bytes(_rows())
    assert len(_optimize(str(p)).trials) == 2


def test_path_object_accepted(tmp_path) -> None:
    p = tmp_path / "p.jsonl"
    p.write_bytes(_rows())
    assert len(_optimize(pathlib.Path(p)).trials) == 2
    assert len(Dataset.from_jsonl(pathlib.Path(p)).examples) == 3


def test_validation_message_has_reason_and_column_not_content(tmp_path) -> None:

    p = tmp_path / "bad.jsonl"
    p.write_bytes(_rows(1) + b"\n" + ('{"input": "' + CANARY + '"').encode() + b"\n")
    with pytest.raises(Exception) as info:
        _optimize(str(p))
    text = str(info.value)
    assert "Invalid JSON" in text and "column" in text
    assert "Expecting" in text
    assert CANARY not in text


@pytest.mark.parametrize("bad", [5, None])
def test_from_jsonl_non_path_names_file_path(bad) -> None:
    with pytest.raises(ValidationError, match="file_path must be a str or os.PathLike"):
        Dataset.from_jsonl(bad)


@pytest.mark.parametrize(
    "kwargs, field, exc",
    [
        ({"weight": CANARY}, "weight", TypeError),
        ({"weight": True}, "weight", TypeError),
        ({"bounds": (CANARY, "b")}, "bounds", TypeError),
        ({"bounds": CANARY[:2]}, "bounds", ValueError),
        ({"bounds": (None, 1)}, "bounds", TypeError),
        ({"bounds": (0, None)}, "bounds", TypeError),
    ],
)
def test_definition_rejects_mistyped_weight_and_bounds_3196(kwargs, field, exc) -> None:
    with pytest.raises(exc) as info:
        ObjectiveDefinition(
            name="accuracy", orientation="maximize", **{"weight": 1.0, **kwargs}
        )
    assert field in str(info.value)
    assert CANARY not in str(info.value)


def test_definition_valid_weight_and_bounds_control() -> None:
    d = ObjectiveDefinition(
        name="accuracy", orientation="maximize", weight=2, bounds=(0, 1.5)
    )
    assert d.weight == 2 and d.bounds == (0, 1.5)
