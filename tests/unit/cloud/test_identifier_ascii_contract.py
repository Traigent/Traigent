"""Identifier checks match the ASCII ``^[a-zA-Z_][a-zA-Z0-9_]*$`` contract."""

import pytest

from traigent.cloud.dtos import ExampleMeasure, MeasuresDict
from traigent.wrapper.service import TraigentService

BAD_NAMES = ["précision", "a٣", "temp\n", "score\n", "1abc", "a-b", ""]
GOOD_NAMES = ["temperature", "_t", "x1", "A_b"]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_example_measure_rejects_bad_metric_key(name: str) -> None:
    with pytest.raises(ValueError):
        ExampleMeasure({"example_id": "e", "metrics": {name: 1.0}})


@pytest.mark.parametrize("name", GOOD_NAMES)
def test_example_measure_accepts_ascii_key(name: str) -> None:
    ExampleMeasure({"example_id": "e", "metrics": {name: 1.0}})


@pytest.mark.parametrize("name", BAD_NAMES)
def test_measures_dict_rejects_bad_key(name: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        MeasuresDict({name: 1.0})
    m = MeasuresDict()
    with pytest.raises((ValueError, TypeError)):
        m[name] = 1.0


@pytest.mark.parametrize("name", GOOD_NAMES)
def test_measures_dict_accepts_ascii_key(name: str) -> None:
    assert MeasuresDict({name: 1.0})[name] == 1.0


def _service() -> TraigentService:
    return TraigentService()


@pytest.mark.parametrize("name", BAD_NAMES)
def test_service_tvar_name_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid TVAR name"):
        _service()._normalize_tvars(
            {name: {"type": "float", "domain": {"range": [0.0, 1.0]}}}
        )


@pytest.mark.parametrize("name", GOOD_NAMES)
def test_service_tvar_name_accepted(name: str) -> None:
    out = _service()._normalize_tvars(
        {name: {"type": "float", "domain": {"range": [0.0, 1.0]}}}
    )
    assert name in out


@pytest.mark.parametrize("name", BAD_NAMES)
def test_service_evaluation_kwarg_name_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid evaluation kwarg name"):
        _service()._normalize_evaluation_kwarg_definition(
            {"name": name, "type": "str", "domain": {"values": ["a"]}}
        )


@pytest.mark.parametrize("name", GOOD_NAMES)
def test_service_evaluation_kwarg_name_accepted(name: str) -> None:
    out = _service()._normalize_evaluation_kwarg_definition(
        {"name": name, "type": "str", "domain": {"values": ["a"]}}
    )
    assert out.name == name
