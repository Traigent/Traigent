"""Characterization tests for LocalExecutionAdapter._normalize_examples."""

import pytest

from traigent.adapters.execution_adapter import LocalExecutionAdapter
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.fixture
def adapter():
    return LocalExecutionAdapter(agent_builder=lambda spec: None)


def test_dict_dataset_normalizes_fields_and_defaults(adapter):
    out = adapter._normalize_examples(
        {
            "examples": [
                {"input": {"q": 1}, "expected_output": "a", "id": "x"},
                {"input": 2, "metadata": None},
                {"input": 3, "metadata": {"k": "v"}, "id": 0},
            ]
        }
    )
    assert out == [
        {"id": "x", "input": {"q": 1}, "expected_output": "a", "metadata": {}},
        {"id": 1, "input": 2, "expected_output": None, "metadata": {}},
        {"id": 0, "input": 3, "expected_output": None, "metadata": {"k": "v"}},
    ]


def test_dict_empty_examples_list_is_accepted(adapter):
    assert adapter._normalize_examples({"examples": []}) == []


@pytest.mark.parametrize(
    "bad", [{}, {"examples": None}, {"examples": "x"}, {"examples": {}}]
)
def test_dict_without_list_examples_raises(adapter, bad):
    with pytest.raises(ValueError, match=r"must be a non-empty list"):
        adapter._normalize_examples(bad)


def test_non_dict_example_raises_with_index(adapter):
    with pytest.raises(ValueError, match="example at index 1 must be a dict"):
        adapter._normalize_examples({"examples": [{"input": 1}, "no"]})


def test_missing_input_raises_with_index(adapter):
    with pytest.raises(ValueError, match="example at index 0 missing 'input'"):
        adapter._normalize_examples({"examples": [{"expected_output": 1}]})


def test_non_dict_metadata_raises_with_index(adapter):
    with pytest.raises(ValueError, match="example at index 1 has non-dict 'metadata'"):
        adapter._normalize_examples(
            {"examples": [{"input": 1}, {"input": 2, "metadata": [1]}]}
        )


def test_type_check_precedes_input_check(adapter):
    with pytest.raises(ValueError, match="must be a dict"):
        adapter._normalize_examples({"examples": [[]]})


def test_dataset_instance_normalizes_with_example_id_metadata(adapter):
    ds = Dataset(
        examples=[
            EvaluationExample(
                input_data={"a": 1},
                expected_output="x",
                metadata={"example_id": "e1"},
            ),
            EvaluationExample(input_data={"a": 2}, expected_output="y"),
        ]
    )
    assert adapter._normalize_examples(ds) == [
        {
            "id": "e1",
            "input": {"a": 1},
            "expected_output": "x",
            "metadata": {"example_id": "e1"},
        },
        {"id": 1, "input": {"a": 2}, "expected_output": "y", "metadata": {}},
    ]


def test_dataset_non_dict_metadata_raises(adapter):
    ds = Dataset(examples=[EvaluationExample(input_data=1, expected_output=1)])
    ds.examples[0].metadata = ["bad"]
    with pytest.raises(ValueError, match="example at index 0 has non-dict metadata"):
        adapter._normalize_examples(ds)


@pytest.mark.parametrize("bad", [None, [], "abc", 3])
def test_other_types_raise_type_error(adapter, bad):
    with pytest.raises(TypeError, match="dict with 'examples' or a Dataset"):
        adapter._normalize_examples(bad)
