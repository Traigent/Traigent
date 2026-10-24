"""Characterization tests for OptimizedFunction._dataset_from_source path lists."""

from types import SimpleNamespace

import pytest

from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ConfigurationError


def _dataset(*labels: str) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": label}, expected_output=label)
            for label in labels
        ],
        name="part",
    )


def _resolve(source):
    owner = SimpleNamespace(
        _combine_dataset_paths=OptimizedFunction._combine_dataset_paths
    )
    return OptimizedFunction._dataset_from_source(owner, source)


def test_path_list_combines_examples_in_order(monkeypatch):
    parts = {"a.jsonl": _dataset("a1", "a2"), "b.jsonl": _dataset("b1")}
    monkeypatch.setattr(Dataset, "from_jsonl", classmethod(lambda c, p: parts[p]))

    combined = _resolve(["a.jsonl", "b.jsonl"])

    assert [e.expected_output for e in combined.examples] == ["a1", "a2", "b1"]
    assert combined.name == "combined_dataset"
    assert combined.description == "Combined dataset from 2 files"


def test_path_list_failure_names_the_failing_path(monkeypatch):
    def _load(_cls, path):
        if path == "bad.jsonl":
            raise ValueError("unreadable")
        return _dataset("ok")

    monkeypatch.setattr(Dataset, "from_jsonl", classmethod(_load))

    with pytest.raises(ConfigurationError, match="bad.jsonl: unreadable"):
        _resolve(["ok.jsonl", "bad.jsonl"])
