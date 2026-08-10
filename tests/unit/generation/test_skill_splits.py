from __future__ import annotations

import logging

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.generation.skill_train.splits import split_dataset


def _dataset(size: int) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"i": i}, expected_output=i)
            for i in range(size)
        ],
        name="base",
        description="d",
    )


def _ids(dataset: Dataset) -> set[int]:
    return {example.input_data["i"] for example in dataset.examples}


def test_split_is_deterministic_by_seed_and_names() -> None:
    first = split_dataset(_dataset(50), 0.2, 0.1, 7)
    second = split_dataset(_dataset(50), 0.2, 0.1, 7)

    assert [_ids(part) if part is not None else set() for part in first] == [
        _ids(part) if part is not None else set() for part in second
    ]
    train, selection, test = first
    assert (len(train.examples), len(selection.examples), len(test.examples)) == (
        35,
        10,
        5,
    )
    assert train.name == "base__train"
    assert selection.name == "base__selection"
    assert test is not None and test.name == "base__test"


def test_split_changes_with_seed() -> None:
    first = split_dataset(_dataset(50), 0.2, 0.1, 7)
    second = split_dataset(_dataset(50), 0.2, 0.1, 8)

    assert _ids(first[1]) != _ids(second[1])


def test_split_has_no_example_overlap() -> None:
    train, selection, test = split_dataset(_dataset(50), 0.2, 0.1, 11)

    assert _ids(train).isdisjoint(_ids(selection))
    assert test is not None
    assert _ids(train).isdisjoint(_ids(test))
    assert _ids(selection).isdisjoint(_ids(test))


def test_selection_minimum_error() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        split_dataset(_dataset(20), 0.2, 0.0, 1)


def test_selection_under_ten_warns(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)

    split_dataset(_dataset(45), 0.2, 0.0, 1)

    assert "only 9 examples" in caplog.text
