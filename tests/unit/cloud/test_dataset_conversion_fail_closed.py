"""#1722 (g6:dataset-row-drop) — a dropped dataset row must not read as success.

`DatasetConverter.sdk_dataset_to_backend_examples` caught every per-row
conversion failure, logged one warning, and incremented a LOCAL `error_count`
that it then discarded: the return type is `(examples, metadata)`, so no caller
could ever see it. Worse, `ExampleSetMetadata.total_examples` was set to
`len(examples)` — the POST-drop count — so a dataset that lost half its rows
uploaded as a complete one, and `upload_sdk_dataset_to_backend` returned
`success=True, errors=[]`.

That is the same class as the metric-exception half of #1722, which now fails
closed for objective metrics: a silently shorter dataset changes what the
optimization is measured against while every count the caller can see still says
it worked.

The sibling `metric_errors` accumulator in `evaluators/local.py` is the model for
the non-strict path here.
"""

from __future__ import annotations

import pytest

from traigent.cloud.dataset_converter import (
    MAX_REPORTED_CONVERSION_ERRORS,
    DatasetConverter,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ValidationError


class _UnconvertibleExample(EvaluationExample):
    """An example whose conversion raises, standing in for a malformed row."""


def _dataset(*examples: EvaluationExample, name: str = "ds") -> Dataset:
    return Dataset(name=name, examples=list(examples))


def _good(value: str = "hello") -> EvaluationExample:
    return EvaluationExample(input_data={"text": value}, expected_output="ok")


@pytest.fixture
def converter() -> DatasetConverter:
    return DatasetConverter(backend_base_url="https://example.invalid")


def _break_one(monkeypatch, converter: DatasetConverter, failing_indexes: set[int]):
    """Make conversion raise for the given example indexes."""
    original = converter._convert_evaluation_example_to_backend

    def _maybe_fail(example, index, privacy_mode):
        if index in failing_indexes:
            raise ValueError(f"synthetic failure for row {index}")
        return original(example, index, privacy_mode)

    monkeypatch.setattr(
        converter, "_convert_evaluation_example_to_backend", _maybe_fail
    )


class TestStrictIsTheDefault:
    def test_a_single_bad_row_raises_rather_than_shrinking_the_dataset(
        self, monkeypatch, converter
    ):
        dataset = _dataset(_good("a"), _good("b"), _good("c"))
        _break_one(monkeypatch, converter, {1})

        with pytest.raises(ValidationError) as excinfo:
            converter.sdk_dataset_to_backend_examples(dataset)

        message = str(excinfo.value)
        assert "1 of 3" in message, "the error must state the scale of the loss"
        assert "example 1" in message, "and name the row, not just the count"

    def test_the_error_names_the_underlying_cause(self, monkeypatch, converter):
        """A row index alone does not tell anyone WHY it failed."""
        dataset = _dataset(_good("a"), _good("b"))
        _break_one(monkeypatch, converter, {0})

        with pytest.raises(ValidationError) as excinfo:
            converter.sdk_dataset_to_backend_examples(dataset)

        assert "ValueError" in str(excinfo.value)
        assert "synthetic failure for row 0" in str(excinfo.value)

    def test_a_fully_valid_dataset_is_unaffected(self, converter):
        """The behaviour that must NOT change: strict is not stricter about
        anything except rows that were previously dropped."""
        dataset = _dataset(_good("a"), _good("b"), _good("c"))

        examples, metadata = converter.sdk_dataset_to_backend_examples(dataset)

        assert len(examples) == 3
        assert metadata.total_examples == 3

    def test_an_empty_dataset_still_converts_rather_than_raising(self, converter):
        """No rows failed, so there is nothing to fail closed over. An empty
        dataset is a different complaint and not this function's to make."""
        examples, metadata = converter.sdk_dataset_to_backend_examples(_dataset())

        assert examples == []
        assert metadata.total_examples == 0

    def test_the_error_is_capped_for_a_wholly_malformed_dataset(
        self, monkeypatch, converter
    ):
        count = MAX_REPORTED_CONVERSION_ERRORS + 5
        dataset = _dataset(*[_good(str(i)) for i in range(count)])
        _break_one(monkeypatch, converter, set(range(count)))

        with pytest.raises(ValidationError) as excinfo:
            converter.sdk_dataset_to_backend_examples(dataset)

        message = str(excinfo.value)
        assert f"{count} of {count}" in message
        assert (
            "and 5 more" in message
        ), "the remainder must be acknowledged, not dropped"
        # The cap is real: the last row must not be spelled out.
        assert f"example {count - 1}:" not in message


class TestNonStrictSurfacesWhatItDropped:
    def test_opting_out_still_reports_every_dropped_row(self, monkeypatch, converter):
        dataset = _dataset(_good("a"), _good("b"), _good("c"))
        _break_one(monkeypatch, converter, {0, 2})
        errors: list[dict] = []

        examples, _ = converter.sdk_dataset_to_backend_examples(
            dataset, strict=False, conversion_errors=errors
        )

        assert len(examples) == 1, "the drop is accepted when explicitly asked for"
        assert [e["example_index"] for e in errors] == [0, 2]
        assert all(e["error_type"] == "ValueError" for e in errors)
        assert all("synthetic failure" in e["error_message"] for e in errors)

    def test_non_strict_without_an_accumulator_does_not_raise(
        self, monkeypatch, converter
    ):
        """Explicitly opting out and declining the records is allowed — it is a
        deliberate choice, not the silent default it used to be."""
        dataset = _dataset(_good("a"), _good("b"))
        _break_one(monkeypatch, converter, {0})

        examples, _ = converter.sdk_dataset_to_backend_examples(dataset, strict=False)

        assert len(examples) == 1

    def test_the_accumulator_is_appended_to_not_replaced(self, monkeypatch, converter):
        """Mirrors metric_errors: a caller may reuse one list across datasets."""
        dataset = _dataset(_good("a"))
        _break_one(monkeypatch, converter, {0})
        errors: list[dict] = [{"example_index": 99, "pre": "existing"}]

        converter.sdk_dataset_to_backend_examples(
            dataset, strict=False, conversion_errors=errors
        )

        assert len(errors) == 2
        assert errors[0]["pre"] == "existing"


class TestMetadataNoLongerOverstatesCompleteness:
    def test_total_examples_cannot_silently_describe_a_shortened_dataset(
        self, monkeypatch, converter
    ):
        """`total_examples=len(examples)` was the post-drop count, so the upload
        looked complete. Under strict that state is now unreachable — the only
        way to get metadata back is for every row to have converted."""
        dataset = _dataset(_good("a"), _good("b"), _good("c"))
        _break_one(monkeypatch, converter, {1})

        with pytest.raises(ValidationError):
            converter.sdk_dataset_to_backend_examples(dataset)

    def test_total_examples_agrees_with_the_input_when_nothing_failed(self, converter):
        # A separate converter: the monkeypatch above keys on the row INDEX, so
        # reusing the patched one here would fail row 1 again and this would be
        # asserting the wrong thing.
        dataset = _dataset(_good("a"), _good("b"), _good("c"))

        examples, metadata = converter.sdk_dataset_to_backend_examples(dataset)

        assert metadata.total_examples == len(examples) == len(dataset.examples) == 3
