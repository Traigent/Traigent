"""Tests for resolving failing example ids back to dataset rows (Traigent#2565).

``select_examples_by_id`` joins outcome-matrix example ids to rows of the
original local ``Dataset`` by RECORDED CONTENT: the run's per-example records
say which ``input_data`` / ``expected_output`` it evaluated under each id, and
the id resolves to the one row with exactly that content. No id scheme
(positional ``example_<i>``, ``metadata["example_id"]``, keyed ``ex1:``, a
custom evaluator's own ids) is ever assumed, so an id can never silently pick
another row. Every failure mode raises ``UnresolvedExampleIdError`` before any
output exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.api.types import ExampleResult
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.regression_dataset import (
    UnresolvedExampleIdError,
    select_examples_by_id,
)

Row = tuple[Any, Any, dict[str, Any] | None]


def _dataset(rows: list[Row]) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data=inp, expected_output=exp, metadata=meta)
            for inp, exp, meta in rows
        ],
        name="regression_source",
    )


def _record(example_id: str, input_data: Any, expected: Any) -> ExampleResult:
    return ExampleResult(
        example_id=example_id,
        input_data=input_data,
        expected_output=expected,
        actual_output="x",
        metrics={"accuracy": 0.0},
        execution_time=0.0,
        success=True,
    )


def _run(*trials: list[Any]) -> SimpleNamespace:
    """A run whose trials carry the given per-example records."""
    return SimpleNamespace(
        trials=[
            SimpleNamespace(metadata={"example_results": records}) for records in trials
        ]
    )


def _run_for(rows: list[Row], ids: list[str]) -> SimpleNamespace:
    """One trial that evaluated ``rows[i]`` under ``ids[i]``."""
    return _run(
        [_record(eid, inp, exp) for eid, (inp, exp, _) in zip(ids, rows, strict=True)]
    )


def _contents(dataset: Dataset) -> list[tuple[Any, Any, dict[str, Any]]]:
    return [(ex.input_data, ex.expected_output, ex.metadata) for ex in dataset]


POSITIONAL: list[Row] = [
    ({"q": "zero"}, "0", None),
    ({"q": "one"}, "1", None),
    ({"q": "two"}, None, None),
    ({"q": "three"}, ["3", "III"], None),
]
POSITIONAL_IDS = [f"example_{i}" for i in range(len(POSITIONAL))]


class TestResolution:
    def test_rows_return_in_source_order_with_their_source_id(self) -> None:
        source = _dataset(POSITIONAL)
        before = _contents(source)

        selected = select_examples_by_id(
            source,
            ["example_3", "example_0"],
            _run_for(POSITIONAL, POSITIONAL_IDS),
        )

        assert isinstance(selected, Dataset)
        assert selected is not source
        assert _contents(selected) == [
            ({"q": "zero"}, "0", {"example_id": "example_0"}),
            ({"q": "three"}, ["3", "III"], {"example_id": "example_3"}),
        ]
        assert _contents(source) == before  # source not mutated

    def test_explicit_null_expected_output_is_kept(self) -> None:
        selected = select_examples_by_id(
            _dataset(POSITIONAL), ["example_2"], _run_for(POSITIONAL, POSITIONAL_IDS)
        )
        assert _contents(selected) == [
            ({"q": "two"}, None, {"example_id": "example_2"})
        ]

    def test_existing_metadata_is_kept_unchanged(self) -> None:
        rows: list[Row] = [
            ({"q": "a"}, "A", {"example_id": "q-a", "tag": "x"}),
            ({"q": "b"}, "B", {"example_id": "q-b", "tag": "y"}),
        ]
        selected = select_examples_by_id(
            _dataset(rows), ["q-b"], _run_for(rows, ["q-a", "q-b"])
        )
        assert _contents(selected) == [
            ({"q": "b"}, "B", {"example_id": "q-b", "tag": "y"})
        ]

    def test_custom_evaluator_one_based_ids_export_the_evaluated_row(self) -> None:
        # Review finding: a custom evaluator emits example_1 for row 0. A
        # positional join would export row 1; the content join exports row 0.
        rows = POSITIONAL[:2]
        run = _run_for(rows, ["example_1", "example_2"])

        selected = select_examples_by_id(_dataset(rows), ["example_1"], run)

        assert [ex.input_data for ex in selected] == [{"q": "zero"}]

    def test_metadata_alias_of_a_keyed_id_does_not_capture_the_join(self) -> None:
        # Review finding: row B carries row A's keyed id as a user annotation.
        keyed = "ex1:k1:digest-a"
        rows: list[Row] = [
            ({"q": "a"}, "A", None),
            ({"q": "b"}, "B", {"example_id": keyed}),
        ]
        run = _run(
            [
                _record(keyed, {"q": "a"}, "A"),
                _record("ex1:k1:digest-b", {"q": "b"}, "B"),
            ]
        )

        selected = select_examples_by_id(_dataset(rows), [keyed], run)

        assert [ex.input_data for ex in selected] == [{"q": "a"}]

    def test_metadata_id_equal_to_another_rows_positional_handle_resolves(self) -> None:
        # Review finding: ids ["example_1", "z"] were falsely ambiguous.
        rows: list[Row] = [
            ({"q": "a"}, "A", {"example_id": "example_1"}),
            ({"q": "b"}, "B", {"example_id": "z"}),
        ]
        selected = select_examples_by_id(
            _dataset(rows), ["example_1"], _run_for(rows, ["example_1", "z"])
        )
        assert [ex.input_data for ex in selected] == [{"q": "a"}]

    def test_mixed_explicit_and_positional_subset_stays_loadable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review finding: compacting rows 1 and 2 used to recompute both keys
        # as example_1 and fail Dataset validation.
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
        rows: list[Row] = [
            ({"q": "a"}, "A", {"example_id": "foo"}),
            ({"q": "b"}, "B", {"example_id": "example_1"}),
            ({"q": "c"}, "C", None),
        ]
        run = _run_for(rows, ["foo", "example_1", "example_2"])

        selected = select_examples_by_id(
            _dataset(rows), ["example_1", "example_2"], run
        )
        out = tmp_path / "subset.jsonl"
        selected.to_jsonl(out)
        reloaded = Dataset.from_jsonl(str(out))

        assert _contents(reloaded) == [
            ({"q": "b"}, "B", {"example_id": "example_1"}),
            ({"q": "c"}, "C", {"example_id": "example_2"}),
        ]

    def test_dict_records_from_a_saved_result_resolve(self) -> None:
        run = _run([_record("example_1", {"q": "one"}, "1").to_dict()])
        selected = select_examples_by_id(_dataset(POSITIONAL), ["example_1"], run)
        assert [ex.input_data for ex in selected] == [{"q": "one"}]

    def test_key_order_inside_the_input_does_not_matter(self) -> None:
        rows: list[Row] = [({"a": 1, "b": 2}, "x", None)]
        run = _run([_record("r", {"b": 2, "a": 1}, "x")])
        selected = select_examples_by_id(_dataset(rows), ["r"], run)
        assert len(selected) == 1

    def test_records_stored_redacted_resolve_to_their_raw_rows(self) -> None:
        # The SDK stores records after redact_sensitive_data: an email or a
        # 9-digit number is a placeholder in the record, raw in the dataset.
        from traigent.security.redaction import redact_sensitive_data

        rows: list[Row] = [
            ({"text": "mail bob@example.com"}, "BOB", None),
            ({"text": "order 123456789"}, "ORDER", None),
            ({"text": "plain"}, "PLAIN", None),
        ]
        records = [
            redact_sensitive_data(_record(f"example_{i}", inp, exp).to_dict())
            for i, (inp, exp, _) in enumerate(rows)
        ]
        assert records[0]["input_data"] == {"text": "mail [REDACTED:email]"}

        selected = select_examples_by_id(
            _dataset(rows), ["example_0", "example_1"], _run(records)
        )

        assert [ex.input_data for ex in selected] == [
            {"text": "mail bob@example.com"},
            {"text": "order 123456789"},
        ]


class TestFailLoudly:
    def test_id_without_any_record_is_unverifiable(self) -> None:
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(
                _dataset(POSITIONAL),
                ["example_9"],
                _run_for(POSITIONAL, POSITIONAL_IDS),
            )
        assert info.value.unverifiable == ["example_9"]

    def test_record_without_input_content_is_unverifiable(self) -> None:
        # Example content not logged: the record is redacted to None.
        record = _record("example_0", {"q": "zero"}, "0").to_dict()
        record["input_data"] = None
        record["expected_output"] = None
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(POSITIONAL), ["example_0"], _run([record]))
        assert info.value.unverifiable == ["example_0"]

    def test_id_naming_different_inputs_across_trials_is_unverifiable(self) -> None:
        # e.g. subset trials re-number positions: example_0 meant two rows.
        run = _run(
            [_record("example_0", {"q": "zero"}, "0")],
            [_record("example_0", {"q": "one"}, "1")],
        )
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(POSITIONAL), ["example_0"], run)
        assert info.value.unverifiable == ["example_0"]

    def test_content_absent_from_the_dataset_is_unresolved(self) -> None:
        run = _run([_record("example_0", {"q": "zero"}, "CHANGED")])
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(POSITIONAL), ["example_0"], run)
        assert info.value.unresolved == ["example_0"]
        assert "example_0" in str(info.value)

    def test_content_matching_two_rows_is_ambiguous(self) -> None:
        rows: list[Row] = [
            ({"q": "same"}, "S", {"example_id": "a"}),
            ({"q": "same"}, "S", {"example_id": "b"}),
        ]
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(rows), ["a"], _run_for(rows, ["a", "b"]))
        assert info.value.ambiguous == ["a"]

    def test_rows_differing_only_inside_a_redacted_span_are_ambiguous(
        self,
    ) -> None:
        # Review finding: a row already holding the placeholder and a row with
        # a real email redact to the same form. Picking either is a guess.
        from traigent.security.redaction import redact_sensitive_data

        rows: list[Row] = [
            ({"text": "mail [REDACTED:email]"}, "X", None),
            ({"text": "mail bob@example.com"}, "X", None),
        ]
        record = redact_sensitive_data(
            _record("example_1", {"text": "mail bob@example.com"}, "X").to_dict()
        )
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(rows), ["example_1"], _run([record]))
        assert info.value.ambiguous == ["example_1"]

    def test_record_without_canonical_json_form_is_unverifiable(self) -> None:
        run = _run([_record("r", {1: "a", "b": 2}, "x")])
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(POSITIONAL), ["r"], run)
        assert info.value.unverifiable == ["r"]

    def test_two_ids_resolving_to_one_row_are_ambiguous(self) -> None:
        run = _run(
            [
                _record("x", {"q": "zero"}, "0"),
                _record("y", {"q": "zero"}, "0"),
            ]
        )
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(POSITIONAL), ["x", "y"], run)
        assert sorted(info.value.ambiguous) == ["x", "y"]

    def test_every_bad_id_is_reported_together(self) -> None:
        run = _run([_record("bad", {"q": "nope"}, "?")])
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(POSITIONAL), ["bad", "missing"], run)
        assert info.value.unresolved == ["bad"]
        assert info.value.unverifiable == ["missing"]

    def test_empty_id_list_raises(self) -> None:
        with pytest.raises(ValueError):
            select_examples_by_id(_dataset(POSITIONAL), [], _run())

    def test_single_string_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            select_examples_by_id(_dataset(POSITIONAL), "example_0", _run())


class TestJsonlSourcedDatasets:
    @pytest.fixture(autouse=True)
    def _dataset_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))

    def test_top_level_example_id_rows_resolve_and_keep_their_id(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"input": {"q": "a"}, "output": "A", "example_id": "row-a"})
            + "\n"
            + json.dumps(
                {
                    "input": {"q": "b"},
                    "output": "B",
                    "metadata": {"example_id": "row-b"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run = _run(
            [_record("row-a", {"q": "a"}, "A"), _record("row-b", {"q": "b"}, "B")]
        )
        selected = select_examples_by_id(
            Dataset.from_jsonl(str(path)), ["row-b", "row-a"], run
        )
        assert _contents(selected) == [
            ({"q": "a"}, "A", {"example_id": "row-a"}),
            ({"q": "b"}, "B", {"example_id": "row-b"}),
        ]
