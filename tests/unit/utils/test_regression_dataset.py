"""Tests for resolving failing example ids back to dataset rows (Traigent#2565).

``select_examples_by_id`` joins outcome-matrix example ids to the rows of the
original local ``Dataset`` using the same correlation keys the evaluators emit:

- the local evaluator's ``_example_correlation_key`` (``metadata["example_id"]``
  when present, else positional ``example_<i>``);
- the simple-scoring evaluator's positional ``example_<i>`` handle.

Candidates are normalized exactly like the matrix (``str()``), and every id
must resolve to exactly one row; otherwise ``UnresolvedExampleIdError`` lists
every unresolved / ambiguous id. Keyed ``ex1:`` content identities are not
supported by the local exporter and are reported as unresolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.regression_dataset import (
    UnresolvedExampleIdError,
    select_examples_by_id,
)


def _dataset(rows: list[tuple[Any, Any, dict[str, Any] | None]]) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data=inp, expected_output=exp, metadata=meta)
            for inp, exp, meta in rows
        ],
        name="regression_source",
    )


def _contents(dataset: Dataset) -> list[tuple[Any, Any, dict[str, Any]]]:
    return [(ex.input_data, ex.expected_output, ex.metadata) for ex in dataset]


POSITIONAL = [
    ({"q": "zero"}, "0", None),
    ({"q": "one"}, "1", None),
    ({"q": "two"}, None, None),
    ({"q": "three"}, ["3", "III"], None),
]


class TestResolution:
    def test_positional_ids_return_rows_in_original_order(self) -> None:
        source = _dataset(POSITIONAL)
        before = _contents(source)

        selected = select_examples_by_id(source, ["example_3", "example_0"])

        assert isinstance(selected, Dataset)
        assert selected is not source
        assert _contents(selected) == [
            ({"q": "zero"}, "0", {}),
            ({"q": "three"}, ["3", "III"], {}),
        ]
        # The source dataset is not mutated.
        assert _contents(source) == before

    def test_explicit_null_expected_output_is_kept(self) -> None:
        selected = select_examples_by_id(_dataset(POSITIONAL), ["example_2"])
        assert _contents(selected) == [({"q": "two"}, None, {})]

    def test_metadata_example_ids_resolve(self) -> None:
        source = _dataset(
            [
                ({"q": "a"}, "A", {"example_id": "q-a", "tag": "x"}),
                ({"q": "b"}, "B", {"example_id": "q-b", "tag": "y"}),
            ]
        )
        selected = select_examples_by_id(source, ["q-b"])
        assert _contents(selected) == [
            ({"q": "b"}, "B", {"example_id": "q-b", "tag": "y"})
        ]

    def test_simple_scoring_positional_handle_resolves_metadata_id_rows(
        self,
    ) -> None:
        # SimpleScoringEvaluator always emits example_<i>, even when the row
        # carries a metadata example_id.
        source = _dataset(
            [
                ({"q": "a"}, "A", {"example_id": "q-a"}),
                ({"q": "b"}, "B", {"example_id": "q-b"}),
            ]
        )
        selected = select_examples_by_id(source, ["example_1"])
        assert _contents(selected) == [({"q": "b"}, "B", {"example_id": "q-b"})]

    def test_non_string_metadata_ids_are_matched_by_their_string_form(self) -> None:
        # build_outcome_matrix stringifies ids, so a metadata id 7 shows up as "7".
        source = _dataset(
            [
                ({"q": "a"}, "A", {"example_id": 7}),
                ({"q": "b"}, "B", {"example_id": 8}),
            ]
        )
        selected = select_examples_by_id(source, ["8"])
        assert _contents(selected) == [({"q": "b"}, "B", {"example_id": 8})]


class TestFailLoudly:
    def test_unresolved_ids_are_all_reported(self) -> None:
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(
                _dataset(POSITIONAL), ["example_1", "nope", "example_99"]
            )
        err = info.value
        assert isinstance(err, ValueError)
        assert sorted(err.unresolved) == ["example_99", "nope"]
        assert err.ambiguous == []
        assert "nope" in str(err)
        assert "example_99" in str(err)

    def test_id_matching_two_rows_through_different_schemes_is_ambiguous(
        self,
    ) -> None:
        # Row 0's metadata id is "example_1"; row 1's positional handle is also
        # "example_1". The dataset itself is valid (correlation keys
        # "example_1" and "z" are unique), but the join cannot pick a row.
        source = _dataset(
            [
                ({"q": "a"}, "A", {"example_id": "example_1"}),
                ({"q": "b"}, "B", {"example_id": "z"}),
            ]
        )
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(source, ["example_1"])
        assert info.value.ambiguous == ["example_1"]
        assert info.value.unresolved == []
        assert "example_1" in str(info.value)

    def test_stringification_collision_is_ambiguous(self) -> None:
        # 7 and "7" are distinct dataset keys but both appear as "7" in the
        # matrix, so the id cannot be attributed to one row.
        source = _dataset(
            [
                ({"q": "a"}, "A", {"example_id": 7}),
                ({"q": "b"}, "B", {"example_id": "7"}),
            ]
        )
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(source, ["7"])
        assert info.value.ambiguous == ["7"]

    def test_keyed_content_identity_is_reported_as_unsupported(self) -> None:
        keyed = "ex1:0123456789abcdef"
        with pytest.raises(UnresolvedExampleIdError) as info:
            select_examples_by_id(_dataset(POSITIONAL), ["example_0", keyed])
        assert info.value.unresolved == [keyed]
        message = str(info.value)
        assert keyed in message
        assert "keyed" in message.lower()

    def test_empty_id_list_raises(self) -> None:
        with pytest.raises(ValueError):
            select_examples_by_id(_dataset(POSITIONAL), [])


class TestJsonlSourcedDatasets:
    """Positional ids index *parsed* examples, and top-level ids become metadata."""

    @pytest.fixture(autouse=True)
    def _dataset_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))

    def test_blank_lines_do_not_shift_positional_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"input": {"q": "first"}, "output": "1"})
            + "\n\n   \n"
            + json.dumps({"input": {"q": "second"}, "output": "2"})
            + "\n",
            encoding="utf-8",
        )
        selected = select_examples_by_id(Dataset.from_jsonl(str(path)), ["example_1"])
        assert _contents(selected) == [({"q": "second"}, "2", {})]

    def test_top_level_example_id_is_a_correlation_key(self, tmp_path: Path) -> None:
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
        selected = select_examples_by_id(
            Dataset.from_jsonl(str(path)), ["row-b", "row-a"]
        )
        assert _contents(selected) == [
            ({"q": "a"}, "A", {"example_id": "row-a"}),
            ({"q": "b"}, "B", {"example_id": "row-b"}),
        ]
