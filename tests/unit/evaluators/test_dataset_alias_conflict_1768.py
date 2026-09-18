"""Regression tests for issue #1768 — dataset expected-output alias conflicts
and metadata-field nesting in ``_coerce_dataset_example_mapping``.

Before the fix, a dataset row carrying MORE THAN ONE expected-output alias
(``output``, ``expected``, ``expected_output``, ``answer``, ``target``,
``label``) resolved silently by first-alias-wins order, with zero warning,
AND the losing alias was not discarded: it leaked into ``example.metadata``
and was delivered to every metric function that declares a ``metadata``
parameter, masquerading as real user metadata. Separately, an explicit
``metadata`` row field was not special-cased -- it nested one level deep
(``example.metadata == {"metadata": {...}}``), silently breaking
``example_id``-based correlation for any row shaped like the
``EvaluationExample`` dataclass itself.

These tests assert the decided behavior:

* multiple aliases -> the precedence-order winner is used, EVERY losing alias
  is dropped (never present anywhere in ``example.metadata``), and exactly
  one ``logger.warning`` fires naming the winner and the losers;
* a single alias -> unchanged behavior, no warning (regression safety);
* a dict-valued ``metadata`` row field is merged at the TOP LEVEL of
  ``example.metadata``, not nested under ``metadata["metadata"]``;
* an ``example_id`` inside that row-level ``metadata`` dict is now found by
  ``_example_correlation_key`` instead of silently falling back to a
  positional ``example_N`` key;
* a non-dict ``metadata`` value degrades to an ordinary extra field instead
  of crashing or being silently dropped;
* the fix reaches the public ``load_inline_dataset`` entry point, not just
  the internal coercion helper.

Each behavioral assertion is written so it FAILS on the old behavior (silent
first-alias-wins with a metadata leak, nested ``metadata["metadata"]``) and
PASSES on the new.
"""

from __future__ import annotations

import logging

import pytest

from traigent.evaluators.base import (
    _EXPECTED_OUTPUT_FIELDS,
    _coerce_dataset_example_mapping,
    _example_correlation_key,
    load_inline_dataset,
)

_LOGGER_NAME = "traigent.evaluators.base"
_WARNING_MARKER = "multiple expected-output aliases"


def _warning_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _WARNING_MARKER in r.getMessage()]


def test_alias_precedence_order_is_documented_and_stable() -> None:
    # Locks the precedence order the warning message and the docstring rely on.
    assert _EXPECTED_OUTPUT_FIELDS == (
        "output",
        "expected",
        "expected_output",
        "answer",
        "target",
        "label",
    )


# --------------------------------------------------------------------------- #
# Multiple aliases: winner used, losers dropped (never leaked), one warning     #
# --------------------------------------------------------------------------- #
def test_dual_alias_winner_and_loser_dropped_not_leaked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    item = {"input": "q", "output": "A", "expected_output": "B"}

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        input_data, expected_output, metadata = _coerce_dataset_example_mapping(
            item, source="test-source", location="example 0"
        )

    # "output" outranks "expected_output" in _EXPECTED_OUTPUT_FIELDS order.
    assert expected_output == "A"
    # The losing alias must be ABSENT everywhere in metadata -- not just
    # excluded as the "winner" key, but genuinely dropped.
    assert "expected_output" not in metadata
    assert metadata == {}

    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "'output'" in message
    assert "expected_output" in message


def test_three_alias_conflict_names_all_losers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    item = {"input": "q", "answer": "gold-answer", "label": "gold-label", "target": "t"}

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _, expected_output, metadata = _coerce_dataset_example_mapping(
            item, source="test-source", location="example 0"
        )

    # "answer" outranks "target" and "label".
    assert expected_output == "gold-answer"
    assert "label" not in metadata
    assert "target" not in metadata
    assert metadata == {}
    assert len(_warning_records(caplog)) == 1


# --------------------------------------------------------------------------- #
# Single alias: unchanged, no warning (regression safety)                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alias", _EXPECTED_OUTPUT_FIELDS)
def test_single_alias_no_warning(alias: str, caplog: pytest.LogCaptureFixture) -> None:
    item = {"input": "q", alias: "gold", "difficulty": "hard"}

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _, expected_output, metadata = _coerce_dataset_example_mapping(
            item, source="test-source", location="example 0"
        )

    assert expected_output == "gold"
    assert alias not in metadata
    assert metadata == {"difficulty": "hard"}
    assert _warning_records(caplog) == []


def test_no_alias_present_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    item = {"input": "q", "difficulty": "hard"}

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _, expected_output, metadata = _coerce_dataset_example_mapping(
            item, source="test-source", location="example 0"
        )

    assert expected_output is None
    assert metadata == {"difficulty": "hard"}
    assert _warning_records(caplog) == []


# --------------------------------------------------------------------------- #
# Explicit dict-valued "metadata" field: merged at top level, not nested        #
# --------------------------------------------------------------------------- #
def test_explicit_metadata_dict_merges_at_top_level() -> None:
    item = {
        "input": "q",
        "expected_output": "y",
        "metadata": {"a": 1, "difficulty": "easy"},
    }

    _, _, metadata = _coerce_dataset_example_mapping(
        item, source="test-source", location="example 0"
    )

    # NOT nested under metadata["metadata"] -- merged at the top level.
    assert "metadata" not in metadata
    assert metadata == {"a": 1, "difficulty": "easy"}


def test_explicit_metadata_example_id_restores_correlation() -> None:
    """The exact scenario from the issue: an EvaluationExample-shaped row's
    example_id inside "metadata" must be found by the correlation key, not
    fall back to a positional example_N key."""
    item = {
        "input_data": {"q": "1"},
        "expected_output": "y",
        "metadata": {"example_id": "custom-key"},
    }

    dataset = load_inline_dataset([item])
    example = dataset.examples[0]

    assert example.metadata.get("example_id") == "custom-key"
    assert _example_correlation_key(example, index=0) == "custom-key"


def test_explicit_metadata_collides_with_extra_field_metadata_wins() -> None:
    """On a name collision between the row's explicit "metadata" dict and an
    ordinary extra field, the deliberate metadata dict wins."""
    item = {
        "input": "q",
        "output": "gold",
        "source": "row-level-extra",  # would be an extra key
        "metadata": {"source": "explicit-metadata"},
    }

    _, _, metadata = _coerce_dataset_example_mapping(
        item, source="test-source", location="example 0"
    )

    assert metadata["source"] == "explicit-metadata"


def test_non_dict_metadata_value_kept_as_extra_field() -> None:
    """A non-dict "metadata" value cannot be merged as the metadata dict, so
    it degrades to an ordinary extra field instead of crashing or vanishing."""
    item = {"input": "q", "output": "gold", "metadata": "not-a-dict"}

    _, _, metadata = _coerce_dataset_example_mapping(
        item, source="test-source", location="example 0"
    )

    assert metadata == {"metadata": "not-a-dict"}


# --------------------------------------------------------------------------- #
# End-to-end via the public inline-dataset loader                               #
# --------------------------------------------------------------------------- #
def test_load_inline_dataset_drops_losing_alias(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        dataset = load_inline_dataset(
            [{"input": "q", "output": "A", "expected_output": "B"}]
        )

    example = dataset.examples[0]
    assert example.expected_output == "A"
    assert "expected_output" not in example.metadata
    assert len(_warning_records(caplog)) == 1


class TestTheJsonlLoaderGetsTheSameTreatment:
    """`Dataset.from_jsonl` was untested, and it is the path real datasets take.

    Every case above goes through `load_inline_dataset` or the coercion helper
    directly. A second reviewer predicted that rewrapping the JSONL loader's
    result -- `metadata=metadata` -> `metadata={"metadata": metadata}` at
    `_parse_jsonl_examples` -- would restore the nesting bug for every file-
    backed dataset and leave this module green. It would have.
    """

    def test_a_jsonl_row_metadata_dict_merges_at_the_top_level(
        self, tmp_path, monkeypatch
    ) -> None:
        import json as _json

        from traigent.evaluators.base import Dataset

        # `_resolve_dataset_source` refuses a path outside the dataset root.
        # That guard is doing its job; point it at tmp_path rather than
        # around it.
        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
        path = tmp_path / "rows.jsonl"
        path.write_text(
            "\n".join(
                _json.dumps(row)
                for row in (
                    {
                        "input": "q1",
                        "expected_output": "A",
                        "metadata": {"example_id": "row-7", "split": "dev"},
                    },
                    {"input": "q2", "expected_output": "B", "difficulty": "hard"},
                )
            ),
            encoding="utf-8",
        )

        dataset = Dataset.from_jsonl(str(path))

        assert dataset.examples[0].metadata == {"example_id": "row-7", "split": "dev"}
        assert _example_correlation_key(dataset.examples[0], 0) == "row-7", (
            "a file-backed row's example_id must reach the correlation key, "
            "not fall back to a positional example_N"
        )
        assert dataset.examples[1].metadata == {"difficulty": "hard"}
        assert _example_correlation_key(dataset.examples[1], 1) == "example_1"

    def test_a_jsonl_row_drops_its_losing_alias(
        self, tmp_path, caplog, monkeypatch
    ) -> None:
        import json as _json

        from traigent.evaluators.base import Dataset

        monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
        path = tmp_path / "dual.jsonl"
        path.write_text(
            _json.dumps({"input": "q", "output": "A", "answer": "B"}), encoding="utf-8"
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            dataset = Dataset.from_jsonl(str(path))

        assert dataset.examples[0].expected_output == "A"
        assert dataset.examples[0].metadata == {}
        assert _WARNING_MARKER in caplog.text


class TestAnExplicitNullMetadataFieldIsNotAbsence:
    """`item.get("metadata")` cannot tell `"metadata": null` from no field.

    The first version keyed the non-dict branch on `row_metadata is not None`,
    so an explicitly-null field was dropped -- contradicting the same
    function's promise to keep a non-dict value rather than silently discard
    it. Measured against develop, which reported `{"metadata": None}`.
    """

    def test_an_explicit_null_is_kept(self) -> None:
        _, _, metadata = _coerce_dataset_example_mapping(
            {"input": "q", "output": "A", "metadata": None},
            source="s",
            location="row",
        )
        assert metadata == {"metadata": None}

    def test_an_absent_field_stays_absent(self) -> None:
        """Control: presence, not truthiness -- a missing field adds nothing."""
        _, _, metadata = _coerce_dataset_example_mapping(
            {"input": "q", "output": "A"}, source="s", location="row"
        )
        assert metadata == {}
