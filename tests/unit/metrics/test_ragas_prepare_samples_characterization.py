"""Characterization tests pinning ragas_metrics._prepare_samples."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.metrics import ragas_metrics as rm
from traigent.metrics.ragas_metrics import RagasConfig, _prepare_samples


@dataclass
class _Sample:
    user_input: Any = None
    response: Any = None
    reference: Any = None
    retrieved_contexts: Any = None
    reference_contexts: Any = None


@pytest.fixture(autouse=True)
def _fake_sample(monkeypatch):
    monkeypatch.setattr(rm, "SingleTurnSample", _Sample)


def _res(
    actual="out", expected=None, input_data=None, metadata=None
) -> SimpleNamespace:
    return SimpleNamespace(
        actual_output=actual,
        expected_output=expected,
        input_data=input_data if input_data is not None else {},
        metadata=metadata,
    )


def _prep(results, required=frozenset(), config=None, **kw):
    return _prepare_samples(
        results,
        config=config or RagasConfig(),
        required_columns=set(required),
        **kw,
    )


def test_empty_results():
    assert _prep([]) == []


def test_basic_sample_fields():
    out = _prep(
        [
            _res(
                actual="resp",
                expected="gold",
                input_data={"question": "q?"},
                metadata={"contexts": ["c1", "c2"], "ground_truth_contexts": "g"},
            )
        ]
    )
    assert out == [
        _Sample(
            user_input="q?",
            response="resp",
            reference="gold",
            retrieved_contexts=["c1", "c2"],
            reference_contexts=["g"],
        )
    ]


def test_response_none_row_dropped_unless_outputs_supply_it():
    rows = [_res(actual=None, expected="g"), _res(actual=None, expected="g")]
    assert _prep(rows) == []
    out = _prep(rows, outputs=["only-first"])
    assert [s.response for s in out] == ["only-first"]


def test_actual_output_wins_over_outputs_and_is_stringified():
    out = _prep([_res(actual=0)], outputs=["other"])
    assert out[0].response == "0"


def test_user_input_fallbacks():
    # no matching key -> str(input_data); empty dict input_data -> "{}"
    out = _prep([_res(input_data={"zzz": 1})])
    assert out[0].user_input == "{'zzz': 1}"
    out = _prep([_res(input_data="plain text", expected="g")])
    assert out[0].user_input == "plain text"


def test_input_data_non_mapping_not_used_as_source():
    out = _prep([_res(input_data=["x"], expected="gold")])
    assert out[0].user_input == "['x']"


def test_dataset_example_input_and_metadata_sources_used():
    ex = SimpleNamespace(
        input_data={"query": "from-dataset"}, metadata={"context": ["dc"]}
    )
    out = _prep([_res(input_data={})], dataset_examples=[ex])
    assert out[0].user_input == "from-dataset"
    assert out[0].retrieved_contexts == ["dc"]


def test_result_metadata_precedes_dataset_metadata():
    ex = SimpleNamespace(input_data=None, metadata={"contexts": ["ds"]})
    out = _prep([_res(metadata={"contexts": ["res"]})], dataset_examples=[ex])
    assert out[0].retrieved_contexts == ["res"]


def test_dataset_examples_shorter_than_results():
    ex = SimpleNamespace(input_data={"query": "q0"}, metadata=None)
    out = _prep([_res(), _res()], dataset_examples=[ex])
    assert out[0].user_input == "q0"
    assert out[1].user_input == "{}"


def test_reference_column_map_wins_over_expected():
    cfg = RagasConfig(column_map={"reference": "gold_col"})
    out = _prep(
        [
            _res(
                expected="fallback",
                input_data={"gold_col": "mapped"},
                metadata={"gold_col": "meta-mapped"},
            )
        ],
        config=cfg,
    )
    # metadata sources are consulted before input sources
    assert out[0].reference == "meta-mapped"
    out = _prep([_res(expected="fallback", input_data={"other": 1})], config=cfg)
    assert out[0].reference == "fallback"


def test_reference_not_looked_up_without_column_map_entry():
    out = _prep([_res(expected=None, metadata={"reference": "ignored"})])
    assert out[0].reference is None


def test_user_input_falls_back_to_reference_never_when_input_data_present():
    # hasattr(result, "input_data") is always true for ExampleResult, so
    # str(input_data) is used before the reference fallback.
    out = _prep([_res(expected="gold", input_data={})])
    assert out[0].user_input == "{}"


def test_none_input_data_is_stringified():
    r = SimpleNamespace(
        actual_output="a", expected_output=None, metadata=None, input_data=None
    )
    out = _prep([r])
    assert out[0].user_input == "None"


def test_required_columns_drop_rows():
    rows = [
        _res(expected=None),
        _res(expected="gold"),
        _res(expected="gold", metadata={"contexts": []}),
        _res(expected="gold", metadata={"contexts": ["c"]}),
    ]
    out = _prep(rows, required={"reference"})
    assert len(out) == 3
    out = _prep(rows, required={"reference", "retrieved_contexts"})
    assert len(out) == 1 and out[0].retrieved_contexts == ["c"]
    out = _prep(rows, required={"reference_contexts"})
    assert out == []


def test_required_column_unknown_name_drops_row():
    assert _prep([_res(expected="g")], required={"nonexistent"}) == []


def test_custom_column_map_for_contexts_and_input():
    cfg = RagasConfig(
        column_map={
            "user_input": "q",
            "retrieved_contexts": "docs",
            "reference_contexts": "refdocs",
        }
    )
    out = _prep(
        [_res(input_data={"q": "Q"}, metadata={"docs": "d", "refdocs": ["r"]})],
        config=cfg,
    )
    assert out[0].user_input == "Q"
    assert out[0].retrieved_contexts == ["d"]
    assert out[0].reference_contexts == ["r"]


def test_empty_metadata_falsy_ignored():
    out = _prep([_res(metadata={})])
    assert out[0].retrieved_contexts is None
