"""PR #27 parity matrix -- the cross-cutting compatibility cells in one place.

JSONL aliases, whole-mapping vs keyword expansion, positional-only payloads,
grouped/flat option equivalence, the injected ``config_param`` in the call
shape, and the evaluator context parameters.
"""

from __future__ import annotations

import json

import pytest

from traigent.api.decorators import InjectionOptions
from traigent.contract import validate_evaluation_contract

from ._support import finding_codes, load_signature_fixtures

FIXTURES = load_signature_fixtures()


def _agent(question):
    return "x"


# --------------------------------------------------------------------------- #
# JSONL aliases (input/input_data + expected-output aliases) via from_jsonl
# --------------------------------------------------------------------------- #
def test_jsonl_input_and_expected_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    rows = [
        {"input": {"question": "q1"}, "output": "a1"},
        {"input_data": {"question": "q2"}, "expected": "a2"},
        {"input": {"question": "q3"}, "answer": "a3", "difficulty": "hard"},
    ]
    path = tmp_path / "aliases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = validate_evaluation_contract(func=_agent, dataset=str(path))
    assert report.ok
    assert len(report.examples) == 3
    assert all(ex.input_keys == ("question",) for ex in report.examples)
    assert all(ex.has_expected_output for ex in report.examples)
    # Non-consumed keys survive as metadata.
    assert report.examples[2].metadata_keys == ("difficulty",)


# --------------------------------------------------------------------------- #
# whole-mapping vs keyword expansion
# --------------------------------------------------------------------------- #
def test_keyword_expansion_vs_whole_mapping():
    def multi(question, answer):
        return "x"

    def single(payload):
        return "x"

    expanded = validate_evaluation_contract(
        func=multi, dataset=[{"input": {"question": "q", "answer": "a"}}]
    ).call_shapes[0]
    whole = validate_evaluation_contract(
        func=single, dataset=[{"input": {"question": "q"}}]
    ).call_shapes[0]

    assert expanded.expanded is True
    assert expanded.runtime_args_count == 0
    assert expanded.runtime_keyword_names == ("answer", "question")

    assert whole.expanded is False
    assert whole.runtime_args_count == 1
    assert whole.runtime_keyword_names == ()


# --------------------------------------------------------------------------- #
# positional-only payloads
# --------------------------------------------------------------------------- #
def test_single_positional_only_takes_whole_mapping():
    def agent(payload, /):
        return "x"

    shape = validate_evaluation_contract(
        func=agent, dataset=[{"input": {"question": "q"}}]
    ).call_shapes[0]
    # A single positional-only param receives the whole mapping positionally.
    assert shape.expanded is False
    assert shape.runtime_args_count == 1
    assert shape.bind_ok is True


def test_multi_positional_only_rejects_expanded_mapping():
    # posonly_only(a, b, /) from the shared fixture: expansion to keywords is
    # rejected by positional-only params.
    shape = validate_evaluation_contract(
        func=FIXTURES["posonly_only"], dataset=[{"input": {"a": 1, "b": 2}}]
    ).call_shapes[0]
    assert shape.expanded is True
    assert shape.bind_ok is False


# --------------------------------------------------------------------------- #
# grouped/flat equivalence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("config_param", ["config", "settings"])
def test_grouped_flat_equivalence(config_param):
    def agent(question, **kw):
        return "x"

    dataset = [{"input": {"question": "q"}}]
    flat = validate_evaluation_contract(
        func=agent,
        dataset=dataset,
        injection_mode="parameter",
        config_param=config_param,
        config={"model": "m"},
    )
    grouped = validate_evaluation_contract(
        func=agent,
        dataset=dataset,
        injection_options=InjectionOptions(
            injection_mode="parameter", config_param=config_param
        ),
        config={"model": "m"},
    )
    assert (
        flat.injection.effective_mode == grouped.injection.effective_mode == "parameter"
    )
    assert flat.injection.config_param == grouped.injection.config_param == config_param
    assert (
        flat.call_shapes[0].effective_keyword_names
        == grouped.call_shapes[0].effective_keyword_names
    )


# --------------------------------------------------------------------------- #
# injected config_param appears in the effective call shape
# --------------------------------------------------------------------------- #
def test_injected_config_param_in_call_shape():
    def agent(question, config):
        return "x"

    shape = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q"}}],
        injection_mode="parameter",
        config={"model": "m"},
    ).call_shapes[0]

    assert "config" not in shape.runtime_keyword_names
    assert "config" in shape.effective_keyword_names
    assert shape.effective_keyword_names == ("config", "question")
    assert shape.bind_ok is True


# --------------------------------------------------------------------------- #
# evaluator context parameters
# --------------------------------------------------------------------------- #
def test_evaluator_context_parameters():
    def metric(example, input_data, metadata, config, example_index):
        return 1.0

    report = validate_evaluation_contract(
        func=_agent,
        dataset=[{"input": {"question": "q"}, "output": "gold"}],
        scoring_function=metric,
    )
    binding = report.evaluator_bindings[0]
    assert binding.binding_mode == "keyword"
    assert binding.matched_parameters == (
        "config",
        "example",
        "example_index",
        "input_data",
        "metadata",
    )
    assert binding.bind_ok is True
    assert finding_codes(report) == []
