"""Per-example agent (target function) call-shape stage.

Reuses the shared ``signature_check`` fixtures to cover positional-only,
keyword-only, defaults, ``*args`` and ``**kwargs`` signatures, plus the
mapping-expand / whole-mapping / scalar decision, an unintrospectable callable,
and parity of ``runtime_*`` with ``BaseEvaluator._prepare_call_arguments``.
"""

from __future__ import annotations

import pytest

from traigent.contract import ContractCode, validate_evaluation_contract
from traigent.evaluators.local import LocalEvaluator

from ._support import error_codes, find_finding, load_signature_fixtures

FIXTURES = load_signature_fixtures()


def _shape(func, input_data, **kwargs):
    report = validate_evaluation_contract(
        func=func, dataset=[{"input": input_data}], **kwargs
    )
    return report, report.call_shapes[0]


# --------------------------------------------------------------------------- #
# Mapping-expand vs whole-mapping vs scalar
# --------------------------------------------------------------------------- #
def test_mapping_expands_to_keywords():
    def agent(question, answer):
        return "x"

    _report, shape = _shape(agent, {"question": "q", "answer": "a"})
    assert shape.expanded is True
    assert shape.runtime_args_count == 0
    assert shape.runtime_keyword_names == ("answer", "question")
    assert shape.bind_ok is True


def test_whole_mapping_passed_positionally():
    # A single pos-or-kw param whose name is NOT a payload key -> whole mapping.
    def agent(payload):
        return "x"

    _report, shape = _shape(agent, {"question": "q"})
    assert shape.expanded is False
    assert shape.runtime_args_count == 1
    assert shape.runtime_keyword_names == ()
    assert shape.bind_ok is True


def test_scalar_passed_positionally():
    def agent(payload):
        return "x"

    _report, shape = _shape(agent, "plain string")
    assert shape.expanded is False
    assert shape.runtime_args_count == 1
    assert shape.bind_ok is True


# --------------------------------------------------------------------------- #
# Reused fixture signatures
# --------------------------------------------------------------------------- #
def test_posonly_only_mapping_cannot_bind():
    # posonly_only(a, b, /): two positional-only params. A mapping expands to
    # keywords, which positional-only params reject -> bind failure.
    report, shape = _shape(FIXTURES["posonly_only"], {"a": 1, "b": 2})
    assert shape.expanded is True
    assert shape.bind_ok is False
    assert ContractCode.AGENT_BIND_FAILED in error_codes(report)


def test_posonly_with_regular_mapping_cannot_bind():
    report, shape = _shape(FIXTURES["posonly_with_regular"], {"a": 1, "b": 2, "c": 3})
    assert shape.bind_ok is False
    assert ContractCode.AGENT_BIND_FAILED in error_codes(report)


def test_kwonly_only_binds():
    _report, shape = _shape(FIXTURES["kwonly_only"], {"a": 1, "b": 2})
    assert shape.expanded is True
    assert shape.runtime_keyword_names == ("a", "b")
    assert shape.bind_ok is True


def test_kwonly_with_defaults_binds_with_only_required():
    # kwonly_with_defaults(*, a, b=10, c=20): supplying only 'a' still binds.
    _report, shape = _shape(FIXTURES["kwonly_with_defaults"], {"a": 1})
    assert shape.runtime_keyword_names == ("a",)
    assert shape.bind_ok is True


def test_full_signature_posonly_prefix_cannot_bind():
    # full_signature has **kwargs (forces expansion) but posonly a, b -> keyword
    # expansion cannot satisfy the positional-only prefix.
    _report, shape = _shape(
        FIXTURES["full_signature"], {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    )
    assert shape.expanded is True
    assert shape.bind_ok is False


def test_var_keyword_agent_binds_any_mapping():
    def agent(**kwargs):
        return "x"

    _report, shape = _shape(agent, {"anything": 1, "here": 2})
    assert shape.expanded is True
    assert shape.bind_ok is True


def test_var_positional_agent_rejects_expanded_mapping():
    # def agent(*args): a mapping expands to keywords, which *args cannot accept.
    def agent(*args):
        return "x"

    _report, shape = _shape(agent, {"a": 1})
    assert shape.expanded is True
    assert shape.bind_ok is False


def test_defaults_agent_binds():
    def agent(question, extra="default"):
        return "x"

    _report, shape = _shape(agent, {"question": "q"})
    assert shape.runtime_keyword_names == ("question",)
    assert shape.bind_ok is True


# --------------------------------------------------------------------------- #
# Bind failure & unintrospectable signature
# --------------------------------------------------------------------------- #
def test_bind_failure_emits_agent_bind_failed():
    def agent(question, answer):
        return "x"

    report, shape = _shape(agent, {"question": "q"})  # missing "answer"
    assert shape.bind_ok is False
    assert report.ok is False
    finding = find_finding(report, ContractCode.AGENT_BIND_FAILED)
    assert finding is not None
    assert finding.severity == "error"
    assert finding.example_index == 0


def test_unintrospectable_signature_warns():
    # NOTE (reality vs matrix): the matrix suggested ``len``, but
    # ``inspect.signature(len)`` succeeds on this interpreter. ``max`` is a
    # builtin whose signature genuinely cannot be introspected.
    report = validate_evaluation_contract(
        func=max, dataset=[{"input": {"a": 1, "b": 2}}]
    )
    finding = find_finding(report, ContractCode.AGENT_SIGNATURE_UNAVAILABLE)
    assert finding is not None
    assert finding.severity == "warning"
    assert report.call_shapes[0].bind_ok is False
    assert report.call_shapes[0].bind_error is None


# --------------------------------------------------------------------------- #
# runtime_* parity with the production evaluator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "input_data",
    [
        {"question": "q", "answer": "a"},  # mapping expand
        {"unmatched": "x"},  # whole mapping (single param name mismatch)
        "scalar",  # scalar positional
    ],
)
def test_runtime_shape_matches_base_evaluator(input_data):
    def agent(payload):
        return "x"

    evaluator = LocalEvaluator()
    args, kwargs = evaluator._prepare_call_arguments(agent, {"model": "m"}, input_data)

    _report, shape = _shape(agent, input_data, config={"model": "m"})
    assert shape.runtime_args_count == len(args)
    assert set(shape.runtime_keyword_names) == set(kwargs)
