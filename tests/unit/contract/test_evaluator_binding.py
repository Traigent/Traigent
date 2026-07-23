"""Evaluator / metric-binding stage of the evaluation contract.

Covers the recognized-parameter alias families, the ``*args`` fast path,
positional fallbacks of decreasing arity, the no-recognized-params hazard,
over-arity bind failure, ``scoring_function`` placement via
``build_metric_functions``, and the custom-evaluator "unsupported" path.
"""

from __future__ import annotations

import warnings

import pytest

import traigent
from traigent.contract import ContractCode, validate_evaluation_contract

from ._support import find_finding, finding_codes

DATASET = [{"input": {"question": "q1"}, "output": "gold"}]


def _agent(question):
    return "x"


def _binding_for(
    scoring_function=None, metric_functions=None, objectives=("accuracy",)
):
    report = validate_evaluation_contract(
        func=_agent,
        dataset=DATASET,
        scoring_function=scoring_function,
        metric_functions=metric_functions,
        objectives=objectives,
    )
    return report, report.evaluator_bindings[0]


# --------------------------------------------------------------------------- #
# Alias families -> keyword binding
# --------------------------------------------------------------------------- #
def test_output_expected_aliases():
    def metric(output, expected):
        return 1.0

    report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "keyword"
    assert binding.matched_parameters == ("expected", "output")
    assert binding.bind_ok is True
    assert finding_codes(report) == []


def test_actual_ground_truth_aliases():
    def metric(actual, ground_truth):
        return 1.0

    _report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "keyword"
    assert binding.matched_parameters == ("actual", "ground_truth")
    assert binding.bind_ok is True


def test_llm_metrics_alias():
    def metric(output, llm_metrics):
        return 1.0

    _report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "keyword"
    assert binding.matched_parameters == ("llm_metrics", "output")
    assert binding.bind_ok is True


def test_context_param_aliases():
    def metric(output, example, input_data, metadata, config, example_index):
        return 1.0

    _report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "keyword"
    assert binding.matched_parameters == (
        "config",
        "example",
        "example_index",
        "input_data",
        "metadata",
        "output",
    )
    assert binding.bind_ok is True


# --------------------------------------------------------------------------- #
# var-positional / var-keyword
# --------------------------------------------------------------------------- #
def test_var_positional_metric():
    def metric(*args):
        return 1.0

    _report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "var_positional"
    assert binding.bind_ok is True


def test_var_keyword_metric():
    def metric(**kwargs):
        return 1.0

    _report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "keyword"
    assert binding.matched_parameters == ("expected", "llm_metrics", "output")
    assert binding.bind_ok is True


# --------------------------------------------------------------------------- #
# Positional fallbacks with unrecognized names -> NO_RECOGNIZED_PARAMS
# --------------------------------------------------------------------------- #
def test_positional_3_fallback():
    def metric(a, b, c):
        return 1.0

    report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "positional_3"
    assert binding.matched_parameters == ()
    assert binding.bind_ok is True
    assert ContractCode.EVALUATOR_NO_RECOGNIZED_PARAMS in finding_codes(report)


def test_positional_2_fallback():
    def metric(a, b):
        return 1.0

    _report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "positional_2"
    assert binding.unmatched_parameters == ("a", "b")


def test_positional_1_fallback():
    def metric(a):
        return 1.0

    _report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "positional_1"


def test_positional_0_fallback():
    def metric():
        return 1.0

    report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "positional_0"
    assert binding.bind_ok is True
    assert ContractCode.EVALUATOR_NO_RECOGNIZED_PARAMS in finding_codes(report)


def test_posonly_metric_binds_positionally():
    def metric(output, expected, /):
        return 1.0

    report, binding = _binding_for(scoring_function=metric)
    # Positional-only names can never be filled by keyword, so they fall back to
    # positional binding with zero recognized params (the mis-scoring hazard).
    assert binding.binding_mode == "positional_2"
    assert binding.matched_parameters == ()
    assert ContractCode.EVALUATOR_NO_RECOGNIZED_PARAMS in finding_codes(report)


# --------------------------------------------------------------------------- #
# Over-arity -> bind failure
# --------------------------------------------------------------------------- #
def test_over_arity_bind_failed():
    def metric(a, b, c, d):
        return 1.0

    report, binding = _binding_for(scoring_function=metric)
    assert binding.binding_mode == "unbound"
    assert binding.bind_ok is False
    assert report.ok is False
    finding = find_finding(report, ContractCode.EVALUATOR_BIND_FAILED)
    assert finding is not None
    assert finding.metric_name == "accuracy"


# --------------------------------------------------------------------------- #
# scoring_function placement via build_metric_functions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "objectives,expected_name",
    [
        (("accuracy",), "accuracy"),
        (("score",), "score"),
        (("f1",), "f1"),  # first objective when neither accuracy nor score
    ],
)
def test_scoring_function_mapped_to_objective(objectives, expected_name):
    def metric(output, expected):
        return 1.0

    report = validate_evaluation_contract(
        func=_agent,
        dataset=DATASET,
        scoring_function=metric,
        objectives=objectives,
    )
    assert [b.metric_name for b in report.evaluator_bindings] == [expected_name]


def test_metric_functions_and_scoring_function_both_reported():
    def scorer(output, expected):
        return 1.0

    def custom_metric(output, expected):
        return 0.5

    report = validate_evaluation_contract(
        func=_agent,
        dataset=DATASET,
        scoring_function=scorer,
        metric_functions={"custom_metric": custom_metric},
        objectives=("accuracy",),
    )
    names = sorted(b.metric_name for b in report.evaluator_bindings)
    assert names == ["accuracy", "custom_metric"]


# --------------------------------------------------------------------------- #
# Custom evaluator -> unsupported
# --------------------------------------------------------------------------- #
def test_custom_evaluator_unsupported():
    def custom_eval(func, config, example):
        return 1.0

    def raw_agent(question):
        return "x"

    # The phantom-config decoration warning is irrelevant to this contract test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt_agent = traigent.optimize(
            configuration_space={"model": ["a", "b"]}, custom_evaluator=custom_eval
        )(raw_agent)

    report = validate_evaluation_contract(func=opt_agent, dataset=DATASET)
    assert "custom_evaluator" in report.unsupported
    finding = find_finding(report, ContractCode.CUSTOM_EVALUATOR_UNSUPPORTED)
    assert finding is not None
    assert finding.severity == "warning"
    # A custom evaluator alone is advisory, not fatal.
    assert report.ok is True
