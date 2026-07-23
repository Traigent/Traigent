"""Drift guard: the contract's verdict must match a real ``LocalEvaluator`` run.

For representative matrix cells we run the ACTUAL production evaluator with
pure-Python functions and metrics (no LLM, no cost) and assert the contract's
``bind_ok`` / per-example shape matches whether the real run produced output vs.
hit a call-shape error. This pins the no-execution contract to the executing
path so a future refactor of either cannot silently diverge.
"""

from __future__ import annotations

import pytest

from traigent.contract import ContractCode, validate_evaluation_contract
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.exceptions import EvaluationError

from ._support import find_finding


def _dataset(input_data, expected):
    return Dataset(
        examples=[
            EvaluationExample(
                input_data=input_data, expected_output=expected, metadata={}
            )
        ]
    )


async def _run_real(func, input_data, expected, metric_functions):
    evaluator = LocalEvaluator(
        metric_functions=metric_functions, metrics=list(metric_functions)
    )
    return await evaluator.evaluate(func, {}, _dataset(input_data, expected))


def _good_metric(output, expected):
    return 1.0 if output else 0.0


async def test_parity_success_cell():
    def agent(question):
        return "out-" + question

    result = await _run_real(
        agent, {"question": "q1"}, "gold", {"accuracy": _good_metric}
    )
    # Real run produced output, no error.
    assert result.outputs == ["out-q1"]
    assert result.errors == [None]

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}, "output": "gold"}],
        metric_functions={"accuracy": _good_metric},
    )
    assert report.call_shapes[0].bind_ok is True
    assert report.evaluator_bindings[0].bind_ok is True
    assert report.ok is True


async def test_parity_agent_bind_failure_cell():
    def agent(question, answer):  # needs two positional; mapping supplies one
        return "x"

    result = await _run_real(
        agent, {"question": "q1"}, "gold", {"accuracy": _good_metric}
    )
    # Real run could not call the agent -> no output, an error recorded.
    assert result.outputs == [None]
    assert result.errors[0] is not None

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}, "output": "gold"}],
        metric_functions={"accuracy": _good_metric},
    )
    # Contract predicts the same failure without executing anything.
    assert report.call_shapes[0].bind_ok is False
    assert find_finding(report, ContractCode.AGENT_BIND_FAILED) is not None


async def test_parity_metric_bind_failure_cell():
    def agent(question):
        return "out"

    def over_arity_metric(a, b, c, d):  # cannot bind to (output, expected, ...)
        return 1.0

    # A failing OBJECTIVE metric fails the real trial closed.
    with pytest.raises(EvaluationError):
        await _run_real(
            agent, {"question": "q1"}, "gold", {"accuracy": over_arity_metric}
        )

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}, "output": "gold"}],
        metric_functions={"accuracy": over_arity_metric},
    )
    assert report.evaluator_bindings[0].bind_ok is False
    assert find_finding(report, ContractCode.EVALUATOR_BIND_FAILED) is not None


async def test_parity_positional_metric_cell():
    def agent(question):
        return "out"

    def positional_metric(prediction, gold):  # unrecognized names -> positional
        return 1.0 if prediction == "out" else 0.0

    result = await _run_real(
        agent, {"question": "q1"}, "gold", {"accuracy": positional_metric}
    )
    # Real run bound positionally and produced a metric value.
    assert result.outputs == ["out"]
    assert result.errors == [None]

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}, "output": "gold"}],
        metric_functions={"accuracy": positional_metric},
    )
    binding = report.evaluator_bindings[0]
    assert binding.bind_ok is True
    assert binding.binding_mode == "positional_2"
    # The mis-scoring hazard is surfaced as a warning, not an error.
    assert ContractCode.EVALUATOR_NO_RECOGNIZED_PARAMS in [
        str(f.code) for f in report.findings
    ]
    assert report.ok is True


async def test_parity_keyword_metric_cell():
    def agent(question):
        return "out"

    def keyword_metric(output, expected):
        return 1.0 if output == "out" else 0.0

    result = await _run_real(
        agent, {"question": "q1"}, "gold", {"accuracy": keyword_metric}
    )
    assert result.outputs == ["out"]
    assert result.errors == [None]

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}, "output": "gold"}],
        metric_functions={"accuracy": keyword_metric},
    )
    binding = report.evaluator_bindings[0]
    assert binding.bind_ok is True
    assert binding.binding_mode == "keyword"
    assert binding.matched_parameters == ("expected", "output")
