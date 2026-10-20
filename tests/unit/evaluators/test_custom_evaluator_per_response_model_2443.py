"""Each captured response is priced at the model it reports (issue #2443)."""

from types import SimpleNamespace

import pytest

from traigent.api.types import ExampleResult
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.evaluators.base import EvaluationExample
from traigent.utils.cost_calculator import cost_from_tokens

AGENT = "gpt-4o"
JUDGE = "gpt-4o-mini"
TOKENS_IN = 1000
TOKENS_OUT = 500


def _response(model):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=TOKENS_IN,
            completion_tokens=TOKENS_OUT,
            total_tokens=TOKENS_IN + TOKENS_OUT,
        ),
    )


def _wrapper(responses):
    async def evaluator(func, config, example):
        raise AssertionError("not called")

    wrapper = CustomEvaluatorWrapper(evaluator, metrics=["accuracy"])
    wrapper._metrics_available = True
    wrapper.capture_llm_metrics = True
    wrapper._get_all_captured_responses = lambda: responses
    return wrapper


def _capture(wrapper, config):
    example = EvaluationExample(input_data={"q": "x"}, expected_output="y")
    result = ExampleResult(
        example_id="e0",
        input_data=example.input_data,
        expected_output="y",
        actual_output="y",
        metrics={},
        execution_time=0.0,
        success=True,
        error_message=None,
        metadata={},
    )
    return wrapper._capture_llm_metrics_for_example(config, example, result, 0)


def _price(model):
    return sum(cost_from_tokens(TOKENS_IN, TOKENS_OUT, model))


def test_two_model_example_priced_at_each_calls_own_model():
    wrapper = _wrapper([_response(AGENT), _response(JUDGE)])
    metrics = _capture(wrapper, {"model": AGENT})
    assert metrics.cost.total_cost == pytest.approx(_price(AGENT) + _price(JUDGE))
    assert _price(AGENT) != _price(JUDGE)


def test_response_without_identity_falls_back_to_config_model():
    wrapper = _wrapper([_response(None)])
    metrics = _capture(wrapper, {"model": AGENT})
    assert metrics.cost.total_cost == pytest.approx(_price(AGENT))


def test_extract_model_prefers_response_over_config():
    wrapper = _wrapper([])
    assert (
        wrapper._extract_model_from_response(_response(JUDGE), {"model": AGENT})
        == JUDGE
    )
    assert (
        wrapper._extract_model_from_response(_response(None), {"model": AGENT}) == AGENT
    )
    assert wrapper._extract_model_from_response(None, {"model": AGENT}) == AGENT
