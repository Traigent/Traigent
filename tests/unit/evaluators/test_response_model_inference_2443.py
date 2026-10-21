"""Shared response-model inference: dict / llm_output identities (#2443, #2411)."""

from types import SimpleNamespace

import pytest

from traigent.evaluators.base import SimpleScoringEvaluator
from traigent.utils.response_model import infer_response_model
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


def _dict_response(model):
    body = {
        "usage": {
            "prompt_tokens": TOKENS_IN,
            "completion_tokens": TOKENS_OUT,
            "total_tokens": TOKENS_IN + TOKENS_OUT,
        }
    }
    if model:
        body["model"] = model
    return body


def test_infer_covers_attribute_dict_metadata_and_llm_output():
    assert infer_response_model(SimpleNamespace(model=JUDGE)) == JUDGE
    assert infer_response_model({"model": JUDGE}) == JUDGE
    assert infer_response_model({"model_name": JUDGE}) == JUDGE
    assert (
        infer_response_model(SimpleNamespace(response_metadata={"model_name": JUDGE}))
        == JUDGE
    )
    assert (
        infer_response_model(SimpleNamespace(llm_output={"model_name": JUDGE})) == JUDGE
    )
    assert infer_response_model({"llm_output": {"model_name": JUDGE}}) == JUDGE
    assert infer_response_model(object()) is None
    assert infer_response_model(None) is None


def test_dict_response_priced_at_its_own_model():
    metrics = _capture(_wrapper([_dict_response(JUDGE)]), {"model": AGENT})
    assert metrics.cost.total_cost == pytest.approx(_price(JUDGE))
    assert _price(JUDGE) != _price(AGENT)


def test_wrapper_extracts_llm_output_model_name():
    wrapper = _wrapper([])
    resp = SimpleNamespace(llm_output={"model_name": JUDGE})
    assert wrapper._extract_model_from_response(resp, {"model": AGENT}) == JUDGE


def test_config_fallback_only_when_response_has_no_identity():
    wrapper = _wrapper([])
    assert wrapper._extract_model_from_response({}, {"model": AGENT}) == AGENT
    metrics = _capture(_wrapper([_dict_response(None)]), {"model": AGENT})
    assert metrics.cost.total_cost == pytest.approx(_price(AGENT))


def test_local_evaluator_helper_reads_dict_identity():
    assert SimpleScoringEvaluator._model_of_captured_response({"model": JUDGE}) == JUDGE
    assert SimpleScoringEvaluator._model_of_captured_response({"x": 1}) is None


def test_unknown_model_stays_unmeasured():
    metrics = _capture(
        _wrapper([_dict_response("totally-unknown-model-xyz")]), {"model": AGENT}
    )
    assert metrics.cost.unpriced is True
    assert metrics.cost.total_cost == 0
