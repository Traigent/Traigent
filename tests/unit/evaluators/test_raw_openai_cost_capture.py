"""Raw ``openai`` SDK calls must reach trial cost/token metrics (issue #2441).

Before the fix only LangChain, LiteLLM and the Bedrock client captured usage.
An agent calling ``openai.OpenAI().chat.completions.create`` directly -- the
shape of every OpenAI-compatible gateway client -- was never captured, and the
custom-evaluator lane then aggregated the missing measurement as ``0.0``: a
paid run uploaded ``$0`` / ``0`` tokens for every trial.

These tests drive the REAL ``openai`` client against an ``httpx.MockTransport``
(no network, no spend), so the patched ``Completions.create`` is the one the
SDK actually ships.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

openai = pytest.importorskip("openai")

from traigent.api.types import ExampleResult  # noqa: E402
from traigent.config.context import ConfigurationContext  # noqa: E402
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper  # noqa: E402
from traigent.evaluators.base import Dataset, EvaluationExample  # noqa: E402
from traigent.integrations.framework_override import (  # noqa: E402
    FrameworkOverrideManager,
)
from traigent.utils import cost_calculator as cc  # noqa: E402
from traigent.utils.cost_calculator import cost_from_tokens  # noqa: E402
from traigent.utils.langchain_interceptor import (  # noqa: E402
    clear_captured_responses,
    get_all_captured_responses,
)

PROMPT_TOKENS = 11
COMPLETION_TOKENS = 7
KNOWN_MODEL = "gpt-4o-mini"
ALIAS_MODEL = "acme-gateway/house-model"
ALIAS_IN = 2e-6
ALIAS_OUT = 5e-6
LLM_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_cost",
    "output_cost",
    "total_cost",
)


def _completion_body(model: str, *, usage: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        body["usage"] = {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        }
    return body


def _handler(*, usage: bool = True):
    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}
                ],
            }
            text = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
            return httpx.Response(
                200, text=text, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=_completion_body(payload["model"], usage=usage))

    return handle


def _sync_client(*, usage: bool = True) -> Any:
    return openai.OpenAI(
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://gateway.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler(usage=usage))),
        max_retries=0,
    )


def _async_client(*, usage: bool = True) -> Any:
    return openai.AsyncOpenAI(
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://gateway.invalid/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_handler(usage=usage))
        ),
        max_retries=0,
    )


def _dataset(n: int = 2) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"question": f"q{i}"}, expected_output="ok")
            for i in range(n)
        ],
        name="raw-openai",
    )


def _evaluator(*, calls_per_example: int = 1) -> CustomEvaluatorWrapper:
    """A custom evaluator that runs the agent and scores exact match."""

    async def custom_evaluator(func, config, example):
        output = None
        for _ in range(calls_per_example):
            output = func(**example.input_data)
            if hasattr(output, "__await__"):
                output = await output
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0 if output == "ok" else 0.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    return CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy"])


@pytest.fixture
def openai_overrides():
    """Activate the openai.OpenAI / AsyncOpenAI framework overrides."""
    manager = FrameworkOverrideManager()
    manager.activate_overrides(["openai.OpenAI", "openai.AsyncOpenAI"])
    clear_captured_responses()
    try:
        yield manager
    finally:
        manager.deactivate_overrides()
        clear_captured_responses()


@pytest.fixture
def alias_pricing(monkeypatch):
    monkeypatch.setenv(
        "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
        json.dumps(
            {
                ALIAS_MODEL: {
                    "input_cost_per_token": ALIAS_IN,
                    "output_cost_per_token": ALIAS_OUT,
                }
            }
        ),
    )
    monkeypatch.delenv("TRAIGENT_CUSTOM_MODEL_PRICING_FILE", raising=False)
    cc._CUSTOM_PRICING_CACHE = None
    cc._CUSTOM_PRICING_CACHE_KEY = None
    yield
    cc._CUSTOM_PRICING_CACHE = None
    cc._CUSTOM_PRICING_CACHE_KEY = None


@pytest.fixture(autouse=True)
def _lenient_cost_accounting(monkeypatch):
    # Pricing is asserted explicitly below; strict mode would only turn an
    # unexpected miss into a raise instead of a clear assertion message.
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    monkeypatch.delenv("TRAIGENT_STRICT_METRICS_NULLS", raising=False)


def _sync_agent(client: Any):
    def agent(question: str) -> str:
        response = client.chat.completions.create(
            model=KNOWN_MODEL, messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    return agent


def _assert_measured(result, model: str, n: int, calls: int = 1) -> None:
    in_cost, out_cost = cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, model)
    assert in_cost + out_cost > 0, f"test premise: {model} must be priced"
    for row in result.example_results:
        assert row.metrics["input_tokens"] == PROMPT_TOKENS * calls
        assert row.metrics["output_tokens"] == COMPLETION_TOKENS * calls
        assert row.metrics["total_cost"] == pytest.approx((in_cost + out_cost) * calls)
        assert row.metadata["llm_usage_measured"] is True
    agg = result.aggregated_metrics
    assert agg["total_tokens"] == (PROMPT_TOKENS + COMPLETION_TOKENS) * n * calls
    assert agg["total_cost"] == pytest.approx((in_cost + out_cost) * n * calls)


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_sync_openai_usage_reaches_trial_metrics(openai_overrides):
    client = _sync_client()
    result = await _evaluator().evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    _assert_measured(result, KNOWN_MODEL, n=2)


@pytest.mark.asyncio
async def test_gateway_alias_is_priced_from_custom_pricing(
    openai_overrides, alias_pricing
):
    client = _sync_client()

    def agent(question: str) -> str:
        response = client.chat.completions.create(
            model=ALIAS_MODEL, messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    result = await _evaluator().evaluate(agent, {}, _dataset())
    expected = PROMPT_TOKENS * ALIAS_IN + COMPLETION_TOKENS * ALIAS_OUT
    for row in result.example_results:
        assert row.metrics["total_cost"] == pytest.approx(expected)
    assert result.aggregated_metrics["total_cost"] == pytest.approx(2 * expected)


@pytest.mark.asyncio
async def test_raw_async_openai_usage_reaches_trial_metrics(openai_overrides):
    client = _async_client()

    async def agent(question: str) -> str:
        response = await client.chat.completions.create(
            model=KNOWN_MODEL, messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    result = await _evaluator().evaluate(agent, {"model": KNOWN_MODEL}, _dataset())
    _assert_measured(result, KNOWN_MODEL, n=2)


@pytest.mark.asyncio
async def test_every_call_in_an_example_is_counted(openai_overrides):
    # A multi-call agent (e.g. plan + answer) used to be charged for its
    # first captured response only.
    client = _sync_client()
    result = await _evaluator(calls_per_example=3).evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    _assert_measured(result, KNOWN_MODEL, n=2, calls=3)


@pytest.mark.asyncio
async def test_enable_openai_optimization_turns_on_capture():
    from traigent.integrations.framework_override import disable_framework_overrides
    from traigent.integrations.llms.openai import enable_openai_optimization

    enable_openai_optimization()
    try:
        client = _sync_client()
        result = await _evaluator().evaluate(
            _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
        )
    finally:
        disable_framework_overrides()
        clear_captured_responses()
    _assert_measured(result, KNOWN_MODEL, n=2)


def test_capture_does_not_fire_when_overrides_are_inactive():
    manager = FrameworkOverrideManager()
    manager.activate_overrides(["openai.OpenAI"])
    manager.deactivate_overrides()
    clear_captured_responses()
    _sync_agent(_sync_client())("q")
    assert get_all_captured_responses() == []


def test_nested_instrumented_call_is_not_double_counted(openai_overrides):
    # LangChain's ChatOpenAI.invoke and litellm.completion capture their own
    # response; the openai call they make underneath must not be counted
    # a second time.
    from traigent.utils.langchain_interceptor import instrumented_provider_call

    client = _sync_client()
    with ConfigurationContext({"model": KNOWN_MODEL}):
        with instrumented_provider_call():
            _sync_agent(client)("q")
        assert get_all_captured_responses() == []
        _sync_agent(client)("q")
        assert len(get_all_captured_responses()) == 1


# --------------------------------------------------------------------------
# Honesty: a missing measurement is not $0
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_without_usage_is_unmeasured_not_zero(openai_overrides):
    client = _sync_client(usage=False)
    result = await _evaluator().evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    for row in result.example_results:
        assert row.metadata["llm_usage_measured"] is False
        assert not any(key in row.metrics for key in LLM_KEYS)
    assert not any(key in result.aggregated_metrics for key in LLM_KEYS)


@pytest.mark.asyncio
async def test_client_without_usage_is_unmeasured_not_zero():
    # A response without usage must remain unmeasured rather than free.
    client = _sync_client(usage=False)
    result = await _evaluator().evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    assert not any(key in result.aggregated_metrics for key in LLM_KEYS)


@pytest.mark.asyncio
async def test_strict_metrics_nulls_reports_unmeasured_as_none(monkeypatch):
    monkeypatch.setenv("TRAIGENT_STRICT_METRICS_NULLS", "true")
    client = _sync_client(usage=False)
    result = await _evaluator().evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    for key in LLM_KEYS:
        assert result.aggregated_metrics[key] is None


@pytest.mark.asyncio
async def test_streaming_call_without_usage_is_unmeasured(openai_overrides):
    client = _sync_client()

    def agent(question: str) -> str:
        stream = client.chat.completions.create(
            model=KNOWN_MODEL,
            messages=[{"role": "user", "content": question}],
            stream=True,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in stream)

    result = await _evaluator().evaluate(agent, {"model": KNOWN_MODEL}, _dataset())
    assert [r.actual_output for r in result.example_results] == ["ok", "ok"]
    assert not any(key in result.aggregated_metrics for key in LLM_KEYS)


@pytest.mark.asyncio
async def test_partially_measured_trial_sums_only_measured_rows(openai_overrides):
    measured = _sync_client()
    unmeasured = _sync_client(usage=False)

    def agent(question: str) -> str:
        client = measured if question == "q0" else unmeasured
        return _sync_agent(client)(question)

    result = await _evaluator().evaluate(agent, {"model": KNOWN_MODEL}, _dataset())
    assert [r.metadata["llm_usage_measured"] for r in result.example_results] == [
        True,
        False,
    ]
    assert (
        result.aggregated_metrics["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS
    )


# --------------------------------------------------------------------------
# End to end: what a trial carries (and therefore what is uploaded)
# --------------------------------------------------------------------------


def _optimize(monkeypatch, tmp_path, *, framework_targets: list[str] | None):
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    client = _sync_client()

    def agent(question: str) -> str:
        response = client.chat.completions.create(
            model="placeholder", messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    def custom_evaluator(func, config, example):
        output = func(**example.input_data)
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    opt_func = OptimizedFunction(
        func=agent,
        configuration_space={"model": [KNOWN_MODEL]},
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        max_trials=1,
        auto_override_frameworks=framework_targets is not None,
        framework_targets=framework_targets,
    )
    return opt_func.optimize_sync(algorithm="grid", max_trials=1, progress_bar=False)


def test_optimize_with_openai_override_records_real_cost(monkeypatch, tmp_path):
    result = _optimize(monkeypatch, tmp_path, framework_targets=["openai.OpenAI"])
    (trial,) = result.trials
    in_cost, out_cost = cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, KNOWN_MODEL)
    assert trial.metrics["total_tokens"] == 2 * (PROMPT_TOKENS + COMPLETION_TOKENS)
    assert trial.metrics["total_cost"] == pytest.approx(2 * (in_cost + out_cost))


def test_optimize_captures_raw_usage_without_framework_injection(monkeypatch, tmp_path):
    result = _optimize(monkeypatch, tmp_path, framework_targets=None)
    (trial,) = result.trials
    assert trial.metrics["total_cost"] > 0
    assert trial.metrics["total_tokens"] == 2 * (PROMPT_TOKENS + COMPLETION_TOKENS)


# --------------------------------------------------------------------------
# Review F1: an unmeasured trial must not win a cost objective
# --------------------------------------------------------------------------


def _evaluator_cost_objective() -> CustomEvaluatorWrapper:
    async def custom_evaluator(func, config, example):
        output = func(**example.input_data)
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    return CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy", "cost"])


@pytest.mark.asyncio
async def test_unmeasured_cost_objective_is_unknown_not_zero(openai_overrides):
    client = _sync_client(usage=False)
    result = await _evaluator_cost_objective().evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    assert "cost" not in result.aggregated_metrics


@pytest.mark.asyncio
async def test_unmeasured_cost_objective_is_none_under_strict_nulls(
    openai_overrides, monkeypatch
):
    monkeypatch.setenv("TRAIGENT_STRICT_METRICS_NULLS", "true")
    client = _sync_client(usage=False)
    result = await _evaluator_cost_objective().evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    assert result.aggregated_metrics["cost"] is None


@pytest.mark.asyncio
async def test_measured_cost_objective_is_the_trial_total(openai_overrides):
    client = _sync_client()
    result = await _evaluator_cost_objective().evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    in_cost, out_cost = cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, KNOWN_MODEL)
    assert result.aggregated_metrics["cost"] == pytest.approx(2 * (in_cost + out_cost))


def _optimize_mixed_coverage(monkeypatch, tmp_path, *, measured_models: set[str]):
    """Two models; only ``measured_models`` get a gateway that returns usage."""
    from traigent.config.context import get_config
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    measured, unmeasured = _sync_client(), _sync_client(usage=False)

    def agent(question: str) -> str:
        config = get_config()
        model = (
            config.get("model")
            if isinstance(config, dict)
            else getattr(config, "model", None)
        )
        client = measured if model in measured_models else unmeasured
        response = client.chat.completions.create(
            model="placeholder", messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    def custom_evaluator(func, config, example):
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=func(**example.input_data),
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    opt_func = OptimizedFunction(
        func=agent,
        configuration_space={"model": ["gpt-4o", "gpt-4o-mini"]},
        objectives=["accuracy", "cost"],
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        max_trials=2,
        auto_override_frameworks=True,
        framework_targets=["openai.OpenAI"],
    )
    return opt_func.optimize_sync(algorithm="grid", max_trials=2, progress_bar=False)


def test_unmeasured_trial_cannot_win_a_cost_objective(monkeypatch, tmp_path):
    # The reviewer's scenario: gpt-4o measured, gpt-4o-mini's gateway omits
    # usage. The mini trial used to report cost 0.0 and win on cost.
    result = _optimize_mixed_coverage(monkeypatch, tmp_path, measured_models={"gpt-4o"})
    by_model = {t.config["model"]: t for t in result.trials}
    assert by_model["gpt-4o"].metrics["cost"] > 0
    assert by_model["gpt-4o-mini"].metrics.get("cost") is None
    assert by_model["gpt-4o-mini"].metrics.get("total_cost") is None
    assert result.best_config == {"model": "gpt-4o"}
    assert "COST_OBJECTIVE_PARTIAL_USAGE_CAPTURED" in result.warning_codes
    assert "COST_OBJECTIVE_NO_USAGE_CAPTURED" not in result.warning_codes


def test_no_trial_measured_keeps_the_no_usage_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    result = _optimize_mixed_coverage(monkeypatch, tmp_path, measured_models=set())
    assert all(t.metrics.get("cost") is None for t in result.trials)
    assert "COST_OBJECTIVE_NO_USAGE_CAPTURED" in result.warning_codes
    assert "COST_OBJECTIVE_PARTIAL_USAGE_CAPTURED" not in result.warning_codes


def test_fully_measured_run_has_no_coverage_warning(monkeypatch, tmp_path):
    result = _optimize_mixed_coverage(
        monkeypatch, tmp_path, measured_models={"gpt-4o", "gpt-4o-mini"}
    )
    assert result.best_config == {"model": "gpt-4o-mini"}  # cheaper, same accuracy
    assert "COST_OBJECTIVE_PARTIAL_USAGE_CAPTURED" not in result.warning_codes


# --------------------------------------------------------------------------
# Review F3: a client built inside the optimized function
# --------------------------------------------------------------------------


def test_client_constructed_inside_the_function_is_not_given_model(
    monkeypatch, tmp_path
):
    # The documented pattern builds the client inside the function. The
    # constructor override used to inject call-time params such as ``model``
    # into ``openai.OpenAI.__init__``, which raises TypeError.
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))

    def agent(question: str) -> str:
        client = _sync_client()
        response = client.chat.completions.create(
            model="placeholder", messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    def custom_evaluator(func, config, example):
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=func(**example.input_data),
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    opt_func = OptimizedFunction(
        func=agent,
        configuration_space={"model": [KNOWN_MODEL]},
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        max_trials=1,
        auto_override_frameworks=True,
        framework_targets=["openai.OpenAI", "openai.AsyncOpenAI"],
    )
    result = opt_func.optimize_sync(algorithm="grid", max_trials=1, progress_bar=False)
    (trial,) = result.trials
    assert trial.is_successful
    assert trial.metrics["total_tokens"] == 2 * (PROMPT_TOKENS + COMPLETION_TOKENS)


def test_constructor_override_skips_params_the_constructor_rejects():
    from traigent.integrations.framework_override import _constructor_keyword_names

    class Strict:
        def __init__(self, api_key=None, *, base_url=None):
            pass

    class Loose:
        def __init__(self, **kwargs):
            pass

    assert _constructor_keyword_names(Strict.__init__) == {
        "self",
        "api_key",
        "base_url",
    }
    assert _constructor_keyword_names(Loose.__init__) is None
    assert "model" not in _constructor_keyword_names(openai.OpenAI.__init__)


# --------------------------------------------------------------------------
# F2, owner decision: the unknown-cost safety cap applies only to runs the
# user did not size explicitly (max_trials / max_total_examples).
# --------------------------------------------------------------------------


def _run(
    monkeypatch,
    tmp_path,
    *,
    unmeasured_when=lambda temperature: True,
    configs: int = 15,
    construct_kwargs: dict | None = None,
    **optimize_kwargs,
):
    """``configs`` temperatures; the agent's usage is captured unless
    ``unmeasured_when(temperature)``. No run size is set unless passed."""
    from traigent.config.context import get_config
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    measured, unmeasured = _sync_client(), _sync_client(usage=False)

    def agent(question: str) -> str:
        config = get_config()
        temperature = (
            config.get("temperature")
            if isinstance(config, dict)
            else getattr(config, "temperature", None)
        )
        client = unmeasured if unmeasured_when(temperature) else measured
        return _sync_agent(client)(question)

    def custom_evaluator(func, config, example):
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=func(**example.input_data),
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    opt_func = OptimizedFunction(
        func=agent,
        configuration_space={"temperature": [i / 10 for i in range(configs)]},
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        auto_override_frameworks=True,
        framework_targets=["openai.OpenAI"],
        **(construct_kwargs or {}),
    )
    return opt_func.optimize_sync(
        algorithm="grid", progress_bar=False, **optimize_kwargs
    )


def _stop_message(result) -> str:
    (message,) = [w for w in result.warnings if w.startswith("Cost could not be")]
    return message


def test_default_sized_unmeasured_run_stops_at_the_safety_limit(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)
    result = _run(monkeypatch, tmp_path)

    assert len(result.trials) == 10
    assert result.stop_reason == "cost_limit"
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" in result.warning_codes
    message = _stop_message(result)
    assert "none of the 10 trials' cost could be measured" in message
    for text in (
        "stopped at the default safety limit of 10 trials",
        "Set max_trials explicitly on @traigent.optimize or .optimize() to run "
        "more trials",
        "max_total_examples caps the total examples and also counts as your "
        "explicit consent, but does not raise max_trials",
        ".invoke/.ainvoke/.batch/.abatch",
        "docs/user-guide/cost_capture.md",
        "enable_openai_optimization()",
        "TRAIGENT_FALLBACK_TRIAL_LIMIT",
    ):
        assert text in message
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" in caplog.text


def test_env_var_overrides_the_default_safety_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", "4")
    result = _run(monkeypatch, tmp_path)
    assert len(result.trials) == 4
    assert "default safety limit of 4 trials" in _stop_message(result)


def test_env_var_applies_when_cost_limit_is_set(monkeypatch, tmp_path):
    # cost_limit builds the enforcer config directly; that path used to ignore
    # TRAIGENT_FALLBACK_TRIAL_LIMIT and use 10.
    monkeypatch.setenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", "4")
    result = _run(monkeypatch, tmp_path, cost_limit=5.0)
    assert len(result.trials) == 4


@pytest.mark.parametrize("where", ["call", "constructor"])
def test_explicit_max_trials_runs_to_that_size_with_a_warning(
    monkeypatch, tmp_path, where
):
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)
    if where == "call":
        result = _run(monkeypatch, tmp_path, max_trials=15)
    else:
        result = _run(monkeypatch, tmp_path, construct_kwargs={"max_trials": 15})

    assert len(result.trials) == 15
    assert result.stop_reason != "cost_limit"
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" not in result.warning_codes
    assert "COST_UNMEASURED_TRIALS_RAN" in result.warning_codes
    (message,) = [w for w in result.warnings if "ran with a cost that" in w]
    assert message.startswith("15 of 15 trials ran with a cost")


def test_constructor_max_trials_none_is_not_explicit(monkeypatch, tmp_path):
    # Review of #2447, F1: OptimizedFunction(max_trials=None) used to count as
    # an explicit size, so an unsized, unbounded run ignored the safety limit.
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)
    result = _run(monkeypatch, tmp_path, construct_kwargs={"max_trials": None})
    assert len(result.trials) == 10
    assert result.stop_reason == "cost_limit"
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" in result.warning_codes
    assert "COST_UNMEASURED_TRIALS_RAN" not in result.warning_codes


def test_decorator_max_trials_counts_as_explicit(monkeypatch, tmp_path):
    import traigent

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)

    def custom_evaluator(func, config, example):
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=func(**example.input_data),
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    @traigent.optimize(
        configuration_space={"temperature": [i / 10 for i in range(12)]},
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        max_trials=12,
    )
    def agent(question: str) -> str:
        return "ok"  # no captured LLM call

    result = agent.optimize_sync(algorithm="grid", progress_bar=False)
    assert len(result.trials) == 12
    assert "COST_UNMEASURED_TRIALS_RAN" in result.warning_codes


def test_explicit_max_total_examples_counts_as_explicit(monkeypatch, tmp_path):
    # With the default safety limit lowered to 4, a run that sets only
    # max_total_examples runs past it: 2 examples per trial, 40 examples,
    # bounded by the default max_trials of 10.
    monkeypatch.setenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", "4")
    result = _run(monkeypatch, tmp_path, max_total_examples=40)
    assert len(result.trials) == 10
    assert result.stop_reason != "cost_limit"
    assert "COST_UNMEASURED_TRIALS_RAN" in result.warning_codes


def test_measured_run_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)
    result = _run(
        monkeypatch,
        tmp_path,
        unmeasured_when=lambda temperature: False,
        max_trials=15,
        cost_limit=1.0,
    )
    assert len(result.trials) == 15
    assert result.stop_reason != "cost_limit"
    assert not {
        "COST_UNMEASURED_TRIAL_LIMIT_REACHED",
        "COST_UNMEASURED_TRIALS_RAN",
    } & set(result.warning_codes)


def test_mixed_run_stop_message_counts_unmeasured_and_measured_trials(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)
    result = _run(
        monkeypatch, tmp_path, unmeasured_when=lambda temperature: temperature == 0.0
    )
    assert len(result.trials) == 10
    assert result.stop_reason == "cost_limit"
    message = _stop_message(result)
    assert "1 of 10 trials had no measurable cost" in message
    assert "9 were measured; one unmeasured trial is enough" in message
    assert "for every configuration" in message
    assert "none of the" not in message


# --------------------------------------------------------------------------
# Review of 078314f5: metric_limit on a cost metric, results table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "objectives"),
    [("cost", ["accuracy", "cost"]), ("total_cost", ["accuracy"])],
)
def test_metric_limit_on_cost_does_not_fail_an_unmeasured_run(
    monkeypatch, tmp_path, metric_name, objectives
):
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")

    def custom_evaluator(func, config, example):
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=func(**example.input_data),
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    opt_func = OptimizedFunction(
        func=lambda question: "ok",
        configuration_space={"temperature": [0.1, 0.2, 0.3]},
        objectives=objectives,
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        max_trials=3,
    )
    result = opt_func.optimize_sync(
        algorithm="grid",
        max_trials=3,
        progress_bar=False,
        metric_limit=1.0,
        metric_name=metric_name,
    )
    assert len(result.trials) == 3
    assert all(t.is_successful for t in result.trials)


def test_metric_limit_still_requires_non_cost_metrics():
    from traigent.api.types import TrialResult, TrialStatus
    from traigent.core.stop_conditions import MetricLimitStopCondition

    condition = MetricLimitStopCondition(limit=1.0, metric_name="tokens_used")
    trial = TrialResult(
        trial_id="t1",
        config={},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=None,
    )
    with pytest.raises(ValueError, match="Mandatory metric 'tokens_used' missing"):
        condition.should_stop([trial])


def test_results_table_shows_unmeasured_cost_as_na():
    from traigent.utils.results_table import _render_metric_cell

    assert _render_metric_cell("cost", None) == "n/a"
    assert _render_metric_cell("total_cost", None) == "n/a"
    assert _render_metric_cell("cost", 0.0) != "n/a"
    assert _render_metric_cell("accuracy", None) == _render_metric_cell("accuracy", 0.0)


# --------------------------------------------------------------------------
# Per-call model pricing (#2443)
# --------------------------------------------------------------------------


@pytest.fixture
def clean_capture():
    """Leave no captured response behind for later tests on this worker."""
    clear_captured_responses()
    try:
        yield
    finally:
        clear_captured_responses()


def _captured_completion(model: str | None) -> Any:
    """Record one OpenAI-shaped response the way an instrumented wrapper does."""
    from openai.types.chat import ChatCompletion

    from traigent.utils.langchain_interceptor import capture_langchain_response

    response = ChatCompletion.model_validate(_completion_body(model or "x"))
    if model is None:
        object.__setattr__(response, "model", "")
    capture_langchain_response(response)
    return response


@pytest.mark.asyncio
async def test_each_captured_call_is_priced_at_its_own_model(clean_capture):
    # The agent runs gpt-4o; its judge runs gpt-4o-mini in the same example.
    # Pricing both at config["model"] charged the judge at gpt-4o's rate.
    agent_model, judge_model = "gpt-4o", KNOWN_MODEL
    agent_cost = sum(cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, agent_model))
    judge_cost = sum(cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, judge_model))
    assert agent_cost != pytest.approx(judge_cost), "test premise: prices differ"

    def agent(question: str) -> str:
        _captured_completion(agent_model)
        _captured_completion(judge_model)
        return "ok"

    result = await _evaluator().evaluate(agent, {"model": agent_model}, _dataset())

    for row in result.example_results:
        _failure_detail = (
            "a judge call on another model was priced at the trial's model (#2443)"
        )
        _actual_matches = row.metrics["total_cost"] == pytest.approx(
            agent_cost + judge_cost
        )
        assert _actual_matches, _failure_detail
    assert result.aggregated_metrics["total_cost"] == pytest.approx(
        2 * (agent_cost + judge_cost)
    )


@pytest.mark.asyncio
async def test_call_without_a_reported_model_falls_back_to_config_model(clean_capture):
    agent_model = "gpt-4o"
    agent_cost = sum(cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, agent_model))

    def agent(question: str) -> str:
        _captured_completion(None)
        return "ok"

    result = await _evaluator().evaluate(agent, {"model": agent_model}, _dataset())

    for row in result.example_results:
        assert row.metrics["total_cost"] == pytest.approx(agent_cost)


@pytest.mark.asyncio
async def test_unpriced_reported_model_falls_back_to_priced_config_model(
    alias_pricing, clean_capture
):
    # A gateway that reports an internal name keeps the configured name's price.
    def agent(question: str) -> str:
        _captured_completion("acme-internal/unpriced-build-7")
        return "ok"

    result = await _evaluator().evaluate(agent, {"model": ALIAS_MODEL}, _dataset())

    expected = PROMPT_TOKENS * ALIAS_IN + COMPLETION_TOKENS * ALIAS_OUT
    for row in result.example_results:
        assert row.metrics["total_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_simple_scoring_lane_charges_every_captured_call(clean_capture):
    # SimpleScoringEvaluator priced captured_responses[0] only, so an agent
    # plus a judge call on another model was charged for the agent call alone
    # (#2444 item 1), at the trial's model (#2443).
    from traigent.evaluators.base import SimpleScoringEvaluator

    agent_model, judge_model = "gpt-4o", KNOWN_MODEL
    agent_cost = sum(cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, agent_model))
    judge_cost = sum(cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, judge_model))

    def agent(question: str) -> str:
        _captured_completion(agent_model)
        _captured_completion(judge_model)
        return "ok"

    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda output, expected: 1.0, metrics=["accuracy"]
    )
    result = await evaluator.evaluate(agent, {"model": agent_model}, _dataset())

    assert len(result.example_results) == 2
    for row in result.example_results:
        _failure_detail = (
            "only the first captured call of the example was charged (#2444)"
        )
        _actual_matches = row.metrics["total_cost"] == pytest.approx(
            agent_cost + judge_cost
        )
        assert _actual_matches, _failure_detail
        assert row.metrics["total_tokens"] == 2 * (PROMPT_TOKENS + COMPLETION_TOKENS)


# --------------------------------------------------------------------------
# LangChain async entry points (#2445)
# --------------------------------------------------------------------------


def _langchain_chat(**kwargs: Any) -> Any:
    langchain_openai = pytest.importorskip("langchain_openai")
    from traigent.utils.langchain_interceptor import (
        patch_langchain_for_metadata_capture,
    )

    patch_langchain_for_metadata_capture()
    return langchain_openai.ChatOpenAI(
        model=KNOWN_MODEL,
        api_key="test-key",  # pragma: allowlist secret
        base_url="https://gateway.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler())),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler())),
        max_retries=0,
        **kwargs,
    )


@pytest.fixture
def real_llm_path(monkeypatch):
    """The capture wrappers return a canned reply in mock mode; these tests
    exercise the real (mock-transport) provider path."""
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    monkeypatch.delenv("TRAIGENT_GENERATE_MOCKS", raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["ainvoke", "abatch", "batch"])
async def test_langchain_async_and_batch_calls_are_captured(
    entry_point, real_llm_path, clean_capture
):
    llm = _langchain_chat()

    async def agent(question: str) -> str:
        if entry_point == "ainvoke":
            message = await llm.ainvoke(question)
        elif entry_point == "abatch":
            (message,) = await llm.abatch([question])
        else:
            (message,) = llm.batch([question])
        return message.content

    result = await _evaluator().evaluate(agent, {"model": KNOWN_MODEL}, _dataset())
    _assert_measured(result, KNOWN_MODEL, n=2)


def test_async_langchain_agent_is_not_cut_at_the_fallback_limit(
    monkeypatch, tmp_path, real_llm_path, clean_capture
):
    # Before #2445 every ainvoke trial was unmeasured, so a run with no
    # explicit size stopped at TRAIGENT_FALLBACK_TRIAL_LIMIT (4 here) instead
    # of running all 6 configurations.
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", "4")
    llm = _langchain_chat()

    async def agent(question: str) -> str:
        return (await llm.ainvoke(question)).content

    async def custom_evaluator(func, config, example):
        output = await func(**example.input_data)
        return ExampleResult(
            example_id=example.input_data["question"],
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    opt_func = OptimizedFunction(
        func=agent,
        configuration_space={"temperature": [i / 10 for i in range(6)]},
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
    )
    result = opt_func.optimize_sync(algorithm="grid", progress_bar=False)

    assert len(result.trials) == 6
    assert result.stop_reason != "cost_limit"
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" not in result.warning_codes
    in_cost, out_cost = cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, KNOWN_MODEL)
    for trial in result.trials:
        assert trial.metrics["total_cost"] == pytest.approx(2 * (in_cost + out_cost))


# Missing usage must remain unknown in the default evaluator lanes (#2517).
def _default_measurement_evaluator(lane, *, custom_cost=None):
    from traigent.evaluators.base import SimpleScoringEvaluator
    from traigent.evaluators.local import LocalEvaluator

    if lane == "local":
        kwargs = (
            {}
            if custom_cost is None
            else {"metric_functions": {"cost": lambda output, expected: custom_cost}}
        )
        return LocalEvaluator(metrics=["accuracy", "cost"], detailed=True, **kwargs)

    def score(output, expected):
        values = {"accuracy": float(output == expected)}
        if custom_cost is not None:
            values["cost"] = custom_cost
        return values

    return SimpleScoringEvaluator(metrics=["accuracy", "cost"], scoring_function=score)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local", "simple"])
async def test_raw_response_without_usage_is_unknown_in_default_lanes(
    lane, clean_capture
):
    client = _sync_client(usage=False)
    try:
        result = await _default_measurement_evaluator(lane).evaluate(
            _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
        )
        assert not get_all_captured_responses()
        assert result.aggregated_metrics.get("cost") is None
        assert result.aggregated_metrics.get("total_cost") is None
        assert result.aggregated_metrics["accuracy"] == 1.0
        for row in result.example_results:
            assert row.metrics.get("cost") is None
            assert row.metrics.get("total_cost") is None
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local", "simple"])
@pytest.mark.parametrize("custom_cost", [0.0, 0.25])
async def test_explicit_custom_cost_stays_authoritative_without_usage(
    lane, custom_cost, clean_capture
):
    result = await _default_measurement_evaluator(
        lane, custom_cost=custom_cost
    ).evaluate(lambda question: "ok", {"model": KNOWN_MODEL}, _dataset())
    assert result.aggregated_metrics["cost"] == custom_cost


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local", "simple"])
async def test_unpriced_captured_usage_has_unknown_cost_in_default_lanes(
    lane, clean_capture
):
    def agent(question):
        _captured_completion("fictional/unpriced-model")
        return "ok"

    result = await _default_measurement_evaluator(lane).evaluate(
        agent, {"model": "fictional/unpriced-model"}, _dataset()
    )
    assert result.aggregated_metrics["cost_unpriced"] == 1.0
    assert result.aggregated_metrics.get("cost") is None
    assert result.aggregated_metrics.get("total_cost") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local", "simple"])
async def test_measured_calls_preserve_real_cost_in_default_lanes(lane, clean_capture):
    def agent(question):
        _captured_completion(KNOWN_MODEL)
        return "ok"

    result = await _default_measurement_evaluator(lane).evaluate(
        agent, {"model": KNOWN_MODEL}, _dataset()
    )
    charge = sum(cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, KNOWN_MODEL))
    expected = 2 * charge if lane == "local" else charge
    assert result.aggregated_metrics["cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local", "simple"])
async def test_provider_reported_zero_stays_a_real_measurement(lane, clean_capture):
    def agent(question):
        response = _captured_completion(KNOWN_MODEL)
        object.__setattr__(response.usage, "cost", 0.0)
        return "ok"

    result = await _default_measurement_evaluator(lane).evaluate(
        agent, {"model": KNOWN_MODEL}, _dataset()
    )
    assert result.aggregated_metrics["cost"] == 0.0
    assert all(row.metrics["total_cost"] == 0.0 for row in result.example_results)
    assert result.aggregated_metrics["cost_unpriced"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local", "simple"])
async def test_custom_zero_objective_does_not_replace_real_captured_spend(
    lane, clean_capture
):
    def agent(question):
        _captured_completion(KNOWN_MODEL)
        return "ok"

    result = await _default_measurement_evaluator(lane, custom_cost=0.0).evaluate(
        agent, {"model": KNOWN_MODEL}, _dataset()
    )
    assert result.aggregated_metrics["cost"] == 0.0
    expected = sum(cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, KNOWN_MODEL))
    assert sum(
        row.metrics["total_cost"] for row in result.example_results
    ) == pytest.approx(2 * expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local", "simple"])
async def test_free_call_plus_unpriced_call_is_unknown_not_free(lane, clean_capture):
    def agent(question):
        response = _captured_completion(KNOWN_MODEL)
        object.__setattr__(response.usage, "cost", 0.0)
        _captured_completion("fictional/unpriced-model")
        return "ok"

    result = await _default_measurement_evaluator(lane).evaluate(
        agent, {"model": "fictional/unpriced-model"}, _dataset()
    )
    assert result.aggregated_metrics["cost"] is None
    assert result.aggregated_metrics["cost_unpriced"] == 1.0


def test_default_optimize_captures_provider_spend_without_framework_injection(
    monkeypatch, tmp_path, real_llm_path, clean_capture
):
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    client = _sync_client()
    provider_usage = []

    def agent(question):
        response = client.chat.completions.create(
            model=KNOWN_MODEL, messages=[{"role": "user", "content": question}]
        )
        provider_usage.append(response.usage.total_tokens)
        return response.choices[0].message.content

    optimized = OptimizedFunction(
        func=agent,
        configuration_space={"model": [KNOWN_MODEL]},
        objectives=["accuracy", "cost"],
        eval_dataset=_dataset(1),
        max_trials=1,
    )
    try:
        result = optimized.optimize_sync(
            algorithm="grid", max_trials=1, progress_bar=False
        )
    finally:
        client.close()
    assert provider_usage == [PROMPT_TOKENS + COMPLETION_TOKENS]
    (trial,) = result.trials
    assert trial.metrics["accuracy"] == 1.0
    assert trial.metrics["cost"] == sum(
        cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, KNOWN_MODEL)
    )
    assert trial.metrics["total_cost"] == trial.metrics["cost"]
    assert trial.metrics["cost_unpriced"] == 0.0
    assert "COST_OBJECTIVE_NO_USAGE_CAPTURED" not in result.warning_codes


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.asyncio
async def test_default_capture_scope_records_raw_nonstream_calls(async_client):
    from traigent.utils.langchain_interceptor import capture_scope

    client = _async_client() if async_client else _sync_client()
    try:
        async with capture_scope():
            for _ in range(2):
                response = client.chat.completions.create(
                    model=KNOWN_MODEL,
                    messages=[{"role": "user", "content": "unchanged request"}],
                )
                if async_client:
                    response = await response
                assert response.usage.prompt_tokens == PROMPT_TOKENS
            captured = get_all_captured_responses()
            assert len(captured) == 2
            assert all(
                r.usage.total_tokens == PROMPT_TOKENS + COMPLETION_TOKENS
                for r in captured
            )
    finally:
        if async_client:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["local", "simple"])
async def test_default_evaluator_uses_actual_raw_openai_usage(lane):

    client = _sync_client()
    try:
        evaluator = _default_measurement_evaluator(lane)
        result = await evaluator.evaluate(
            _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
        )
        for row in result.example_results:
            assert row.metrics["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS
            assert row.metrics["cost"] == pytest.approx(
                sum(cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, KNOWN_MODEL))
            )
        assert result.aggregated_metrics["cost"] > 0
    finally:
        client.close()


@pytest.mark.parametrize("activate_inside_scope", [False, True])
def test_capture_and_framework_override_restore_in_either_installation_order(
    activate_inside_scope,
):
    from openai.resources.chat.completions import Completions

    from traigent.utils.langchain_interceptor import capture_scope

    original = Completions.create
    manager = FrameworkOverrideManager()
    client = _sync_client()
    try:
        if not activate_inside_scope:
            manager.activate_overrides(["openai.OpenAI"])
        with capture_scope():
            if activate_inside_scope:
                manager.activate_overrides(["openai.OpenAI"])
            with ConfigurationContext({"model": KNOWN_MODEL}):
                _sync_agent(client)("one call")
            assert len(get_all_captured_responses()) == 1
        manager.deactivate_overrides()
        assert Completions.create is original
    finally:
        manager.deactivate_overrides()
        client.close()


@pytest.mark.asyncio
async def test_concurrent_raw_capture_scopes_keep_response_ownership():
    import asyncio

    from openai.resources.chat.completions import AsyncCompletions

    from traigent.utils.langchain_interceptor import (
        capture_key,
        capture_scope,
        get_captured_responses_by_key,
    )

    original = AsyncCompletions.create
    ready = asyncio.Event()
    entered = 0

    async def run(key):
        nonlocal entered
        client = _async_client()
        try:
            async with capture_scope():
                with capture_key(key):
                    entered += 1
                    if entered == 2:
                        ready.set()
                    await ready.wait()
                    response = await client.chat.completions.create(
                        model=KNOWN_MODEL, messages=[{"role": "user", "content": key}]
                    )
                    captured = get_all_captured_responses()
                    assert len(captured) == 1 and captured[0] is response
                    owned = get_captured_responses_by_key()
                    assert list(owned) == [key]
                    assert owned[key][0] is response
                    return response
        finally:
            await client.close()

    first, second = await asyncio.gather(run("first"), run("second"))
    assert first is not second
    assert AsyncCompletions.create is original


@pytest.mark.asyncio
async def test_cancelled_provider_call_restores_scoped_resource_method():
    import asyncio

    from openai.resources.chat.completions import AsyncCompletions

    from traigent.utils.langchain_interceptor import capture_scope

    original = AsyncCompletions.create
    started = asyncio.Event()

    async def delayed(request):
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled request unexpectedly resumed")

    client = openai.AsyncOpenAI(
        api_key="test-key",  # pragma: allowlist secret
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(delayed)),
        max_retries=0,
    )

    async def request():
        async with capture_scope() as bucket:
            try:
                await client.chat.completions.create(
                    model=KNOWN_MODEL, messages=[{"role": "user", "content": "cancel"}]
                )
            finally:
                assert bucket.responses == []

    task = asyncio.create_task(request())
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert AsyncCompletions.create is original
    finally:
        await client.close()


@pytest.mark.parametrize("response_model", [KNOWN_MODEL, "fictional-unpriced-model"])
def test_default_public_run_without_size_uses_raw_usage_and_preserves_cost_safety(
    response_model,
    monkeypatch,
    tmp_path,
    real_llm_path,
):
    from traigent.config.context import get_config
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=_completion_body(response_model))

    client = openai.OpenAI(
        api_key="test-key",  # pragma: allowlist secret
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    )

    def agent(question):
        config = get_config()
        temperature = config.get("temperature")
        response = client.chat.completions.create(
            model=response_model,
            temperature=temperature,
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content

    optimized = OptimizedFunction(
        func=agent,
        configuration_space={"temperature": [i / 20 for i in range(12)]},
        objectives=["accuracy", "cost"],
        eval_dataset=_dataset(1),
    )
    try:
        result = optimized.optimize_sync(algorithm="grid", progress_bar=False)
    finally:
        client.close()
    assert len(requests) == len(result.trials)
    assert all(
        t.metrics["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS
        for t in result.trials
    )
    assert all(r["model"] == response_model for r in requests)
    assert len({r["temperature"] for r in requests}) == len(requests)
    if response_model == KNOWN_MODEL:
        # No explicit run size keeps the documented constructor default.
        assert len(result.trials) == 10
        assert result.stop_reason == "max_trials_reached"
        assert all(t.metrics["total_cost"] > 0 for t in result.trials)
        assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" not in result.warning_codes
    else:
        assert len(result.trials) == 10
        assert all(t.metrics.get("cost") is None for t in result.trials)
        assert all(t.metrics.get("total_cost") is None for t in result.trials)
        assert all(t.metrics["cost_unpriced"] == 1 for t in result.trials)
        assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" in result.warning_codes
        assert result.stop_reason == "cost_limit"
        warning = " ".join(result.warnings)
        assert "Pricing is UNKNOWN" in warning
        assert "unavailable" in warning
        assert "report $0" not in warning
        assert "recorded as $0" not in warning


@pytest.mark.parametrize("activate_inside_scope", [False, True])
def test_deactivate_framework_preserves_active_default_capture(activate_inside_scope):
    from openai.resources.chat.completions import Completions

    from traigent.utils.langchain_interceptor import capture_scope

    original = Completions.create
    manager = FrameworkOverrideManager()
    client = _sync_client()
    try:
        if not activate_inside_scope:
            manager.activate_overrides(["openai.OpenAI"])
        with capture_scope():
            if activate_inside_scope:
                manager.activate_overrides(["openai.OpenAI"])
            manager.deactivate_overrides()
            _sync_agent(client)("after framework deactivation")
            assert len(get_all_captured_responses()) == 1
            with capture_scope():
                _sync_agent(client)("nested active scope")
                assert len(get_all_captured_responses()) == 1
            _sync_agent(client)("outer scope still active")
            assert len(get_all_captured_responses()) == 2
        assert Completions.create is original
    finally:
        manager.deactivate_overrides()
        client.close()


@pytest.mark.asyncio
async def test_framework_deactivation_keeps_other_concurrent_capture_alive():
    import asyncio

    from openai.resources.chat.completions import AsyncCompletions

    from traigent.utils.langchain_interceptor import capture_key, capture_scope

    original = AsyncCompletions.create
    manager = FrameworkOverrideManager()
    manager.activate_overrides(["openai.AsyncOpenAI"])
    ready = asyncio.Event()
    deactivated = asyncio.Event()
    entered = 0

    async def run(key):
        nonlocal entered
        client = _async_client()
        try:
            async with capture_scope():
                with capture_key(key):
                    entered += 1
                    if entered == 2:
                        ready.set()
                    await deactivated.wait()
                    response = await client.chat.completions.create(
                        model=KNOWN_MODEL,
                        messages=[{"role": "user", "content": key}],
                    )
                    assert get_all_captured_responses() == [response]
                    return response
        finally:
            await client.close()

    tasks = [asyncio.create_task(run(key)) for key in ("first", "second")]
    try:
        await ready.wait()
        manager.deactivate_overrides()
        deactivated.set()
        first, second = await asyncio.gather(*tasks)
        assert first is not second
        assert AsyncCompletions.create is original
    finally:
        manager.deactivate_overrides()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_default_public_mixed_pricing_warning_preserves_unknown_cost(
    monkeypatch,
    tmp_path,
    real_llm_path,
):
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    calls = []

    def respond(request):
        model = KNOWN_MODEL if not calls else "fictional-unpriced-model"
        calls.append(model)
        return httpx.Response(200, json=_completion_body(model))

    client = openai.OpenAI(
        api_key="test-key",  # pragma: allowlist secret
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    )

    def agent(question):
        for _ in range(2):
            response = client.chat.completions.create(
                model="fictional-unpriced-model",
                messages=[{"role": "user", "content": question}],
            )
        return response.choices[0].message.content

    optimized = OptimizedFunction(
        func=agent,
        configuration_space={"temperature": [0.1]},
        objectives=["accuracy", "cost"],
        eval_dataset=_dataset(1),
        max_trials=1,
    )
    try:
        result = optimized.optimize_sync(algorithm="grid", progress_bar=False)
    finally:
        client.close()
    assert calls == [KNOWN_MODEL, "fictional-unpriced-model"]
    (trial,) = result.trials
    measured_spend = sum(
        cost_from_tokens(PROMPT_TOKENS, COMPLETION_TOKENS, KNOWN_MODEL)
    )
    assert trial.metrics["cost"] == pytest.approx(measured_spend)
    assert trial.metrics["total_cost"] == pytest.approx(measured_spend)
    assert trial.metrics["cost_unpriced"] == 1
    warning = " ".join(result.warnings)
    assert "Pricing is UNKNOWN" in warning
    assert "monetary measurements remain unavailable" in warning
    assert "only measured calls is a lower bound" in warning
    assert "report $0" not in warning and "recorded as $0" not in warning


def test_last_concurrent_scope_release_during_framework_restore_is_atomic(monkeypatch):
    import threading

    from openai.resources.chat.completions import AsyncCompletions, Completions

    from traigent.utils import openai_interceptor
    from traigent.utils.langchain_interceptor import capture_scope

    original = Completions.create
    async_original = AsyncCompletions.create
    manager = FrameworkOverrideManager()
    manager.activate_overrides(["openai.OpenAI", "openai.AsyncOpenAI"])
    opened = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    failures = []

    def lifetime():
        try:
            with capture_scope():
                opened.set()
                assert release.wait(5)
        except BaseException as exc:
            failures.append(exc)
        finally:
            closed.set()

    worker = threading.Thread(target=lifetime)
    worker.start()
    restore = openai_interceptor.restore_openai_capture_original

    def interposed(*args):
        result = restore(*args)
        if args[0] is Completions:
            release.set()
            assert closed.wait(5)
        return result

    try:
        assert opened.wait(5)
        monkeypatch.setattr(
            openai_interceptor, "restore_openai_capture_original", interposed
        )
        manager.deactivate_overrides()
        worker.join(5)
        assert not worker.is_alive() and not failures
        assert Completions.create is original
        assert AsyncCompletions.create is async_original
    finally:
        release.set()
        worker.join(5)
        manager.deactivate_overrides()


def test_new_scope_acquire_after_atomic_framework_restore_keeps_capture(monkeypatch):
    import threading

    from openai.resources.chat.completions import Completions

    from traigent.utils import openai_interceptor
    from traigent.utils.langchain_interceptor import capture_scope

    original = Completions.create
    manager = FrameworkOverrideManager()
    manager.activate_overrides(["openai.OpenAI"])
    start = threading.Event()
    opened = threading.Event()
    proceed = threading.Event()
    failures = []
    observed = []

    def lifetime():
        client = _sync_client()
        try:
            assert start.wait(5)
            with capture_scope():
                opened.set()
                assert proceed.wait(5)
                _sync_agent(client)("new concurrent capture")
                observed.extend(get_all_captured_responses())
        except BaseException as exc:
            failures.append(exc)
        finally:
            client.close()

    worker = threading.Thread(target=lifetime)
    worker.start()
    restore = openai_interceptor.restore_openai_capture_original

    def interposed(*args):
        result = restore(*args)
        if args[0] is Completions:
            start.set()
            assert opened.wait(5)
        return result

    try:
        monkeypatch.setattr(
            openai_interceptor, "restore_openai_capture_original", interposed
        )
        manager.deactivate_overrides()
        proceed.set()
        worker.join(5)
        assert not worker.is_alive() and not failures
        assert len(observed) == 1
        assert Completions.create is original
    finally:
        start.set()
        proceed.set()
        worker.join(5)
        manager.deactivate_overrides()


def test_capture_acquired_during_framework_activation_survives_deactivation(
    monkeypatch,
):
    import threading

    from openai.resources.chat.completions import Completions

    from traigent.utils.langchain_interceptor import capture_scope

    original = Completions.create
    manager = FrameworkOverrideManager()
    start = threading.Event()
    opened = threading.Event()
    proceed = threading.Event()
    observed = []
    failures = []

    def lifetime():
        client = _sync_client()
        try:
            assert start.wait(5)
            with capture_scope():
                opened.set()
                assert proceed.wait(5)
                _sync_agent(client)("after interleaved activation and deactivation")
                observed.extend(get_all_captured_responses())
        except BaseException as exc:
            failures.append(exc)
        finally:
            client.close()

    worker = threading.Thread(target=lifetime)
    worker.start()
    create_override = manager._create_override_method

    def interposed(original_method, class_name, method_path):
        wrapped = create_override(original_method, class_name, method_path)
        if class_name == "openai.OpenAI" and method_path == "chat.completions.create":
            start.set()
            assert opened.wait(5)
        return wrapped

    try:
        monkeypatch.setattr(manager, "_create_override_method", interposed)
        manager.activate_overrides(["openai.OpenAI"])
        manager.deactivate_overrides()
        proceed.set()
        worker.join(5)
        assert not worker.is_alive() and not failures
        assert len(observed) == 1
        assert Completions.create is original
    finally:
        start.set()
        proceed.set()
        worker.join(5)
        manager.deactivate_overrides()
