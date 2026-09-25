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
        api_key="test-key",
        base_url="https://gateway.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler(usage=usage))),
        max_retries=0,
    )


def _async_client(*, usage: bool = True) -> Any:
    return openai.AsyncOpenAI(
        api_key="test-key",
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
async def test_uncaptured_client_is_unmeasured_not_zero():
    # No override active: nothing can be captured. The aggregate must say
    # "unmeasured" rather than upload $0 / 0 tokens.
    client = _sync_client()
    result = await _evaluator().evaluate(
        _sync_agent(client), {"model": KNOWN_MODEL}, _dataset()
    )
    assert not any(key in result.aggregated_metrics for key in LLM_KEYS)


@pytest.mark.asyncio
async def test_strict_metrics_nulls_reports_unmeasured_as_none(monkeypatch):
    monkeypatch.setenv("TRAIGENT_STRICT_METRICS_NULLS", "true")
    client = _sync_client()
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


def test_optimize_without_capture_does_not_upload_zero_cost(monkeypatch, tmp_path):
    result = _optimize(monkeypatch, tmp_path, framework_targets=None)
    (trial,) = result.trials
    assert trial.metrics.get("total_cost") is None
    assert trial.metrics.get("total_tokens") is None


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
# Review F2 (owner decision: conservative stop, with an actionable message)
# --------------------------------------------------------------------------


def _optimize_fifteen(monkeypatch, tmp_path, *, measured: bool, **optimize_kwargs):
    """15 configurations; the agent's usage is captured only when ``measured``."""
    from traigent.core.optimized_function import OptimizedFunction

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    client = _sync_client()

    def agent(question: str) -> str:
        if not measured:
            return "ok"  # e.g. an HTTP call no interceptor sees
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
        configuration_space={"temperature": [i / 10 for i in range(15)]},
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        max_trials=15,
        auto_override_frameworks=measured,
        framework_targets=["openai.OpenAI"] if measured else None,
    )
    return opt_func.optimize_sync(
        algorithm="grid", max_trials=15, progress_bar=False, **optimize_kwargs
    )


def test_unmeasured_run_stops_at_fallback_limit_and_says_how_to_continue(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)
    result = _optimize_fifteen(monkeypatch, tmp_path, measured=False)

    assert len(result.trials) == 10
    assert result.stop_reason == "cost_limit"
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" in result.warning_codes
    (message,) = [w for w in result.warnings if "could not be measured" in w]
    for remedy in (
        "TRAIGENT_FALLBACK_TRIAL_LIMIT",
        "cost_limit=",
        "TRAIGENT_RUN_COST_LIMIT",
        "docs/user-guide/cost_capture.md",
        "enable_openai_optimization()",
    ):
        assert remedy in message
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" in caplog.text


def test_raising_the_fallback_limit_lets_an_unmeasured_run_continue(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", "20")
    result = _optimize_fifteen(monkeypatch, tmp_path, measured=False)

    assert len(result.trials) == 15
    assert result.stop_reason != "cost_limit"
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" not in result.warning_codes


def test_measured_run_is_bounded_by_its_cost_budget_not_the_trial_fallback(
    monkeypatch, tmp_path
):
    # Once cost is measured the cost budget governs: 15 cheap trials fit a
    # $1 budget, so the 10-trial fallback does not apply.
    monkeypatch.delenv("TRAIGENT_FALLBACK_TRIAL_LIMIT", raising=False)
    result = _optimize_fifteen(monkeypatch, tmp_path, measured=True, cost_limit=1.0)

    assert len(result.trials) == 15
    assert result.stop_reason != "cost_limit"
    assert "COST_UNMEASURED_TRIAL_LIMIT_REACHED" not in result.warning_codes
