"""Round 21 (#2517): second sol review of the full branch.

* judge uncertainty applies whatever the agent's own measured state is;
* withdrawing cost keys for an unmeasured judge does not break the detailed
  results transfer;
* "unpriced" is decided by pricing LOOKUP success, not the dollar amount (an
  explicit custom rate of 0 is a measured $0);
* a trial whose only failed rows report ``provider_attempts: 0`` materializes
  a measured zero all the way through an explicit cost limit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from traigent.api.types import ExampleResult
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators import local as local_module
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import ExampleMetrics
from traigent.integrations.langchain.handler import TraigentHandler
from traigent.integrations.pydantic_ai import handler as pydantic_ai_handler
from traigent.integrations.pydantic_ai.handler import PydanticAIHandler


class _Usage:
    prompt_tokens = 20
    completion_tokens = 10
    total_tokens = 30


class _PricedJudge:
    model = "gpt-4o-mini"
    usage = _Usage()


class _UsagelessJudge:
    """No usage, no cost, no model."""


# --------------------------------------------------------------------------- #
# 1. Judge uncertainty is independent of the agent's measured state            #
# --------------------------------------------------------------------------- #
def _fold(monkeypatch, responses, row, result=None) -> None:
    monkeypatch.setattr(
        local_module, "drain_captured_responses", lambda: (responses, None, None)
    )
    LocalEvaluator(metrics=["cost"])._fold_metric_function_llm_cost(
        row, example_result=result
    )


def test_unmeasured_agent_row_with_priced_and_usageless_judge_stays_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = ExampleMetrics(measured=False)  # the agent raised before any output

    _fold(monkeypatch, [_PricedJudge(), _UsagelessJudge()], row)

    assert row.cost.unmeasured is True
    assert row.cost_measured is False
    assert row.cost.known_cost > 0  # the priced judge call is kept as a floor


def test_unmeasured_agent_row_with_only_priced_judge_is_still_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = ExampleMetrics(measured=False)

    _fold(monkeypatch, [_PricedJudge()], row)

    assert row.cost.unmeasured is False
    assert row.cost_measured is True
    assert row.cost.total_cost > 0


# --------------------------------------------------------------------------- #
# 2. The detailed-results transfer survives withdrawn cost keys                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["cost", "total_cost"])
async def test_detailed_results_survive_a_cost_named_metric_and_unmeasured_judge(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setattr(
        local_module,
        "drain_captured_responses",
        lambda: ([_UsagelessJudge()], None, None),
    )

    def judge(output, expected, llm_metrics=None) -> float:
        return 0.1

    async def agent(text: str, **config):
        return {"text": "ok"}

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"text": "q"}, expected_output="ok")],
        name="r21",
    )
    evaluator = LocalEvaluator(
        metrics=["cost"], metric_functions={name: judge}, detailed=True
    )

    result = await evaluator.evaluate(agent, {}, dataset)  # must not KeyError

    row = result.example_results[0]
    assert row.metrics.get("cost_unmeasured") == 1.0
    assert "total_cost" not in row.metrics and "cost" not in row.metrics


# --------------------------------------------------------------------------- #
# 3. Unpriced = lookup failure, not a zero amount                              #
# --------------------------------------------------------------------------- #
_FREE = "acme-free-model"
_UNKNOWN = "acme-unpriced-model-9000"


def _free_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "false")
    monkeypatch.setenv(
        "TRAIGENT_CUSTOM_MODEL_PRICING_JSON",
        json.dumps({_FREE: {"input": 0.0, "output": 0.0}}),
    )
    from traigent.utils import cost_calculator as cc

    for name in ("_CUSTOM_PRICING_INDEX", "_custom_pricing_index"):
        if hasattr(cc, name):
            monkeypatch.setattr(cc, name, None)


def _langchain_measures(model: str) -> dict:
    handler = TraigentHandler()
    run_id = str(uuid4())
    handler.on_llm_start(
        serialized={"kwargs": {"model": model}}, prompts=["hi"], run_id=run_id
    )
    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "model_name": model,
        },
        generations=[],
    )
    handler.on_llm_end(response, run_id=run_id)
    return handler.get_measures_dict()


def _pydantic_measures(model: str) -> dict:
    handler = PydanticAIHandler(SimpleNamespace(model=f"acme:{model}"))
    handler._model_name = model
    result = SimpleNamespace(
        usage=lambda: SimpleNamespace(
            input_tokens=100, output_tokens=50, total_tokens=150, requests=1
        )
    )
    handler._record_usage(result, 0.0)
    return handler.get_measures_dict()


def test_langchain_explicit_zero_rate_is_a_measured_zero(monkeypatch) -> None:
    _free_pricing(monkeypatch)
    measures = _langchain_measures(_FREE)
    assert measures["total_cost"] == 0.0
    assert "cost_unmeasured" not in measures


def test_langchain_unknown_model_is_still_unmeasured(monkeypatch) -> None:
    _free_pricing(monkeypatch)
    measures = _langchain_measures(_UNKNOWN)
    assert measures["cost_unmeasured"] == 1.0
    assert "total_cost" not in measures


def test_pydantic_ai_explicit_zero_rate_is_a_measured_zero(monkeypatch) -> None:
    _free_pricing(monkeypatch)
    monkeypatch.setattr(pydantic_ai_handler, "PYDANTICAI_AVAILABLE", True)
    measures = _pydantic_measures(_FREE)
    assert measures["pydantic_ai_total_cost"] == 0.0
    assert "pydantic_ai_cost_unmeasured" not in measures


def test_pydantic_ai_unknown_model_is_still_unmeasured(monkeypatch) -> None:
    _free_pricing(monkeypatch)
    monkeypatch.setattr(pydantic_ai_handler, "PYDANTICAI_AVAILABLE", True)
    measures = _pydantic_measures(_UNKNOWN)
    assert measures["pydantic_ai_cost_unmeasured"] == 1.0
    assert "pydantic_ai_total_cost" not in measures


# --------------------------------------------------------------------------- #
# 4. provider_attempts: 0 materializes a measured zero end to end              #
# --------------------------------------------------------------------------- #
def test_only_zero_attempt_failed_rows_complete_with_cost_zero_under_explicit_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    monkeypatch.delenv("TRAIGENT_REQUIRE_COST_TRACKING", raising=False)
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "5.0")  # explicit limit
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))

    def target(value: int, **_config) -> int:
        return value

    def custom_evaluator(func, config, example):
        return ExampleResult(
            example_id=f"row-{example.input_data['value']}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=None,
            metrics={"accuracy": 0.0, "provider_attempts": 0},
            execution_time=0.0,
            success=False,
            error_message="failed before any provider call",
            metadata={},
        )

    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(2)
        ],
        name="r21-zero-attempts",
    )
    opt_func = OptimizedFunction(
        func=target,
        configuration_space={"model": ["a"]},
        objectives=["accuracy"],
        eval_dataset=dataset,
        custom_evaluator=custom_evaluator,
        max_trials=1,
    )
    result = opt_func.optimize_sync(algorithm="grid", max_trials=1, progress_bar=False)

    trial = result.trials[0]
    assert not trial.metrics.get("cost_unmeasured")
    assert trial.metrics.get("total_cost") == 0.0
    assert datetime.now(UTC)  # (keeps the import used)


# --------------------------------------------------------------------------- #
# Round 3 review: metadata-only pricing entries; zero-attempt rows keep spend   #
# --------------------------------------------------------------------------- #
def test_metadata_only_litellm_entry_is_unpriced_not_free(monkeypatch) -> None:
    import litellm

    from traigent.utils import cost_calculator as cc

    _free_pricing(monkeypatch)
    entry = "acme/metadata-only-model"
    monkeypatch.setitem(litellm.model_cost, entry, {"litellm_provider": "acme"})
    monkeypatch.setattr(cc, "_is_model_known_to_litellm", lambda m: m == entry)

    assert cc.model_pricing_resolves(entry) is False
    assert cc.model_pricing_resolves(_FREE) is True  # explicit custom 0 rates
    assert cc.model_pricing_resolves("gpt-4o-mini") is True  # normally priced
    assert cc.model_pricing_resolves(_UNKNOWN) is False


def _wrapper_aggregate(rows, successes):
    from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper

    results = [
        ExampleResult(f"e{i}", {}, None, None, dict(r), 0.0, ok)
        for i, (r, ok) in enumerate(zip(rows, successes, strict=True))
    ]
    wrapper = object.__new__(CustomEvaluatorWrapper)
    return wrapper._aggregate_llm_metrics([r.metrics for r in results], results)


def test_zero_attempt_row_does_not_overwrite_spend_reported_as_cost() -> None:
    aggregated = _wrapper_aggregate(
        [{"cost": 0.25}, {"provider_attempts": 0}], [True, False]
    )
    assert not aggregated.get("cost_unmeasured")
    assert aggregated["total_cost"] == pytest.approx(0.25)


def test_all_zero_attempt_rows_materialize_zero() -> None:
    aggregated = _wrapper_aggregate(
        [{"provider_attempts": 0}, {"provider_attempts": 0}], [False, False]
    )
    assert aggregated["total_cost"] == 0.0


def test_mixed_aliases_with_zero_attempt_row_sum_all_spend() -> None:
    aggregated = _wrapper_aggregate(
        [{"total_cost": 0.10}, {"cost": 0.25}, {"provider_attempts": 0}],
        [True, True, False],
    )
    assert not aggregated.get("cost_unmeasured")
    assert aggregated["total_cost"] == pytest.approx(0.35)


def test_row_echoing_spend_under_two_aliases_counts_once() -> None:
    aggregated = _wrapper_aggregate(
        [{"cost": 0.25, "total_cost": 0.25}, {"provider_attempts": 0}],
        [True, False],
    )
    assert aggregated["total_cost"] == pytest.approx(0.25)
