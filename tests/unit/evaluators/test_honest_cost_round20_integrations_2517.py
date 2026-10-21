"""Round 20 (#2517): integration findings from the full-branch review.

* an unmeasured judge call makes the example unmeasured (agent spend = floor);
* ``cost.unpriced`` usage is not a measured cost, and the LangChain and
  PydanticAI handlers do not accept an unknown-model estimator zero;
* an incomplete (partial) Langfuse observation fetch is never a complete total;
* the local-search example labels simulated costs whenever mock mode is on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from traigent.api.types import ExampleResult
from traigent.evaluators import local as local_module
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import ExampleMetrics, MetricsTracker
from traigent.integrations.langfuse.client import LangfuseClient
from traigent.integrations.langchain.handler import TraigentHandler
from traigent.integrations.pydantic_ai import handler as pydantic_ai_handler
from traigent.integrations.pydantic_ai.handler import PydanticAIHandler

REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# 1. Judge uncertainty propagates into the example                             #
# --------------------------------------------------------------------------- #
class _Usage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _PricedJudgeResponse:
    model = "gpt-4o-mini"
    usage = _Usage(20, 10)


class _UsagelessJudgeResponse:
    """Nothing extractable: no usage, no cost, no model."""


def _priced_agent_row(cost: float = 0.05) -> ExampleMetrics:
    row = ExampleMetrics(measured=True)
    row.tokens.input_tokens = 100
    row.tokens.output_tokens = 10
    row.cost.input_cost = cost
    row.cost.total_cost = cost
    row.custom_metrics["total_cost"] = cost
    return row


def _fold(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[object],
    row: ExampleMetrics,
    result: ExampleResult | None = None,
) -> None:
    monkeypatch.setattr(
        local_module, "drain_captured_responses", lambda: (responses, None, None)
    )
    LocalEvaluator(metrics=["cost"])._fold_metric_function_llm_cost(
        row, example_result=result
    )


def test_unmeasured_judge_makes_a_priced_agent_example_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _priced_agent_row(0.05)
    result = ExampleResult(
        "e", {}, None, None, {"total_cost": 0.05, "cost": 0.05}, 0.0, True
    )

    _fold(monkeypatch, [_UsagelessJudgeResponse()], row, result)

    assert row.cost.unmeasured is True
    assert row.cost_measured is False
    assert row.cost.known_cost == pytest.approx(0.05)  # agent spend kept as a floor
    assert "total_cost" not in row.custom_metrics
    assert "total_cost" not in result.metrics
    assert result.metrics["cost_unmeasured"] == 1.0
    assert result.metrics["cost_lower_bound"] == pytest.approx(0.05)

    tracker = MetricsTracker()
    tracker.example_metrics.append(row)
    formatted = tracker.format_for_backend()
    assert formatted["cost"] is None
    assert formatted["cost_unmeasured"] == 1.0
    assert formatted["cost_lower_bound"] == pytest.approx(0.05)


def test_priced_judge_plus_unmeasured_judge_keeps_both_as_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _priced_agent_row(0.05)
    _fold(monkeypatch, [_PricedJudgeResponse(), _UsagelessJudgeResponse()], row)

    assert row.cost.unmeasured is True
    assert row.cost.known_cost > 0.05  # agent floor + the priced judge call


def test_fully_measured_judge_keeps_the_example_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _priced_agent_row(0.05)
    _fold(monkeypatch, [_PricedJudgeResponse()], row)

    assert row.cost.unmeasured is False
    assert row.cost_measured is True
    assert row.cost.total_cost > 0.05


# --------------------------------------------------------------------------- #
# 2. Unpriced usage is not a measured cost                                     #
# --------------------------------------------------------------------------- #
def test_unpriced_example_is_not_a_measured_cost() -> None:
    row = ExampleMetrics(measured=True)
    row.tokens.input_tokens = 50
    row.cost.unpriced = True  # tokens captured, no price table covers the model

    assert row.cost_measured is False

    tracker = MetricsTracker()
    tracker.example_metrics.append(row)
    formatted = tracker.format_for_backend()
    assert formatted["cost"] is None
    assert formatted["cost_unmeasured"] == 1.0


def test_priced_part_of_an_unpriced_example_is_kept_as_floor() -> None:
    row = ExampleMetrics(measured=True)
    row.cost.total_cost = 0.07
    row.cost.unpriced = True
    tracker = MetricsTracker()
    tracker.example_metrics.append(row)

    assert tracker.format_for_backend()["cost_lower_bound"] == pytest.approx(0.07)


def _no_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")
    monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "false")


def test_langchain_unknown_model_zero_is_unmeasured_even_when_strict_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_strict(monkeypatch)
    handler = TraigentHandler()
    run_id = str(uuid4())
    handler.on_llm_start(
        serialized={"kwargs": {"model": "acme-unpriced-model-9000"}},
        prompts=["hi"],
        run_id=run_id,
    )
    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "model_name": "acme-unpriced-model-9000",
        },
        generations=[],
    )
    handler.on_llm_end(response, run_id=run_id)

    metrics = handler.get_metrics()
    assert metrics.llm_calls[0].cost_unmeasured is True
    assert metrics.cost_unmeasured is True
    measures = handler.get_measures_dict()
    assert measures["cost_unmeasured"] == 1.0
    assert "total_cost" not in measures


def test_pydantic_ai_unknown_model_zero_is_unmeasured_even_when_strict_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_strict(monkeypatch)
    monkeypatch.setattr(pydantic_ai_handler, "PYDANTICAI_AVAILABLE", True)
    agent = SimpleNamespace(model="acme:unpriced-model-9000")
    handler = PydanticAIHandler(agent)
    result = SimpleNamespace(
        usage=lambda: SimpleNamespace(
            input_tokens=100, output_tokens=50, total_tokens=150, requests=1
        )
    )
    handler._record_usage(result, 0.0)

    measures = handler.get_measures_dict()
    assert measures["pydantic_ai_cost_unmeasured"] == 1.0
    assert "pydantic_ai_total_cost" not in measures


# --------------------------------------------------------------------------- #
# 3. Partial Langfuse observations never claim a complete total                #
# --------------------------------------------------------------------------- #
def _generation(**fields: Any) -> dict[str, Any]:
    return {"id": "o", "name": "gen", "type": "GENERATION", "model": "m", **fields}


@pytest.mark.parametrize("observations", [[], [_generation(calculatedTotalCost=0.25)]])
def test_partial_observations_are_never_a_complete_total(observations) -> None:
    client = LangfuseClient.__new__(LangfuseClient)
    metrics = client._extract_metrics_from_trace(
        {"id": "t", "observations": observations, "observations_partial": True}
    )

    assert metrics.cost_unmeasured is True
    measures = metrics.to_measures_dict(prefix="lf_")
    assert "lf_total_cost" not in measures
    assert measures["lf_cost_unmeasured"] == 1.0


def test_complete_observations_stay_a_measured_total() -> None:
    client = LangfuseClient.__new__(LangfuseClient)
    metrics = client._extract_metrics_from_trace(
        {"id": "t", "observations": [_generation(calculatedTotalCost=0.25)]}
    )

    assert metrics.cost_unmeasured is False
    assert metrics.to_measures_dict()["total_cost"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# 4. Local-search example: simulated-cost label follows mock mode              #
# --------------------------------------------------------------------------- #
def _load_local_search() -> Any:
    path = (
        REPO
        / "examples/advanced/optimization-routing/ex01-local-search/local_search.py"
    )
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("_r20_local_search", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_simulated_cost_label_follows_mock_mode_not_the_langchain_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_local_search()
    module._USING_FALLBACK_LLM = False  # langchain_openai imported fine

    monkeypatch.setattr("traigent.utils.env_config.is_mock_llm", lambda: True)
    assert module.simulated_cost_provenance() == {"cost_source": "simulated_estimate"}
    module.print_simulated_cost_notice()
    assert module.SIMULATED_COST_NOTICE in capsys.readouterr().out

    monkeypatch.setattr("traigent.utils.env_config.is_mock_llm", lambda: False)
    assert module.simulated_cost_provenance() == {}
    module.print_simulated_cost_notice()
    assert module.SIMULATED_COST_NOTICE not in capsys.readouterr().out
