"""Execution and code identity resolve the custom evaluator ONCE per entry.

Decorator evaluator A, runtime override B (distinct recording callables with
different source). Before the fix the cloud path executed A while the identity
named B (``_try_cloud_execution`` passed ``custom_evaluator=None``).
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.api.types import ExampleResult, OptimizationResult, OptimizationStatus
from traigent.config.types import (
    ExecutionIntent,
    ExecutionMode,
    resolve_execution_policy,
)
from traigent.core.execution_policy_runtime import SOURCE_CLOUD_BRAIN
from traigent.core.optimization_pipeline import create_traigent_config
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.artifact_fingerprints import effective_evaluator_code_identity

CALLS = {"A": 0, "B": 0}


def _result(example, tag):
    return ExampleResult(
        example_id="e0",
        input_data=example.input_data,
        expected_output=example.expected_output,
        actual_output="ok",
        metrics={"accuracy": 1.0},
        execution_time=0.0,
        success=True,
        error_message=None,
        metadata={},
    )


def evaluator_a(func, config, example):
    CALLS["A"] += 1  # decorator evaluator
    return _result(example, "A")


def evaluator_b(func, config, example):
    CALLS["B"] += 10  # runtime override (different source from A)
    return _result(example, "B")


def plain_scoring(output, expected):
    return 1.0


def agent(question: str) -> str:
    return "ok"


def _dataset() -> Dataset:
    return Dataset(
        [EvaluationExample(input_data={"question": "q"}, expected_output="ok")],
        name="single-resolution",
    )


def _digest_of(**kwargs) -> str:
    ident = effective_evaluator_code_identity(**kwargs)
    assert ident["origin"] == "code"
    return ident["digest"]


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS["A"] = CALLS["B"] = 0


def _function(
    custom_evaluator=None, scoring_function=None, *, cloud: bool = False
) -> OptimizedFunction:
    extra = {}
    if cloud:
        # Built the way @traigent.optimize() builds the default algorithm="auto".
        extra = {
            "algorithm": "auto",
            "execution_mode": ExecutionMode.HYBRID,
            "execution_policy": resolve_execution_policy(
                algorithm="auto", source_hint="optimize"
            ),
        }
    return OptimizedFunction(
        func=agent,
        configuration_space={"x": [0]},
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        scoring_function=scoring_function,
        max_trials=1,
        **extra,
    )


async def _run_evaluator(evaluator) -> None:
    await evaluator.evaluate(agent, {"x": 0}, _dataset())


# ---------------------------------------------------------------- local path


async def _local(opt: OptimizedFunction, **optimize_kwargs):
    seen: dict = {}
    original = opt._build_optimization_orchestrator

    def spy(**kwargs):
        orchestrator = original(**kwargs)
        seen["evaluator"] = kwargs["evaluator"]
        seen["orchestrator"] = orchestrator
        return orchestrator

    with patch.object(opt, "_build_optimization_orchestrator", side_effect=spy):
        await opt.optimize(
            algorithm="grid", max_trials=1, progress_bar=False, **optimize_kwargs
        )
    return seen


@pytest.mark.asyncio
async def test_local_runtime_override_runs_and_identifies_b() -> None:
    seen = await _local(
        _function(custom_evaluator=evaluator_a), custom_evaluator=evaluator_b
    )
    assert CALLS["B"] >= 10 and CALLS["A"] == 0
    sent = seen["orchestrator"]._session_code_digest_kwargs()
    expected = _digest_of(custom_evaluator=evaluator_b, objectives=["accuracy"])
    assert sent == {"evaluator_code_digest": expected}
    assert expected != _digest_of(custom_evaluator=evaluator_a, objectives=["accuracy"])


@pytest.mark.asyncio
async def test_local_no_override_runs_and_identifies_a() -> None:
    seen = await _local(_function(custom_evaluator=evaluator_a))
    assert CALLS["A"] >= 1 and CALLS["B"] == 0
    assert seen["orchestrator"]._session_code_digest_kwargs() == {
        "evaluator_code_digest": _digest_of(
            custom_evaluator=evaluator_a, objectives=["accuracy"]
        )
    }


@pytest.mark.asyncio
async def test_local_scoring_function_only() -> None:
    seen = await _local(_function(scoring_function=plain_scoring))
    assert seen["orchestrator"]._session_code_digest_kwargs() == {
        "evaluator_code_digest": _digest_of(
            scoring_function=plain_scoring, objectives=["accuracy"]
        )
    }


@pytest.mark.asyncio
async def test_declared_evaluator_id_suppresses_the_digest() -> None:
    opt = _function(custom_evaluator=evaluator_a)
    opt.evaluator_definition_id = "declared-evaluator"
    seen = await _local(opt)
    assert seen["orchestrator"]._session_code_digest_kwargs() == {}


# --------------------------------------------------------------- cloud path


async def _cloud(opt: OptimizedFunction, monkeypatch, **optimize_kwargs):
    """Enter the cloud-brain branch with stubbed transport; return what it built."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg-fake-portal-key-single-resolution")
    seen: dict = {}

    def build(**kwargs):
        seen["evaluator"] = kwargs["evaluator"]
        orchestrator = MagicMock()
        return orchestrator

    result = OptimizationResult(
        trials=[],
        best_config={},
        best_score=0.0,
        optimization_id="x",
        duration=0.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="auto",
        timestamp=__import__("datetime").datetime.now(__import__("datetime").UTC),
        metadata={},
    )
    config_factory = create_traigent_config

    def cloud_config(**kwargs):
        kwargs["execution_policy"] = opt.execution_policy
        kwargs["result_source"] = SOURCE_CLOUD_BRAIN
        return config_factory(**kwargs)

    with (
        patch(
            "traigent.optimizers.interactive_optimizer.InteractiveOptimizer"
        ) as interactive,
        patch("traigent.cloud.client.TraigentCloudClient"),
        patch.object(opt, "_build_optimization_orchestrator", side_effect=build),
        patch.object(
            opt, "_run_and_finalize_optimization", AsyncMock(return_value=result)
        ),
        patch("traigent.core.optimized_function.create_traigent_config", cloud_config),
    ):
        assert opt.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN
        await opt.optimize(max_trials=1, progress_bar=False, **optimize_kwargs)
    seen["interactive_kwargs"] = interactive.call_args.kwargs
    return seen


@pytest.mark.asyncio
async def test_cloud_runtime_override_runs_and_identifies_b(monkeypatch) -> None:
    opt = _function(custom_evaluator=evaluator_a, cloud=True)
    seen = await _cloud(opt, monkeypatch, custom_evaluator=evaluator_b)
    await _run_evaluator(seen["evaluator"])
    assert CALLS["B"] >= 10 and CALLS["A"] == 0
    sent = seen["interactive_kwargs"]["evaluator_code_digest"]
    assert sent == _digest_of(custom_evaluator=evaluator_b, objectives=["accuracy"])
    assert sent != _digest_of(custom_evaluator=evaluator_a, objectives=["accuracy"])


@pytest.mark.asyncio
async def test_cloud_no_override_runs_and_identifies_a(monkeypatch) -> None:
    opt = _function(custom_evaluator=evaluator_a, cloud=True)
    seen = await _cloud(opt, monkeypatch)
    await _run_evaluator(seen["evaluator"])
    assert CALLS["A"] >= 1 and CALLS["B"] == 0
    assert seen["interactive_kwargs"]["evaluator_code_digest"] == _digest_of(
        custom_evaluator=evaluator_a, objectives=["accuracy"]
    )


@pytest.mark.asyncio
async def test_cloud_scoring_function_only(monkeypatch) -> None:
    opt = _function(scoring_function=plain_scoring, cloud=True)
    seen = await _cloud(opt, monkeypatch)
    assert seen["interactive_kwargs"]["evaluator_code_digest"] == _digest_of(
        scoring_function=plain_scoring, objectives=["accuracy"]
    )


@pytest.mark.asyncio
async def test_cloud_privacy_mode_withholds_the_digest(monkeypatch) -> None:
    opt = _function(custom_evaluator=evaluator_a, cloud=True)
    opt.privacy_enabled = True
    seen = await _cloud(opt, monkeypatch)
    assert seen["interactive_kwargs"]["evaluator_code_digest"] is None


# ------------------------------------------------------------- static guards


def test_resolver_precedence_and_identity_helper_reads_no_instance_state() -> None:
    opt = _function(custom_evaluator=evaluator_a)
    assert opt._resolve_effective_custom_evaluator(None) is evaluator_a
    assert opt._resolve_effective_custom_evaluator(evaluator_b) is evaluator_b
    assert _function()._resolve_effective_custom_evaluator(None) is None
    assert "self.custom_evaluator" not in inspect.getsource(
        effective_evaluator_code_identity
    )
