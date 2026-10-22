"""Per-call ``eval_dataset`` on ``.optimize()`` (#2541).

The decorator's dataset stays the default. ``optimize(eval_dataset=...)`` swaps
in one dataset for that call only; it is loaded with the decorator path's own
validation and loader, restored in ``finally``, and reaches the cloud/hybrid
path exactly as the decorator dataset does (no new wire field).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from traigent.api.types import OptimizationResult, OptimizationStatus, TrialResult
from traigent.core.optimized_function import OptimizedFunction
from traigent.core.resume import FINGERPRINTS_METADATA_KEY, plan_resume
from traigent.evaluators.base import Dataset

from .test_fixtures import create_test_dataset


def _mock_result(objectives: list[str]) -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config={"temperature": 0.5},
        best_score=0.9,
        optimization_id="calltime-2541",
        duration=0.1,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives,
        algorithm="grid",
        timestamp=datetime.now(UTC),
        metadata={},
    )


@pytest.fixture
def decorator_dataset() -> Dataset:
    return create_test_dataset(3)


@pytest.fixture
def call_dataset() -> Dataset:
    return create_test_dataset(5)


@pytest.fixture
def opt_func(
    simple_function, sample_config_space, sample_objectives, decorator_dataset
) -> OptimizedFunction:
    return OptimizedFunction(
        func=simple_function,
        configuration_space=sample_config_space,
        objectives=sample_objectives,
        eval_dataset=decorator_dataset,
        max_trials=2,
    )


def _seen_datasets(opt_func: OptimizedFunction):
    """Run ``optimize`` with the orchestrator mocked; return the dataset used.

    The dataset the run uses is the one handed to the artifact fingerprint
    payload in ``_execute_optimization``, so spying there records the real
    dataset the run evaluates without touching the optimizer.
    """
    return patch.object(
        opt_func,
        "_build_artifact_fingerprint_payload",
        wraps=opt_func._build_artifact_fingerprint_payload,
    )


class TestPerCallDatasetIsEvaluated:
    @pytest.mark.asyncio
    async def test_per_call_dataset_is_the_one_run(
        self, opt_func, call_dataset, sample_objectives
    ):
        orchestrator_optimize = AsyncMock(return_value=_mock_result(sample_objectives))
        with (
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as orchestrator_cls,
            _seen_datasets(opt_func) as spy,
        ):
            orchestrator_cls.return_value.optimize = orchestrator_optimize
            await opt_func.optimize(eval_dataset=call_dataset)

        used = spy.call_args.kwargs["dataset"]
        assert len(used.examples) == len(call_dataset.examples) == 5
        assert [e.input_data for e in used.examples] == [
            e.input_data for e in call_dataset.examples
        ]

    @pytest.mark.asyncio
    async def test_decorator_dataset_used_again_on_next_call(
        self, opt_func, call_dataset, decorator_dataset, sample_objectives
    ):
        orchestrator_optimize = AsyncMock(return_value=_mock_result(sample_objectives))
        with (
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as orchestrator_cls,
            _seen_datasets(opt_func) as spy,
        ):
            orchestrator_cls.return_value.optimize = orchestrator_optimize
            await opt_func.optimize(eval_dataset=call_dataset)
            await opt_func.optimize()

        first, second = (c.kwargs["dataset"] for c in spy.call_args_list)
        assert len(first.examples) == 5
        assert len(second.examples) == len(decorator_dataset.examples) == 3
        assert opt_func._dataset_override is None


class TestOverrideIsScopedToTheCall:
    @pytest.mark.asyncio
    async def test_override_restored_after_the_call_raises(
        self, opt_func, call_dataset, sample_objectives
    ):
        seen_during_run: list[object] = []

        async def _boom(*_args, **_kwargs):
            seen_during_run.append(opt_func._dataset_override)
            raise RuntimeError("run failed")

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as orchestrator_cls:
            orchestrator_cls.return_value.optimize = AsyncMock(side_effect=_boom)
            with pytest.raises(Exception):  # noqa: B017 - any run failure
                await opt_func.optimize(eval_dataset=call_dataset)

        assert seen_during_run and seen_during_run[0] is not None
        assert opt_func._dataset_override is None

    @pytest.mark.asyncio
    async def test_preset_guided_override_is_restored(
        self, opt_func, call_dataset, sample_objectives
    ):
        guided = create_test_dataset(7)
        opt_func.set_eval_dataset_override(guided)

        with patch(
            "traigent.core.optimized_function.OptimizationOrchestrator"
        ) as orchestrator_cls:
            orchestrator_cls.return_value.optimize = AsyncMock(
                return_value=_mock_result(sample_objectives)
            )
            await opt_func.optimize(eval_dataset=call_dataset)

        assert opt_func._dataset_override is guided

    @pytest.mark.asyncio
    async def test_bad_dataset_leaves_no_override_behind(
        self, opt_func, sample_objectives
    ):
        opt_func.set_eval_dataset_override(None)
        with pytest.raises(Exception):  # noqa: B017 - loader error, checked below
            await opt_func.optimize(eval_dataset=[{"not_an_example": 1}])
        assert opt_func._dataset_override is None


class TestResumeDoesNotReuseAcrossDatasets:
    def test_fingerprints_differ_and_resume_reuses_nothing(
        self, opt_func, decorator_dataset, call_dataset
    ):
        config_space = opt_func.configuration_space
        fps_decorator = opt_func._build_artifact_fingerprint_payload(
            dataset=decorator_dataset,
            configuration_space=config_space,
            custom_evaluator=None,
        )["artifact_fingerprints"]
        fps_call = opt_func._build_artifact_fingerprint_payload(
            dataset=call_dataset,
            configuration_space=config_space,
            custom_evaluator=None,
        )["artifact_fingerprints"]
        assert fps_decorator["dataset"] != fps_call["dataset"]

        prior = OptimizationResult(
            trials=[
                TrialResult(
                    trial_id="t1",
                    config={"temperature": 0.5},
                    metrics={"accuracy": 1.0, "total_cost": 0.5},
                    status="completed",
                    duration=0.1,
                    timestamp=datetime.now(UTC),
                )
            ],
            best_config=None,
            best_score=None,
            optimization_id="prior-2541",
            duration=1.0,
            convergence_info={},
            status="completed",
            objectives=["accuracy"],
            algorithm="grid",
            timestamp=datetime.now(UTC),
            metadata={FINGERPRINTS_METADATA_KEY: fps_decorator},
        )
        plan = plan_resume(prior, fps_call)
        assert plan.reusable == ()
        assert "dataset" in plan.mismatched_components


class TestBadDatasetMatchesDecorator:
    def test_same_error_as_at_decoration_time(
        self, simple_function, sample_config_space, sample_objectives
    ):
        bad = [{"not_an_example": 1}]
        with pytest.raises(Exception) as at_decorator:  # noqa: B017
            OptimizedFunction(
                func=simple_function,
                configuration_space=sample_config_space,
                objectives=sample_objectives,
                eval_dataset=bad,
            )
        good_wrapper = OptimizedFunction(
            func=simple_function,
            configuration_space=sample_config_space,
            objectives=sample_objectives,
            eval_dataset=create_test_dataset(3),
        )
        with pytest.raises(Exception) as at_call:  # noqa: B017
            good_wrapper._dataset_for_call(bad)

        assert type(at_call.value) is type(at_decorator.value)
        assert str(at_call.value) == str(at_decorator.value)


def test_eval_dataset_is_a_declared_parameter_not_algorithm_kwarg():
    import inspect

    params = inspect.signature(OptimizedFunction.optimize).parameters
    assert "eval_dataset" in params
    assert params["eval_dataset"].default is None
