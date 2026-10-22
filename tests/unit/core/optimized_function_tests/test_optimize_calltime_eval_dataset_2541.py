"""Per-call ``eval_dataset`` on ``.optimize()`` (#2541).

The decorator's dataset stays the default. ``optimize(eval_dataset=...)`` swaps
in one dataset for that call only; it is loaded with the decorator path's own
validation and loader, restored in ``finally``, and reaches the cloud/hybrid
path exactly as the decorator dataset does (no new wire field).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions
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
    async def test_per_call_dataset_never_touches_shared_override(
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

        # The per-call dataset travels as call-local state (#2541 review r84):
        # the shared override stays unset during the run and after it raises.
        assert seen_during_run == [None]
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


class TestConcurrentOverrideDoesNotLeakIntoRun:
    @pytest.mark.asyncio
    async def test_shared_override_cleared_mid_run_still_evaluates_call_dataset(
        self, opt_func, call_dataset, sample_objectives
    ):
        """Deterministic interleaving: guidance writes and clears the shared
        override while the per-call run is in flight, before the dataset load."""
        guided = create_test_dataset(7)
        seen: list[Dataset] = []

        def _interleave(*_args, **_kwargs):
            opt_func.set_eval_dataset_override(guided)
            opt_func.set_eval_dataset_override(None)

        with (
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as orchestrator_cls,
            patch.object(opt_func, "_check_ci_approval", side_effect=_interleave),
            patch.object(
                opt_func,
                "_build_artifact_fingerprint_payload",
                wraps=opt_func._build_artifact_fingerprint_payload,
            ) as spy,
        ):
            orchestrator_cls.return_value.optimize = AsyncMock(
                return_value=_mock_result(sample_objectives)
            )
            await opt_func.optimize(eval_dataset=call_dataset)
            seen.extend(c.kwargs["dataset"] for c in spy.call_args_list)

        assert len(seen) == 1
        assert seen[0] is call_dataset

    @pytest.mark.asyncio
    async def test_run_awaiting_orchestrator_is_unaffected_by_cleanup(
        self, opt_func, call_dataset, sample_objectives
    ):
        release = asyncio.Event()
        entered = asyncio.Event()

        async def _wait(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return _mock_result(sample_objectives)

        with (
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as orchestrator_cls,
            patch.object(
                opt_func,
                "_build_artifact_fingerprint_payload",
                wraps=opt_func._build_artifact_fingerprint_payload,
            ) as spy,
        ):
            orchestrator_cls.return_value.optimize = AsyncMock(side_effect=_wait)
            task = asyncio.ensure_future(opt_func.optimize(eval_dataset=call_dataset))
            await entered.wait()
            opt_func.set_eval_dataset_override(None)  # guided cleanup while awaiting
            release.set()
            await task

        assert spy.call_args.kwargs["dataset"] is call_dataset


class TestGuidanceSeedsFromCallDataset:
    def test_seeds_from_call_dataset_and_rounds_use_grown_dataset(
        self, opt_func, call_dataset, sample_objectives
    ):
        seeded: list[Dataset] = []

        class _FakeLoop:
            def __init__(self, **_kwargs):
                pass

            def run(self, *, optimize_round, config_space, dataset, **_kwargs):
                seeded.append(dataset)
                optimize_round(config_space, dataset)
                grown = Dataset(
                    examples=[*dataset.examples, dataset.examples[0]],
                    name="grown",
                    description="grown by one example",
                )
                optimize_round(config_space, grown)
                return SimpleNamespace(best_result=None)

        def _record(*_args, **_kwargs):
            return _mock_result(sample_objectives)

        with (
            patch("traigent.generation.GuidanceLoop", _FakeLoop),
            patch("traigent.generation.resolve_rewrite_llm", return_value=object()),
            patch("traigent.generation.ExampleSynthesizer", MagicMock()),
            patch("traigent.generation.PromptRewriter", MagicMock()),
            patch(
                "traigent.core.optimized_function.OptimizationOrchestrator"
            ) as orchestrator_cls,
            patch.object(
                opt_func,
                "_build_artifact_fingerprint_payload",
                wraps=opt_func._build_artifact_fingerprint_payload,
            ) as spy,
        ):
            orchestrator_cls.return_value.optimize = AsyncMock(side_effect=_record)
            opt_func.optimize_with_guidance(
                object(), eval_dataset=call_dataset, max_trials=1
            )

        assert seeded == [call_dataset]
        round_sizes = [len(c.kwargs["dataset"].examples) for c in spy.call_args_list]
        assert round_sizes == [5, 6]
        assert opt_func._dataset_override is None


class TestResumeAcrossDatasetsPublicApi:
    @pytest.fixture
    def _offline(self, tmp_path: Path, monkeypatch) -> Path:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
        monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
        (tmp_path / "a.jsonl").write_text(
            '{"input":{"q":"a1"},"output":"x"}\n{"input":{"q":"a2"},"output":"x"}\n'
        )
        (tmp_path / "b.jsonl").write_text(
            '{"input":{"q":"b1"},"output":"x"}\n'
            '{"input":{"q":"b2"},"output":"x"}\n'
            '{"input":{"q":"b3"},"output":"x"}\n'
        )
        return tmp_path

    def test_resume_from_a_run_on_a_evaluates_all_of_b(self, _offline: Path) -> None:
        seen: list[str] = []

        @traigent.optimize(
            configuration_space={"v": ["a"]},
            objectives=["accuracy"],
            offline=True,
            max_trials=1,
            algorithm="grid",
            evaluation=EvaluationOptions(eval_dataset=str(_offline / "a.jsonl")),
        )
        def agent(q: str) -> str:
            seen.append(q)
            return traigent.get_config()["v"]

        first = agent.optimize_sync(max_trials=1)
        assert sorted(seen) == ["a1", "a2"]
        seen.clear()

        agent.optimize_sync(
            max_trials=1,
            resume_from=first,
            eval_dataset=str(_offline / "b.jsonl"),
        )
        assert sorted(seen) == ["b1", "b2", "b3"]


def test_eval_dataset_is_a_declared_parameter_not_algorithm_kwarg():
    import inspect

    params = inspect.signature(OptimizedFunction.optimize).parameters
    assert "eval_dataset" in params
    assert params["eval_dataset"].default is None
