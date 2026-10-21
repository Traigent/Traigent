"""example_timeout: per-example limit, decoupled from the run-level timeout (#2509)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pydantic
import pytest

from traigent.api.decorators import ExecutionOptions, optimize
from traigent.core.optimization_pipeline import create_effective_evaluator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.callbacks import ProgressBarCallback, ProgressInfo


def _make(timeout, example_timeout, custom=None):
    return create_effective_evaluator(
        timeout=timeout,
        custom_evaluator=custom,
        effective_batch_size=None,
        effective_thread_workers=None,
        effective_privacy_enabled=False,
        objectives=["accuracy"],
        execution_mode="local",
        metric_functions=None,
        scoring_function=None,
        decorator_custom_evaluator=None,
        example_timeout=example_timeout,
    )[0]


async def _noop_custom(func, config, example):  # pragma: no cover - never run
    raise AssertionError


class TestEvaluatorWiring:
    def test_local_evaluator_uses_example_timeout(self):
        assert _make(timeout=5.0, example_timeout=7.0).timeout == 7.0

    def test_custom_wrapper_uses_example_timeout(self):
        assert _make(5.0, 9.0, custom=_noop_custom).timeout == 9.0

    @pytest.mark.parametrize("custom", [None, _noop_custom])
    def test_run_timeout_no_longer_sets_per_example_limit(self, custom):
        # Regression: run timeout=5 used to become the per-example limit.
        assert _make(timeout=5.0, example_timeout=None, custom=custom).timeout == 60.0
        assert _make(timeout=500.0, example_timeout=None, custom=custom).timeout == 60.0


class TestValidation:
    def test_default_none(self):
        assert ExecutionOptions().example_timeout is None

    @pytest.mark.parametrize("value", [0, -1, "5", True, float("nan"), float("inf")])
    def test_bad_values_rejected(self, value):
        with pytest.raises(pydantic.ValidationError):
            ExecutionOptions(example_timeout=value)

    def test_direct_kwarg_plumbs_and_bad_value_rejected(self):
        @optimize(
            configuration_space={"m": ["a", "b"]},
            objectives=["accuracy"],
            example_timeout=12,
        )
        def fn(text: str, m: str = "a") -> str:
            return text

        assert fn.example_timeout == 12

        with pytest.raises(ValueError, match="example_timeout"):

            @optimize(
                configuration_space={"m": ["a", "b"]},
                objectives=["accuracy"],
                example_timeout=-3,
            )
            def bad(text: str, m: str = "a") -> str:
                return text

    def test_bundle_plumbs(self):
        @optimize(
            configuration_space={"m": ["a", "b"]},
            objectives=["accuracy"],
            execution=ExecutionOptions(example_timeout=4),
        )
        def fn(text: str, m: str = "a") -> str:
            return text

        assert fn.example_timeout == 4


def _dataset():
    return Dataset(
        examples=[EvaluationExample(input_data={"text": "x"}, expected_output="x")],
        name="t",
    )


class TestTimeoutCounted:
    def test_local_evaluator_counts_timed_out_example(self):
        async def slow(text: str) -> str:
            await asyncio.sleep(2)
            return text

        evaluator = _make(timeout=None, example_timeout=0.05)
        result = asyncio.run(evaluator.evaluate(slow, {}, _dataset()))
        assert result.timed_out_examples == 1
        assert result.to_dict()["timed_out_examples"] == 1
        assert result.successful_examples == 0

    def test_custom_wrapper_enforces_and_counts_timeout(self):
        async def slow_custom(func, config, example):
            await asyncio.sleep(2)

        evaluator = _make(None, 0.05, custom=slow_custom)

        async def fn(text: str) -> str:
            return text

        result = asyncio.run(evaluator.evaluate(fn, {}, _dataset()))
        assert result.timed_out_examples == 1


class TestDisplay:
    def _progress(self):
        return ProgressInfo(1, 2, 1, 1, 0, 0.5, None, 1.0, None, "grid")

    def test_banner_shows_limit(self, capsys):
        cb = ProgressBarCallback()
        cb.on_run_settings({"example_timeout": 60.0})
        cb.on_optimization_start({"m": ["a"]}, ["accuracy"], "grid")
        assert "Per-example time limit: 60s" in capsys.readouterr().out

    def test_banner_silent_without_setting(self, capsys):
        cb = ProgressBarCallback()
        cb.on_optimization_start({"m": ["a"]}, ["accuracy"], "grid")
        assert "Per-example" not in capsys.readouterr().out

    def test_trial_line_shows_timeout_count(self, capsys):
        cb = ProgressBarCallback(update_interval=0)
        trial = SimpleNamespace(metadata={"timed_out_examples": 3})
        cb.on_trial_complete(trial, self._progress())
        assert "3 timed out" in capsys.readouterr().out

    def test_trial_line_omits_zero(self, capsys):
        cb = ProgressBarCallback(update_interval=0)
        cb.on_trial_complete(SimpleNamespace(metadata={}), self._progress())
        assert "timed out" not in capsys.readouterr().out
