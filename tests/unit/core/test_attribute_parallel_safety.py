"""Tests for attribute injection mode + parallel execution safety guards.

This module tests that:
1. ValueError is raised when injection_mode='attribute' with trial_concurrency > 1
2. Warning is logged when allow_parallel_attribute=True is used
3. No error/warning for attribute mode with sequential execution
4. No error/warning for other injection modes with parallel execution
"""

import logging

import pytest

import traigent

# Use an existing dataset from the workspace
EVAL_DATASET = "examples/datasets/hello-world/evaluation_set.jsonl"


def _require(condition: bool, message: str) -> None:
    """Fail the test with a clear message when a condition is not met."""
    if not condition:
        pytest.fail(message)


@pytest.fixture
def mock_mode_env(monkeypatch):
    """Enable mock mode for all tests."""
    monkeypatch.setenv("TRAIGENT_MOCK_MODE", "true")


class TestAttributeParallelSafetyGuard:
    """Tests for the attribute mode + parallel trials safety guard."""

    @pytest.mark.asyncio
    async def test_attribute_mode_parallel_raises_by_default(self, mock_mode_env):
        """Attribute mode with parallel trials should raise ValueError by default."""

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a", "b", "c"]},
            eval_dataset=EVAL_DATASET,
            injection_mode="attribute",
            parallel_config={"trial_concurrency": 2, "mode": "parallel"},
        )
        def my_func(question: str) -> str:
            return "answer"

        with pytest.raises(ValueError) as exc_info:
            await my_func.optimize(max_trials=2)

        error_message = str(exc_info.value)
        _require(
            "injection_mode='attribute'" in error_message,
            "Expected injection_mode='attribute' to be mentioned in the error message.",
        )
        _require(
            "unsafe with parallel trials" in error_message,
            "Expected safety warning about parallel trials in the error message.",
        )
        _require(
            "allow_parallel_attribute" in error_message,
            "Expected allow_parallel_attribute guidance in the error message.",
        )

    @pytest.mark.asyncio
    async def test_attribute_mode_parallel_with_opt_in_warns(
        self, mock_mode_env, caplog
    ):
        """Attribute mode with parallel trials and opt-in should warn but not raise."""

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a", "b"]},
            eval_dataset=EVAL_DATASET,
            injection={"injection_mode": "attribute", "allow_parallel_attribute": True},
            parallel_config={"trial_concurrency": 2, "mode": "parallel"},
        )
        def my_func(question: str) -> str:
            return "answer"

        # Should not raise, but should log warning
        with caplog.at_level(logging.WARNING):
            await my_func.optimize(max_trials=2)

        # Check warning was logged
        warning_found = any(
            "injection_mode='attribute' with parallel trials" in record.message
            and "not safe" in record.message
            for record in caplog.records
        )
        _require(
            warning_found,
            "Expected warning about attribute mode parallel safety, got: "
            f"{[r.message for r in caplog.records]}",
        )

    @pytest.mark.asyncio
    async def test_attribute_mode_parallel_with_top_level_opt_in_warns(
        self, mock_mode_env, caplog
    ):
        """Top-level allow_parallel_attribute should behave like the injection bundle."""

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a", "b"]},
            eval_dataset=EVAL_DATASET,
            injection_mode="attribute",
            allow_parallel_attribute=True,
            parallel_config={"trial_concurrency": 2, "mode": "parallel"},
        )
        def my_func(question: str) -> str:
            return "answer"

        # Should not raise, but should log warning
        with caplog.at_level(logging.WARNING):
            await my_func.optimize(max_trials=2)

        # Check warning was logged
        warning_found = any(
            "injection_mode='attribute' with parallel trials" in record.message
            and "not safe" in record.message
            for record in caplog.records
        )
        _require(
            warning_found,
            "Expected warning about attribute mode parallel safety, got: "
            f"{[r.message for r in caplog.records]}",
        )

    @pytest.mark.asyncio
    async def test_attribute_mode_sequential_no_error(self, mock_mode_env):
        """Attribute mode with sequential execution should work without error."""

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a", "b"]},
            eval_dataset=EVAL_DATASET,
            injection_mode="attribute",
            parallel_config={"trial_concurrency": 1, "mode": "sequential"},
        )
        def my_func(question: str) -> str:
            return "answer"

        # Should not raise
        result = await my_func.optimize(max_trials=2)
        _require(result is not None, "Expected an optimization result, got None.")
        _require(len(result.trials) > 0, "Expected at least one trial result.")

    @pytest.mark.asyncio
    async def test_attribute_mode_trial_concurrency_one_auto_no_error(
        self, mock_mode_env
    ):
        """Attribute mode with trial_concurrency=1 (auto mode) should work."""

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a"]},
            eval_dataset=EVAL_DATASET,
            injection_mode="attribute",
            parallel_config={"trial_concurrency": 1},
        )
        def my_func(question: str) -> str:
            return "answer"

        result = await my_func.optimize(max_trials=1)
        _require(result is not None, "Expected an optimization result, got None.")
        _require(len(result.trials) == 1, "Expected exactly one trial result.")

    @pytest.mark.asyncio
    async def test_context_mode_parallel_no_error(self, mock_mode_env):
        """Context mode with parallel execution should work without error."""

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a", "b"]},
            eval_dataset=EVAL_DATASET,
            injection_mode="context",
            parallel_config={"trial_concurrency": 2, "mode": "parallel"},
        )
        def my_func(question: str) -> str:
            return "answer"

        # Should not raise
        result = await my_func.optimize(max_trials=2)
        _require(result is not None, "Expected an optimization result, got None.")

    @pytest.mark.asyncio
    async def test_parameter_mode_parallel_no_error(self, mock_mode_env):
        """Parameter mode with parallel execution should work without error."""

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a", "b"]},
            eval_dataset=EVAL_DATASET,
            injection_mode="parameter",
            config_param="config",
            parallel_config={"trial_concurrency": 2, "mode": "parallel"},
        )
        def my_func(question: str, config: dict) -> str:
            return "answer"

        # Should not raise
        result = await my_func.optimize(max_trials=2)
        _require(result is not None, "Expected an optimization result, got None.")


class TestAttributeParallelSafetyUnitLevel:
    """Unit-level tests for the validation logic."""

    def test_injection_options_has_allow_parallel_attribute_field(self):
        """InjectionOptions should have allow_parallel_attribute field."""
        from traigent.api.decorators import InjectionOptions

        options = InjectionOptions()
        _require(
            hasattr(options, "allow_parallel_attribute"),
            "Expected InjectionOptions to have allow_parallel_attribute.",
        )
        _require(
            options.allow_parallel_attribute is False,
            "Expected allow_parallel_attribute to default to False.",
        )

    def test_injection_options_allow_parallel_attribute_true(self):
        """InjectionOptions should accept allow_parallel_attribute=True."""
        from traigent.api.decorators import InjectionOptions

        options = InjectionOptions(allow_parallel_attribute=True)
        _require(
            options.allow_parallel_attribute is True,
            "Expected allow_parallel_attribute to be True when set.",
        )

    def test_optimized_function_stores_allow_parallel_attribute(self, mock_mode_env):
        """OptimizedFunction should store allow_parallel_attribute."""

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a"]},
            eval_dataset=EVAL_DATASET,
            injection={"injection_mode": "attribute", "allow_parallel_attribute": True},
        )
        def my_func(question: str) -> str:
            return "answer"

        _require(
            hasattr(my_func, "allow_parallel_attribute"),
            "Expected optimized function to expose allow_parallel_attribute.",
        )
        _require(
            my_func.allow_parallel_attribute is True,
            "Expected allow_parallel_attribute to be True on optimized function.",
        )
