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
        assert "injection_mode='attribute'" in error_message
        assert "unsafe with parallel trials" in error_message
        assert "allow_parallel_attribute" in error_message

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
        assert any(
            "injection_mode='attribute' with parallel trials" in record.message
            and "not safe" in record.message
            for record in caplog.records
        ), f"Expected warning about attribute mode parallel safety, got: {[r.message for r in caplog.records]}"

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
        assert any(
            "injection_mode='attribute' with parallel trials" in record.message
            and "not safe" in record.message
            for record in caplog.records
        ), f"Expected warning about attribute mode parallel safety, got: {[r.message for r in caplog.records]}"

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
        assert result is not None
        assert len(result.trials) > 0

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
        assert result is not None
        assert len(result.trials) == 1

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
        assert result is not None

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
        assert result is not None


class TestAttributeParallelSafetyUnitLevel:
    """Unit-level tests for the validation logic."""

    def test_injection_options_has_allow_parallel_attribute_field(self):
        """InjectionOptions should have allow_parallel_attribute field."""
        from traigent.api.decorators import InjectionOptions

        options = InjectionOptions()
        assert hasattr(options, "allow_parallel_attribute")
        assert options.allow_parallel_attribute is False

    def test_injection_options_allow_parallel_attribute_true(self):
        """InjectionOptions should accept allow_parallel_attribute=True."""
        from traigent.api.decorators import InjectionOptions

        options = InjectionOptions(allow_parallel_attribute=True)
        assert options.allow_parallel_attribute is True

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

        assert hasattr(my_func, "allow_parallel_attribute")
        assert my_func.allow_parallel_attribute is True
