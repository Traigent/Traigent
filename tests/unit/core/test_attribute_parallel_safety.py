"""Tests for attribute injection mode + parallel execution safety guards.

This module tests that:
1. ValueError is raised when injection_mode='attribute' with trial_concurrency > 1
2. No error for attribute mode with sequential execution (trial_concurrency=1)
3. No error for other injection modes (context, parameter) with parallel execution

Note: The allow_parallel_attribute opt-in was removed because it provided false
security. Users who opted in could still accidentally use my_func.current_config
(unsafe) instead of traigent.get_config() (safe), leading to silent data corruption.
The block is now unconditional - use injection_mode='context' for parallel trials.
"""

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
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")


class TestAttributeParallelSafetyGuard:
    """Tests for the attribute mode + parallel trials safety guard."""

    @pytest.mark.asyncio
    async def test_attribute_mode_parallel_raises_unconditionally(self, mock_mode_env):
        """Attribute mode with parallel trials should always raise ValueError.

        This is an unconditional block - no opt-in is allowed because attribute
        mode is fundamentally incompatible with parallel execution due to the
        shared mutable function attribute (my_func.current_config).
        """

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
            "not supported with parallel trials" in error_message,
            "Expected clear message that attribute mode is not supported with parallel.",
        )
        _require(
            "race conditions" in error_message,
            "Expected explanation about race conditions in the error message.",
        )
        # Verify alternatives are suggested
        _require(
            "injection_mode='context'" in error_message,
            "Expected context mode to be suggested as alternative.",
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


class TestInjectionOptionsNoAllowParallelAttribute:
    """Verify allow_parallel_attribute has been removed from InjectionOptions."""

    def test_injection_options_no_allow_parallel_attribute_field(self):
        """InjectionOptions should NOT have allow_parallel_attribute field.

        This field was removed because it provided false security - users could
        opt in but still accidentally use the unsafe attribute access pattern.
        """
        from traigent.api.decorators import InjectionOptions

        options = InjectionOptions()
        _require(
            not hasattr(options, "allow_parallel_attribute"),
            "InjectionOptions should NOT have allow_parallel_attribute field "
            "(it was removed for safety reasons).",
        )

    def test_injection_options_rejects_allow_parallel_attribute(self):
        """InjectionOptions should reject allow_parallel_attribute as unknown field."""
        from pydantic import ValidationError

        from traigent.api.decorators import InjectionOptions

        with pytest.raises(ValidationError) as exc_info:
            InjectionOptions(allow_parallel_attribute=True)

        error_str = str(exc_info.value)
        _require(
            "allow_parallel_attribute" in error_str.lower()
            or "extra" in error_str.lower(),
            f"Expected validation error for unknown field, got: {error_str}",
        )
