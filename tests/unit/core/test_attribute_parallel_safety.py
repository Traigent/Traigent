"""Tests for removed attribute injection mode and TRAIGENT_DISABLED functionality.

This module verifies that:
1. Using injection_mode='attribute' raises ConfigurationError with migration guidance
2. Using injection_mode='decorator' raises ConfigurationError with migration guidance
3. TRAIGENT_DISABLED=1 makes @optimize a no-op pass-through
"""

import pytest

import traigent
from traigent.utils.exceptions import ConfigurationError

# Use an existing dataset from the workspace
EVAL_DATASET = "examples/datasets/hello-world/evaluation_set.jsonl"


class TestAttributeModeRemoval:
    """Tests verifying attribute mode has been removed with helpful migration."""

    def test_attribute_mode_raises_configuration_error(self, monkeypatch):
        """Using injection_mode='attribute' should raise ConfigurationError."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        with pytest.raises(ConfigurationError) as exc_info:

            @traigent.optimize(
                objectives=["accuracy"],
                configuration_space={"model": ["a", "b"]},
                eval_dataset=EVAL_DATASET,
                injection_mode="attribute",
            )
            def my_func(question: str) -> str:
                return "answer"

        error_message = str(exc_info.value)
        assert "attribute" in error_message
        assert "removed" in error_message.lower()
        assert "context" in error_message  # Migration suggestion
        assert "seamless" in error_message  # Alternative suggestion

    def test_decorator_mode_raises_configuration_error(self, monkeypatch):
        """Using injection_mode='decorator' (legacy alias) should raise ConfigurationError."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        with pytest.raises(ConfigurationError) as exc_info:

            @traigent.optimize(
                objectives=["accuracy"],
                configuration_space={"model": ["a", "b"]},
                eval_dataset=EVAL_DATASET,
                injection_mode="decorator",
            )
            def my_func(question: str) -> str:
                return "answer"

        error_message = str(exc_info.value)
        assert "decorator" in error_message
        assert "removed" in error_message.lower()

    def test_context_mode_still_works(self, monkeypatch):
        """Context mode should still work correctly."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a"]},
            eval_dataset=EVAL_DATASET,
            injection_mode="context",
        )
        def my_func(question: str) -> str:
            return "answer"

        # Should not raise
        assert my_func is not None
        assert hasattr(my_func, "optimize")

    def test_seamless_mode_still_works(self, monkeypatch):
        """Seamless mode should still work correctly."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        @traigent.optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a"]},
            eval_dataset=EVAL_DATASET,
            injection_mode="seamless",
        )
        def my_func(question: str) -> str:
            model = "default"
            return f"answer from {model}"

        # Should not raise
        assert my_func is not None
        assert hasattr(my_func, "optimize")


class TestTraigentDisabled:
    """Tests for TRAIGENT_DISABLED environment variable."""

    def test_traigent_disabled_makes_decorator_passthrough(self, monkeypatch):
        """When TRAIGENT_DISABLED=1, @optimize should return the original function."""
        monkeypatch.setenv("TRAIGENT_DISABLED", "1")

        # Need to reimport to pick up the env var change
        from importlib import reload

        import traigent.api.decorators

        reload(traigent.api.decorators)
        from traigent.api.decorators import optimize

        def my_func(question: str) -> str:
            return f"answer to {question}"

        decorated = optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a", "b"]},
            eval_dataset=EVAL_DATASET,
        )(my_func)

        # Should be the original function, not an OptimizedFunction
        assert decorated is my_func
        assert decorated("test") == "answer to test"

    def test_traigent_disabled_true_works(self, monkeypatch):
        """TRAIGENT_DISABLED=true should also disable Traigent."""
        monkeypatch.setenv("TRAIGENT_DISABLED", "true")

        from importlib import reload

        import traigent.api.decorators

        reload(traigent.api.decorators)
        from traigent.api.decorators import optimize

        def my_func(question: str) -> str:
            return "answer"

        decorated = optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a"]},
        )(my_func)

        assert decorated is my_func

    def test_traigent_disabled_yes_works(self, monkeypatch):
        """TRAIGENT_DISABLED=yes should also disable Traigent."""
        monkeypatch.setenv("TRAIGENT_DISABLED", "yes")

        from importlib import reload

        import traigent.api.decorators

        reload(traigent.api.decorators)
        from traigent.api.decorators import optimize

        def my_func(question: str) -> str:
            return "answer"

        decorated = optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a"]},
        )(my_func)

        assert decorated is my_func

    def test_traigent_not_disabled_by_default(self, monkeypatch):
        """Without TRAIGENT_DISABLED, @optimize should work normally."""
        monkeypatch.delenv("TRAIGENT_DISABLED", raising=False)
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

        from importlib import reload

        import traigent.api.decorators

        reload(traigent.api.decorators)
        from traigent.api.decorators import optimize

        def my_func(question: str) -> str:
            return "answer"

        decorated = optimize(
            objectives=["accuracy"],
            configuration_space={"model": ["a"]},
            eval_dataset=EVAL_DATASET,
        )(my_func)

        # Should be an OptimizedFunction, not the original
        assert decorated is not my_func
        assert hasattr(decorated, "optimize")
