"""Comprehensive tests for optimizer registry."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from traigent.api.types import TrialResult
from traigent.config.feature_flags import flag_registry
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.registry import (
    _OPTIMIZER_REGISTRY,
    _register_builtin_optimizers,
    clear_registry,
    get_optimizer,
    get_optimizer_info,
    list_optimizers,
    register_optimizer,
)
from traigent.utils.exceptions import OptimizationError, PluginError


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self.test_param = kwargs.get("test_param", "default")

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Mock implementation."""
        return {"x": 1}

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Mock implementation."""
        return False


class AnotherMockOptimizer(BaseOptimizer):
    """Another mock optimizer for testing."""

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        return {"y": 2}

    def should_stop(self, history: list[TrialResult]) -> bool:
        return len(history) >= 10


class NotAnOptimizer:
    """Class that doesn't inherit from BaseOptimizer."""

    pass


@pytest.fixture(autouse=True)
def reset_optuna_flag(monkeypatch):
    """Reset Optuna flag state for registry tests.

    Optuna is enabled by default, so we just ensure consistent state.
    """
    monkeypatch.delenv("TRAIGENT_OPTUNA_ENABLED", raising=False)
    flag_registry.reset()
    yield
    flag_registry.reset()


class TestOptimizerRegistry:
    """Test suite for optimizer registry."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def teardown_method(self):
        """Restore builtin optimizers after each test."""
        clear_registry()
        _register_builtin_optimizers()

    def test_register_optimizer_valid(self):
        """Test registering a valid optimizer."""
        register_optimizer("test_opt", MockOptimizer)

        assert "test_opt" in _OPTIMIZER_REGISTRY
        assert _OPTIMIZER_REGISTRY["test_opt"] == MockOptimizer

    def test_register_optimizer_invalid_name(self):
        """Test registering with invalid name."""
        with pytest.raises(PluginError, match="non-empty string"):
            register_optimizer("", MockOptimizer)

        with pytest.raises(PluginError, match="non-empty string"):
            register_optimizer("   ", MockOptimizer)

        with pytest.raises(PluginError, match="non-empty string"):
            register_optimizer(None, MockOptimizer)

    def test_register_optimizer_invalid_class(self):
        """Test registering invalid optimizer class."""
        with pytest.raises(PluginError, match="must inherit from BaseOptimizer"):
            register_optimizer("invalid", NotAnOptimizer)

        with pytest.raises(PluginError, match="must inherit from BaseOptimizer"):
            register_optimizer("invalid", str)

    def test_register_optimizer_override(self):
        """Test overriding existing optimizer registration."""
        # First registration
        register_optimizer("test_opt", MockOptimizer)
        assert _OPTIMIZER_REGISTRY["test_opt"] == MockOptimizer

        # Override with warning
        with patch("traigent.optimizers.registry.logger") as mock_logger:
            register_optimizer("test_opt", AnotherMockOptimizer)
            mock_logger.warning.assert_called_once()

        assert _OPTIMIZER_REGISTRY["test_opt"] == AnotherMockOptimizer

    def test_get_optimizer_valid(self):
        """Test getting a registered optimizer."""
        register_optimizer("test_opt", MockOptimizer)

        config_space = {"x": [0, 1, 2]}
        objectives = ["accuracy"]

        optimizer = get_optimizer("test_opt", config_space, objectives)

        assert isinstance(optimizer, MockOptimizer)
        assert optimizer.config_space == config_space
        assert optimizer.objectives == objectives

    def test_get_optimizer_with_kwargs(self):
        """Test getting optimizer with additional kwargs."""
        register_optimizer("test_opt", MockOptimizer)

        config_space = {"x": [0, 1]}
        objectives = ["accuracy"]

        optimizer = get_optimizer(
            "test_opt", config_space, objectives, test_param="custom_value"
        )

        assert isinstance(optimizer, MockOptimizer)
        assert optimizer.test_param == "custom_value"

    def test_get_optimizer_unknown(self):
        """Test getting unknown optimizer."""
        register_optimizer("known", MockOptimizer)

        with pytest.raises(OptimizationError) as exc_info:
            get_optimizer("unknown", {}, [])

        assert "Unknown optimizer 'unknown'" in str(exc_info.value)
        assert "Available optimizers: ['known']" in str(exc_info.value)

    def test_get_optimizer_creation_error(self):
        """Test handling optimizer creation errors."""

        class FailingOptimizer(BaseOptimizer):
            def __init__(self, config_space, objectives, **kwargs):
                raise ValueError("Creation failed")

            def suggest_next_trial(self, history):
                pass

            def should_stop(self, history):
                pass

        register_optimizer("failing", FailingOptimizer)

        with pytest.raises(OptimizationError) as exc_info:
            get_optimizer("failing", {}, [])

        assert "Failed to create optimizer 'failing'" in str(exc_info.value)
        assert "Creation failed" in str(exc_info.value)

    def test_list_optimizers_empty(self):
        """Test listing optimizers when registry is empty."""
        optimizers = list_optimizers()
        assert optimizers == []

    def test_list_optimizers_multiple(self):
        """Test listing multiple registered optimizers."""
        register_optimizer("opt1", MockOptimizer)
        register_optimizer("opt2", AnotherMockOptimizer)
        register_optimizer("opt3", MockOptimizer)

        optimizers = list_optimizers()
        assert len(optimizers) == 3
        assert "opt1" in optimizers
        assert "opt2" in optimizers
        assert "opt3" in optimizers

    def test_get_optimizer_info_valid(self):
        """Test getting optimizer information."""
        register_optimizer("test_opt", MockOptimizer)

        info = get_optimizer_info("test_opt")

        assert info["name"] == "test_opt"
        assert info["class"] == "MockOptimizer"
        assert "test_optimizer_registry" in info["module"]
        assert "Mock optimizer for testing" in info["description"]

    def test_get_optimizer_info_no_docstring(self):
        """Test getting info for optimizer without docstring."""

        class NoDocOptimizer(BaseOptimizer):
            def suggest_next_trial(self, history):
                pass

            def should_stop(self, history):
                pass

        register_optimizer("no_doc", NoDocOptimizer)

        info = get_optimizer_info("no_doc")
        assert info["description"] == "No description available"

    def test_get_optimizer_info_unknown(self):
        """Test getting info for unknown optimizer."""
        register_optimizer("known", MockOptimizer)

        with pytest.raises(OptimizationError) as exc_info:
            get_optimizer_info("unknown")

        assert "Unknown optimizer 'unknown'" in str(exc_info.value)
        assert "Available optimizers: ['known']" in str(exc_info.value)

    def test_clear_registry(self):
        """Test clearing the registry."""
        # Add some optimizers
        register_optimizer("opt1", MockOptimizer)
        register_optimizer("opt2", AnotherMockOptimizer)

        assert len(list_optimizers()) == 2

        # Clear
        clear_registry()

        assert len(list_optimizers()) == 0
        assert len(_OPTIMIZER_REGISTRY) == 0

    @patch("traigent.optimizers.registry.register_optimizer")
    def test_register_builtin_optimizers(self, mock_register):
        """Test registering built-in optimizers."""
        # Mock imports to avoid dependencies
        with patch("traigent.optimizers.grid.GridSearchOptimizer", MockOptimizer):
            with patch(
                "traigent.optimizers.random.RandomSearchOptimizer", AnotherMockOptimizer
            ):
                _register_builtin_optimizers()

        # Should register at least grid and random
        assert mock_register.call_count >= 2

        # Check that grid and random were registered
        calls = [call[0][0] for call in mock_register.call_args_list]
        assert "grid" in calls
        assert "random" in calls

    @patch("traigent.optimizers.registry.logger")
    def test_register_builtin_optimizers_bayesian_available(self, mock_logger):
        """Test registering Bayesian optimizer when available."""
        mock_bayesian = Mock(spec=BaseOptimizer)

        with patch("traigent.optimizers.bayesian.BayesianOptimizer", mock_bayesian):
            _register_builtin_optimizers()

        # Should log successful registration
        assert any(
            "Registered Bayesian optimizer" in str(call)
            for call in mock_logger.debug.call_args_list
        )

    @patch("traigent.optimizers.registry.logger")
    def test_register_builtin_optimizers_bayesian_unavailable(self, mock_logger):
        """Test handling when Bayesian optimizer is unavailable."""
        import sys

        # Remove from cache if present and patch the import to fail
        original_modules = sys.modules.copy()
        if "traigent.optimizers.bayesian" in sys.modules:
            del sys.modules["traigent.optimizers.bayesian"]

        # Temporarily add a module that raises ImportError when accessed
        class ImportErrorModule:
            def __getattr__(self, name):
                raise ImportError("No module named 'sklearn'")

        sys.modules["traigent.optimizers.bayesian"] = ImportErrorModule()

        try:
            _register_builtin_optimizers()
        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)

        # Should log that Bayesian is not available
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any(
            "Bayesian optimizer not available" in call for call in debug_calls
        ), f"Debug calls: {debug_calls}"

    def test_registry_state_isolation(self):
        """Test that registry state is properly isolated between tests."""
        # This test verifies our setup_method works correctly
        assert len(list_optimizers()) == 0

        register_optimizer("isolated", MockOptimizer)
        assert "isolated" in list_optimizers()

    def test_thread_safety_considerations(self):
        """Test basic thread safety considerations."""
        # Note: Current implementation is not thread-safe
        # This test documents expected behavior

        import threading

        results = []
        errors = []

        def register_thread(name, optimizer_class):
            try:
                register_optimizer(name, optimizer_class)
                results.append(name)
            except Exception as e:
                errors.append(e)

        # Create multiple threads trying to register
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=register_thread, args=(f"thread_opt_{i}", MockOptimizer)
            )
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All registrations should succeed (though order not guaranteed)
        assert len(results) == 5
        assert len(errors) == 0

    def test_get_optimizer_complex_config_space(self):
        """Test getting optimizer with complex configuration space."""
        register_optimizer("complex", MockOptimizer)

        config_space = {
            "int_param": [1, 2, 3, 4, 5],
            "float_param": (0.0, 1.0),
            "str_param": ["a", "b", "c"],
            "nested": {"inner_param": [True, False], "inner_range": (0, 100)},
        }

        objectives = ["accuracy", "latency", "memory"]

        optimizer = get_optimizer("complex", config_space, objectives)

        assert optimizer.config_space == config_space
        assert optimizer.objectives == objectives

    def test_optimizer_registry_with_inheritance(self):
        """Test registering optimizers with inheritance hierarchy."""

        class BaseCustomOptimizer(MockOptimizer):
            """Base custom optimizer."""

            pass

        class SpecializedOptimizer(BaseCustomOptimizer):
            """Specialized version."""

            pass

        register_optimizer("base_custom", BaseCustomOptimizer)
        register_optimizer("specialized", SpecializedOptimizer)

        # Both should work
        opt1 = get_optimizer("base_custom", {"x": [1]}, ["obj"])
        opt2 = get_optimizer("specialized", {"x": [1]}, ["obj"])

        assert isinstance(opt1, BaseCustomOptimizer)
        assert isinstance(opt2, SpecializedOptimizer)
        assert isinstance(opt2, BaseCustomOptimizer)  # Also true

    def test_registry_performance(self):
        """Test registry performance with many optimizers."""
        # Register many optimizers
        for i in range(100):
            register_optimizer(f"opt_{i}", MockOptimizer)

        # List should be fast
        import time

        start = time.time()
        optimizers = list_optimizers()
        duration = time.time() - start

        assert len(optimizers) == 100
        assert duration < 0.1  # Should be very fast

        # Get should also be fast
        start = time.time()
        optimizer = get_optimizer("opt_50", {"x": [1]}, ["obj"])
        duration = time.time() - start

        assert isinstance(optimizer, MockOptimizer)
        assert duration < 0.1
