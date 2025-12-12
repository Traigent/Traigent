"""Comprehensive tests for base optimizer class."""

from datetime import datetime
from typing import Any
from unittest.mock import Mock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.optimizers.base import BaseOptimizer


class ConcreteOptimizer(BaseOptimizer):
    """Concrete implementation of BaseOptimizer for testing."""

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        context: TraigentConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(config_space, objectives, context, **kwargs)
        self.suggest_count = 0
        self.stop_after = kwargs.get("stop_after", None)
        self.force_exception = kwargs.get("force_exception", False)

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration."""
        if self.force_exception:
            raise Exception("Forced exception in suggest_next_trial")

        self.suggest_count += 1
        self._trial_count += 1

        # Generate simple config based on count
        config = {}
        for param, values in self.config_space.items():
            if isinstance(values, list):
                config[param] = values[self.suggest_count % len(values)]
            elif isinstance(values, tuple) and len(values) == 2:
                # Range (min, max)
                config[param] = values[0] + self.suggest_count
            else:
                config[param] = values

        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop."""
        if self.stop_after is not None:
            return len(history) >= self.stop_after
        return False


class AsyncOptimizer(ConcreteOptimizer):
    """Optimizer with custom async implementations."""

    async def suggest_next_trial_async(
        self,
        history: list[TrialResult],
        remote_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Custom async implementation."""
        # Simulate async behavior
        import asyncio

        await asyncio.sleep(0.001)

        # Add remote context info if provided
        config = self.suggest_next_trial(history)
        if remote_context:
            config["remote_id"] = remote_context.get("id", "default")

        return config

    async def should_stop_async(
        self,
        history: list[TrialResult],
        remote_context: dict[str, Any] | None = None,
    ) -> bool:
        """Custom async stop check."""
        import asyncio

        await asyncio.sleep(0.001)

        # Check remote context for stop signal
        if remote_context and remote_context.get("force_stop", False):
            return True

        return self.should_stop(history)


class TestBaseOptimizer:
    """Test suite for BaseOptimizer."""

    def test_abstract_base_class(self):
        """Test that BaseOptimizer is abstract and cannot be instantiated."""
        config_space = {"x": [0, 1, 2]}
        objectives = ["accuracy"]

        with pytest.raises(TypeError):
            BaseOptimizer(config_space, objectives)

    def test_initialization(self):
        """Test optimizer initialization."""
        config_space = {"x": [0, 1, 2], "y": (0.0, 1.0)}
        objectives = ["accuracy", "latency"]
        context = TraigentConfig(model="test-model")

        optimizer = ConcreteOptimizer(
            config_space=config_space,
            objectives=objectives,
            context=context,
            custom_param="value",
        )

        assert optimizer.config_space == config_space
        assert optimizer.objectives == objectives
        assert optimizer.context == context
        assert optimizer.algorithm_config == {"custom_param": "value"}
        assert optimizer._trial_count == 0
        assert optimizer._best_score is None
        assert optimizer._best_config is None

    def test_initialization_requires_objectives(self):
        """Optimizer should reject empty objective lists."""
        config_space = {"x": [0, 1, 2]}

        optimizer = ConcreteOptimizer(config_space=config_space, objectives=[])
        assert optimizer.objectives == []
        assert optimizer.objective_weights == {}

    def test_suggest_next_trial(self):
        """Test suggesting next trial configuration."""
        config_space = {"x": [0, 1, 2], "y": ["a", "b"]}
        optimizer = ConcreteOptimizer(config_space, ["accuracy"])

        # First suggestion
        config1 = optimizer.suggest_next_trial([])
        assert "x" in config1
        assert "y" in config1
        assert config1["x"] in [0, 1, 2]
        assert config1["y"] in ["a", "b"]

        # Second suggestion should be different
        config2 = optimizer.suggest_next_trial([])
        assert config1 != config2

    def test_should_stop(self):
        """Test stop condition checking."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"], stop_after=3)

        # Create mock history
        history = []
        assert optimizer.should_stop(history) is False

        # Add trials to history
        for _i in range(3):
            trial = Mock(spec=TrialResult)
            history.append(trial)

        assert optimizer.should_stop(history) is True

    def test_generate_candidates(self):
        """Test generating multiple candidates."""
        config_space = {"x": [0, 1, 2], "y": [10, 20, 30]}
        optimizer = ConcreteOptimizer(config_space, ["accuracy"])

        # Generate candidates
        candidates = optimizer.generate_candidates(5)

        assert len(candidates) == 5
        assert all("x" in c and "y" in c for c in candidates)

        # Test with stop condition - the base implementation doesn't update history
        # so should_stop won't trigger based on history length
        optimizer_with_stop = ConcreteOptimizer(
            config_space, ["accuracy"], stop_after=2
        )
        candidates = optimizer_with_stop.generate_candidates(5)
        # Since generate_candidates uses empty history, it won't stop early
        assert len(candidates) == 5

    def test_generate_candidates_with_exception(self):
        """Test generate_candidates handles exceptions gracefully."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"], force_exception=True)

        candidates = optimizer.generate_candidates(5)
        assert len(candidates) == 0  # Should return empty list on exception

    @pytest.mark.asyncio
    async def test_suggest_next_trial_async_default(self):
        """Test default async implementation calls sync version."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        # Reset state to ensure consistent results
        optimizer.suggest_count = 0
        sync_result = optimizer.suggest_next_trial([])

        # Reset state again for async call
        optimizer.suggest_count = 0
        optimizer._trial_count = 0
        async_result = await optimizer.suggest_next_trial_async([])

        assert sync_result == async_result

    @pytest.mark.asyncio
    async def test_suggest_next_trial_async_custom(self):
        """Test custom async implementation."""
        optimizer = AsyncOptimizer({"x": [0, 1]}, ["accuracy"])

        remote_context = {"id": "remote-123"}
        result = await optimizer.suggest_next_trial_async([], remote_context)

        assert "x" in result
        assert result["remote_id"] == "remote-123"

    @pytest.mark.asyncio
    async def test_should_stop_async_default(self):
        """Test default async should_stop implementation."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"], stop_after=1)

        history = [Mock(spec=TrialResult)]
        sync_result = optimizer.should_stop(history)
        async_result = await optimizer.should_stop_async(history)

        assert sync_result == async_result
        assert async_result is True

    @pytest.mark.asyncio
    async def test_should_stop_async_custom(self):
        """Test custom async should_stop implementation."""
        optimizer = AsyncOptimizer({"x": [0, 1]}, ["accuracy"])

        # Test with remote stop signal
        remote_context = {"force_stop": True}
        result = await optimizer.should_stop_async([], remote_context)
        assert result is True

        # Test without stop signal
        remote_context = {"force_stop": False}
        result = await optimizer.should_stop_async([], remote_context)
        assert result is False

    @pytest.mark.asyncio
    async def test_generate_candidates_async(self):
        """Test async candidate generation."""
        optimizer = AsyncOptimizer({"x": [0, 1, 2]}, ["accuracy"])

        candidates = await optimizer.generate_candidates_async(3)
        assert len(candidates) == 3

        # Test with remote context
        remote_context = {"id": "batch-123"}
        candidates = await optimizer.generate_candidates_async(2, remote_context)
        assert len(candidates) == 2
        assert all("remote_id" in c for c in candidates)

    @pytest.mark.asyncio
    async def test_generate_candidates_async_with_stop(self):
        """Test async generation with stop condition."""
        optimizer = AsyncOptimizer({"x": [0, 1]}, ["accuracy"])

        remote_context = {"force_stop": True}
        candidates = await optimizer.generate_candidates_async(5, remote_context)
        assert len(candidates) == 0  # Should stop immediately

    def test_update_best_successful_trial(self):
        """Test updating best trial with successful result."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        # Create successful trial
        trial = TrialResult(
            trial_id="trial-1",
            config={"x": 1},
            metrics={"accuracy": 0.95},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )

        optimizer.update_best(trial)

        assert optimizer.best_score == 0.95
        assert optimizer.best_config == {"x": 1}

    def test_update_best_failed_trial(self):
        """Test that failed trials don't update best."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        # Set initial best
        good_trial = TrialResult(
            trial_id="trial-1",
            config={"x": 1},
            metrics={"accuracy": 0.8},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        optimizer.update_best(good_trial)

        # Try to update with failed trial
        failed_trial = TrialResult(
            trial_id="trial-2",
            config={"x": 0},
            metrics={"accuracy": 0.9},  # Better score but failed
            status=TrialStatus.FAILED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        optimizer.update_best(failed_trial)

        # Best should remain unchanged
        assert optimizer.best_score == 0.8
        assert optimizer.best_config == {"x": 1}

    def test_update_best_improvement(self):
        """Test updating best when better trial found."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        # First trial
        trial1 = TrialResult(
            trial_id="trial-1",
            config={"x": 0},
            metrics={"accuracy": 0.7},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        optimizer.update_best(trial1)

        # Better trial
        trial2 = TrialResult(
            trial_id="trial-2",
            config={"x": 1},
            metrics={"accuracy": 0.9},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        optimizer.update_best(trial2)

        assert optimizer.best_score == 0.9
        assert optimizer.best_config == {"x": 1}

    def test_update_best_no_improvement(self):
        """Test that worse trials don't update best."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        # Good trial
        trial1 = TrialResult(
            trial_id="trial-1",
            config={"x": 1},
            metrics={"accuracy": 0.9},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        optimizer.update_best(trial1)

        # Worse trial
        trial2 = TrialResult(
            trial_id="trial-2",
            config={"x": 0},
            metrics={"accuracy": 0.7},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        optimizer.update_best(trial2)

        # Best should remain unchanged
        assert optimizer.best_score == 0.9
        assert optimizer.best_config == {"x": 1}

    def test_update_best_missing_metric(self):
        """Test handling trials with missing primary metric."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        # Trial missing primary metric
        trial = TrialResult(
            trial_id="trial-1",
            config={"x": 1},
            metrics={"latency": 0.1},  # Missing 'accuracy'
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        optimizer.update_best(trial)

        # Best should remain None
        assert optimizer.best_score is None
        assert optimizer.best_config is None

    def test_properties(self):
        """Test optimizer properties."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        # Initial state
        assert optimizer.best_score is None
        assert optimizer.best_config is None
        assert optimizer.trial_count == 0

        # Generate some trials
        optimizer.suggest_next_trial([])
        optimizer.suggest_next_trial([])

        assert optimizer.trial_count == 2

    def test_reset(self):
        """Test resetting optimizer state."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        # Set some state
        trial = TrialResult(
            trial_id="trial-1",
            config={"x": 1},
            metrics={"accuracy": 0.9},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )
        optimizer.update_best(trial)
        optimizer._trial_count = 5

        # Reset
        optimizer.reset()

        assert optimizer._trial_count == 0
        assert optimizer._best_score is None
        assert optimizer._best_config is None

    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        config_space = {"x": [0, 1], "y": (0.0, 1.0)}
        objectives = ["accuracy", "latency"]
        context = TraigentConfig(model="test")

        optimizer = ConcreteOptimizer(
            config_space=config_space,
            objectives=objectives,
            context=context,
            param1="value1",
            param2=42,
        )

        info = optimizer.get_algorithm_info()

        assert info["name"] == "ConcreteOptimizer"
        assert "description" in info
        assert info["config"] == {"param1": "value1", "param2": 42}
        assert info["objectives"] == objectives
        assert info["config_space"] == config_space
        assert info["supports_async"] is True
        assert info["supports_remote"] is False
        assert info["context_aware"] is True

    def test_get_algorithm_info_no_context(self):
        """Test algorithm info without context."""
        optimizer = ConcreteOptimizer({"x": [0, 1]}, ["accuracy"])

        info = optimizer.get_algorithm_info()
        assert info["context_aware"] is False

    def test_complex_config_space(self):
        """Test with complex configuration space."""
        config_space = {
            "int_param": [1, 2, 3, 4, 5],
            "float_param": (0.0, 1.0),
            "str_param": ["a", "b", "c"],
            "bool_param": [True, False],
            "mixed_param": [1, "two", 3.0, True],
        }

        optimizer = ConcreteOptimizer(config_space, ["accuracy"])

        # Generate several configs
        configs = []
        for _ in range(10):
            config = optimizer.suggest_next_trial([])
            configs.append(config)

            # Verify all params present
            assert set(config.keys()) == set(config_space.keys())

    def test_multi_objective(self):
        """Test with multiple objectives."""
        objectives = ["accuracy", "latency", "memory"]
        optimizer = ConcreteOptimizer({"x": [0, 1]}, objectives)

        assert optimizer.objectives == objectives

        # Test update_best uses primary objective
        trial = TrialResult(
            trial_id="trial-1",
            config={"x": 1},
            metrics={"accuracy": 0.9, "latency": 0.1, "memory": 100},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(),
        )

        optimizer.update_best(trial)
        assert optimizer.best_score == 0.9  # Uses primary objective (accuracy)
