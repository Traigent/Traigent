"""Tests for enhanced BaseOptimizer with context and async support."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.config.types import TraigentConfig
from traigent.optimizers.base import BaseOptimizer


class MockOptimizer(BaseOptimizer):
    """Test implementation of BaseOptimizer for testing."""

    def __init__(
        self,
        config_space: Dict[str, Any],
        objectives: List[str],
        context=None,
        **kwargs,
    ):
        super().__init__(config_space, objectives, context, **kwargs)
        self._suggested_configs = []
        self._trial_count_internal = 0

    def suggest_next_trial(self, history: List[TrialResult]) -> Dict[str, Any]:
        """Test implementation that returns predefined configurations."""
        if self._trial_count_internal >= 3:
            raise Exception("Max trials reached")

        config = {
            "param1": f"value_{self._trial_count_internal}",
            "param2": self._trial_count_internal,
        }
        self._suggested_configs.append(config)
        self._trial_count_internal += 1
        return config

    def should_stop(self, history: List[TrialResult]) -> bool:
        """Stop after 3 trials."""
        return len(history) >= 3


class AsyncMockOptimizer(BaseOptimizer):
    """Test optimizer with async overrides."""

    def __init__(
        self,
        config_space: Dict[str, Any],
        objectives: List[str],
        context=None,
        **kwargs,
    ):
        super().__init__(config_space, objectives, context, **kwargs)
        self._async_call_count = 0
        self._generated_count = 0

    def suggest_next_trial(self, history: List[TrialResult]) -> Dict[str, Any]:
        """Sync fallback."""
        return {"sync": True, "param": "value"}

    async def suggest_next_trial_async(
        self, history: List[TrialResult], remote_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Async implementation with remote context support."""
        self._async_call_count += 1
        self._generated_count += 1
        await asyncio.sleep(0.01)  # Simulate async operation

        config = {"async": True, "param": f"async_value_{self._async_call_count}"}

        if remote_context:
            config.update({"remote_data": remote_context.get("extra_param", "default")})

        return config

    def should_stop(self, history: List[TrialResult]) -> bool:
        return len(history) >= 2

    async def should_stop_async(
        self, history: List[TrialResult], remote_context: Dict[str, Any] = None
    ) -> bool:
        """Async version with remote context."""
        await asyncio.sleep(0.01)  # Simulate async operation

        # Different stopping condition based on remote context
        if remote_context and remote_context.get("early_stop", False):
            return self._generated_count >= 1

        return self._generated_count >= 2


class TestEnhancedBaseOptimizer:
    """Test suite for enhanced BaseOptimizer functionality."""

    def test_context_support(self):
        """Test that optimizer accepts and stores TraigentConfig context."""
        config_space = {"param1": ["a", "b"], "param2": [1, 2]}
        objectives = ["accuracy"]
        context = TraigentConfig(custom_params={"global_setting": "test_value"})

        optimizer = MockOptimizer(config_space, objectives, context=context)

        assert optimizer.context is not None
        assert optimizer.context.custom_params["global_setting"] == "test_value"
        assert optimizer.config_space == config_space
        assert optimizer.objectives == objectives

    def test_no_context(self):
        """Test that optimizer works without context."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]

        optimizer = MockOptimizer(config_space, objectives)

        assert optimizer.context is None
        assert optimizer.config_space == config_space
        assert optimizer.objectives == objectives

    def test_async_suggest_next_trial_default(self):
        """Test that default async implementation calls synchronous version."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = MockOptimizer(config_space, objectives)

        # Test default async implementation
        result = asyncio.run(optimizer.suggest_next_trial_async([]))

        assert result["param1"] == "value_0"
        assert result["param2"] == 0

    def test_async_suggest_next_trial_custom(self):
        """Test custom async implementation."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = AsyncMockOptimizer(config_space, objectives)

        result = asyncio.run(optimizer.suggest_next_trial_async([]))

        assert result["async"] is True
        assert result["param"] == "async_value_1"
        assert optimizer._async_call_count == 1

    def test_async_suggest_with_remote_context(self):
        """Test async suggest with remote context."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = AsyncMockOptimizer(config_space, objectives)

        remote_context = {"extra_param": "remote_value"}
        result = asyncio.run(optimizer.suggest_next_trial_async([], remote_context))

        assert result["async"] is True
        assert result["remote_data"] == "remote_value"

    def test_async_should_stop_default(self):
        """Test that default async should_stop calls synchronous version."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = MockOptimizer(config_space, objectives)

        # Create mock trial results
        trials = [
            TrialResult(
                trial_id="test_1",
                config={"param1": "a"},
                metrics={"accuracy": 0.8},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
        ]

        result = asyncio.run(optimizer.should_stop_async(trials))
        assert result is False  # Should not stop with 1 trial

        # Add more trials
        trials.extend(
            [
                TrialResult(
                    trial_id="test_2",
                    config={"param1": "b"},
                    metrics={"accuracy": 0.9},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                ),
                TrialResult(
                    trial_id="test_3",
                    config={"param1": "a"},
                    metrics={"accuracy": 0.85},
                    status=TrialStatus.COMPLETED,
                    duration=1.0,
                    timestamp=datetime.now(),
                ),
            ]
        )

        result = asyncio.run(optimizer.should_stop_async(trials))
        assert result is True  # Should stop with 3 trials

    def test_async_should_stop_custom(self):
        """Test custom async should_stop implementation."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = AsyncMockOptimizer(config_space, objectives)

        trials = [
            TrialResult(
                trial_id="test_1",
                config={"param1": "a"},
                metrics={"accuracy": 0.8},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime.now(),
            )
        ]

        # Without remote context - should continue (0 generated configs)
        result = asyncio.run(optimizer.should_stop_async(trials))
        assert result is False

        # Generate a config to increment the counter
        asyncio.run(optimizer.suggest_next_trial_async([]))

        # With early stop context - should stop after 1 generated config
        remote_context = {"early_stop": True}
        result = asyncio.run(optimizer.should_stop_async(trials, remote_context))
        assert result is True

    def test_generate_candidates_async(self):
        """Test async candidate generation."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = AsyncMockOptimizer(config_space, objectives)

        candidates = asyncio.run(optimizer.generate_candidates_async(3))

        assert len(candidates) == 2  # Should stop after 2 due to should_stop_async
        assert all(candidate["async"] is True for candidate in candidates)
        assert candidates[0]["param"] == "async_value_1"
        assert candidates[1]["param"] == "async_value_2"

    def test_generate_candidates_async_with_remote_context(self):
        """Test async candidate generation with remote context."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = AsyncMockOptimizer(config_space, objectives)

        remote_context = {"extra_param": "context_value", "early_stop": True}
        candidates = asyncio.run(optimizer.generate_candidates_async(5, remote_context))

        assert len(candidates) == 1  # Should stop early due to remote context
        assert candidates[0]["remote_data"] == "context_value"

    def test_get_algorithm_info_enhanced(self):
        """Test enhanced algorithm info with new fields."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        context = TraigentConfig()
        optimizer = MockOptimizer(config_space, objectives, context=context)

        info = optimizer.get_algorithm_info()

        # Check original fields
        assert info["name"] == "MockOptimizer"
        assert info["objectives"] == ["accuracy"]
        assert info["config_space"] == config_space

        # Check new fields
        assert info["supports_async"] is True
        assert info["supports_remote"] is False  # No remote_service attribute
        assert info["context_aware"] is True  # Has context

    def test_get_algorithm_info_no_context(self):
        """Test algorithm info without context."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = MockOptimizer(config_space, objectives)

        info = optimizer.get_algorithm_info()

        assert info["context_aware"] is False  # No context
        assert info["supports_remote"] is False

    def test_backward_compatibility(self):
        """Test that existing synchronous usage still works."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]
        optimizer = MockOptimizer(config_space, objectives)

        # Original synchronous methods should still work
        config = optimizer.suggest_next_trial([])
        assert config["param1"] == "value_0"

        should_stop = optimizer.should_stop([])
        assert should_stop is False

        candidates = optimizer.generate_candidates(2)
        assert len(candidates) == 2

    def test_context_integration(self):
        """Test that context can be used within optimizer logic."""
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]

        # Create context with custom parameters
        context = TraigentConfig(
            custom_params={"max_trials": 5, "algorithm_setting": "conservative"}
        )

        optimizer = MockOptimizer(config_space, objectives, context=context)

        # Verify optimizer can access context
        assert optimizer.context.custom_params["max_trials"] == 5
        assert optimizer.context.get("algorithm_setting") == "conservative"
        assert optimizer.context.get("nonexistent", "default") == "default"


def test_concurrent_async_operations():
    """Test concurrent async operations on optimizer."""

    async def run_test():
        config_space = {"param1": ["a", "b"]}
        objectives = ["accuracy"]

        # Test individual async operations sequentially to avoid shared state issues
        optimizer1 = AsyncMockOptimizer(config_space, objectives)
        config1 = await optimizer1.suggest_next_trial_async([])

        optimizer2 = AsyncMockOptimizer(config_space, objectives)
        config2 = await optimizer2.suggest_next_trial_async([])

        optimizer3 = AsyncMockOptimizer(config_space, objectives)
        candidates = await optimizer3.generate_candidates_async(2)

        # Verify results
        assert config1["async"] is True
        assert config2["async"] is True
        assert len(candidates) == 2
        assert all(c["async"] is True for c in candidates)

        # Also test true concurrent operations on separate optimizers
        tasks = [
            AsyncMockOptimizer(config_space, objectives).suggest_next_trial_async([]),
            AsyncMockOptimizer(config_space, objectives).suggest_next_trial_async([]),
        ]

        concurrent_results = await asyncio.gather(*tasks)
        assert len(concurrent_results) == 2
        assert all(result["async"] is True for result in concurrent_results)

    # Run the async test
    asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__])
