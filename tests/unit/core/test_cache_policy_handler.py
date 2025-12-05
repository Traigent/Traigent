"""Unit tests for CachePolicyHandler.

This test suite covers:
- Cache policy application for each policy type (allow_repeats, prefer_new, reuse_cached)
- Configuration deduplication logic
- Statistics tracking (configs_deduplicated, cache_policy_used)
- Thread-safe deduplication with storage locking
- Config space keys extraction and usage
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.config.types import TraigentConfig
from traigent.core.cache_policy import CachePolicyHandler
from traigent.optimizers.base import BaseOptimizer


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing."""

    def __init__(self, config_space: dict[str, Any] | None = None):
        self.config_space = config_space or {"temperature": [0.3, 0.5, 0.7]}
        self.objectives = ["accuracy"]

    def suggest_next_trial(self, trials: list) -> dict[str, Any]:
        return {"temperature": 0.5}

    def should_stop(self, trials: list) -> bool:
        return False


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def traigent_config(temp_storage_path):
    """Create TraigentConfig with temporary storage."""
    config = TraigentConfig()
    # Set storage path to temp directory
    config._local_storage_path = str(temp_storage_path)
    return config


@pytest.fixture
def optimizer():
    """Create a mock optimizer."""
    return MockOptimizer(
        config_space={"temperature": [0.3, 0.5, 0.7], "model": ["gpt-3.5", "gpt-4"]}
    )


@pytest.fixture
def cache_handler(traigent_config, optimizer):
    """Create a CachePolicyHandler instance."""
    return CachePolicyHandler(traigent_config=traigent_config, optimizer=optimizer)


class TestCachePolicyHandlerInitialization:
    """Test CachePolicyHandler initialization."""

    def test_initialization(self, cache_handler, traigent_config, optimizer):
        """Test that handler initializes correctly."""
        assert cache_handler._traigent_config == traigent_config
        assert cache_handler._optimizer == optimizer
        assert cache_handler._storage is not None
        assert cache_handler._configs_deduplicated == 0
        assert cache_handler._cache_policy_used is None

    def test_config_keys_extraction(self, cache_handler):
        """Test that config space keys are extracted correctly."""
        config_keys = cache_handler._get_config_keys()
        assert config_keys is not None
        assert set(config_keys) == {"temperature", "model"}

    def test_config_keys_without_config_space(self, traigent_config):
        """Test config keys extraction when optimizer has no config_space."""
        optimizer_no_space = MockOptimizer(config_space=None)
        delattr(optimizer_no_space, "config_space")
        handler = CachePolicyHandler(
            traigent_config=traigent_config, optimizer=optimizer_no_space
        )
        assert handler._get_config_keys() is None


class TestAllowRepeatsPolicy:
    """Test 'allow_repeats' cache policy."""

    def test_allow_repeats_returns_all_configs(self, cache_handler):
        """Test that allow_repeats policy returns all configs unchanged."""
        configs = [
            {"temperature": 0.3, "model": "gpt-3.5"},
            {"temperature": 0.5, "model": "gpt-4"},
            {"temperature": 0.7, "model": "gpt-3.5"},
        ]

        result = cache_handler.apply_policy(
            configs=configs,
            cache_policy="allow_repeats",
            function_name="test_func",
            dataset_name="test_dataset",
        )

        assert result == configs
        assert cache_handler.configs_deduplicated == 0
        assert cache_handler.cache_policy_used == "allow_repeats"

    def test_allow_repeats_with_duplicates(self, cache_handler):
        """Test that allow_repeats allows duplicate configs."""
        config = {"temperature": 0.5, "model": "gpt-4"}
        configs = [config, config, config]  # Same config 3 times

        result = cache_handler.apply_policy(
            configs=configs,
            cache_policy="allow_repeats",
            function_name="test_func",
            dataset_name="test_dataset",
        )

        assert len(result) == 3
        assert result == configs
        assert cache_handler.configs_deduplicated == 0


class TestPreferNewPolicy:
    """Test 'prefer_new' cache policy."""

    def test_prefer_new_with_no_history(self, cache_handler):
        """Test that prefer_new allows all configs when no history exists."""
        config1 = {"temperature": 0.3, "model": "gpt-3.5"}
        config2 = {"temperature": 0.5, "model": "gpt-4"}
        config3 = {"temperature": 0.7, "model": "gpt-3.5"}

        # No previous sessions - all configs should pass through
        result = cache_handler.apply_policy(
            configs=[config1, config2, config3],
            cache_policy="prefer_new",
            function_name="test_func",
            dataset_name="test_dataset",
        )

        assert len(result) == 3
        assert cache_handler.configs_deduplicated == 0
        assert cache_handler.cache_policy_used == "prefer_new"

    def test_prefer_new_calls_storage(self, cache_handler):
        """Test that prefer_new queries storage for config history."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        # Mock is_config_seen to return False (config not seen before)
        with patch.object(cache_handler._storage, "is_config_seen", return_value=False):
            result = cache_handler.apply_policy(
                configs=[config],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )

        assert len(result) == 1
        assert cache_handler.configs_deduplicated == 0

    def test_prefer_new_filters_with_mocked_storage(self, cache_handler):
        """Test that prefer_new filters configs marked as seen by storage."""
        config1 = {"temperature": 0.3, "model": "gpt-3.5"}
        config2 = {"temperature": 0.5, "model": "gpt-4"}

        # Mock is_config_seen to return True for config1, False for config2
        def mock_is_seen(fn, ds, cfg, keys):
            return cfg == config1

        with patch.object(
            cache_handler._storage, "is_config_seen", side_effect=mock_is_seen
        ):
            result = cache_handler.apply_policy(
                configs=[config1, config2],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )

        assert len(result) == 1
        assert result[0] == config2
        assert cache_handler.configs_deduplicated == 1

    def test_prefer_new_empty_result_logging(self, cache_handler, caplog):
        """Test that prefer_new logs info when all configs are filtered."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        # Mock storage to mark config as seen
        with patch.object(cache_handler._storage, "is_config_seen", return_value=True):
            with caplog.at_level("INFO"):
                result = cache_handler.apply_policy(
                    configs=[config],
                    cache_policy="prefer_new",
                    function_name="test_func",
                    dataset_name="test_dataset",
                )

        assert len(result) == 0
        assert "No new configurations to explore" in caplog.text
        assert "allow_repeats" in caplog.text

    def test_prefer_new_different_functions(self, cache_handler):
        """Test that prefer_new passes function_name to storage correctly."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        with patch.object(cache_handler._storage, "is_config_seen") as mock_seen:
            mock_seen.return_value = False

            cache_handler.apply_policy(
                configs=[config],
                cache_policy="prefer_new",
                function_name="function1",
                dataset_name="test_dataset",
            )

            # Verify function_name was passed to storage
            mock_seen.assert_called_once()
            call_args = mock_seen.call_args
            assert call_args[0][0] == "function1"  # function_name

    def test_prefer_new_different_datasets(self, cache_handler):
        """Test that prefer_new passes dataset_name to storage correctly."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        with patch.object(cache_handler._storage, "is_config_seen") as mock_seen:
            mock_seen.return_value = False

            cache_handler.apply_policy(
                configs=[config],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="dataset1",
            )

            # Verify dataset_name was passed to storage
            mock_seen.assert_called_once()
            call_args = mock_seen.call_args
            assert call_args[0][1] == "dataset1"  # dataset_name


class TestReuseCachedPolicy:
    """Test 'reuse_cached' cache policy (deferred for v1)."""

    def test_reuse_cached_behaves_like_prefer_new(self, cache_handler):
        """Test that reuse_cached currently behaves like prefer_new (v1 limitation)."""
        config1 = {"temperature": 0.3, "model": "gpt-3.5"}
        config2 = {"temperature": 0.5, "model": "gpt-4"}

        # Mock storage to return False (configs not seen)
        with patch.object(cache_handler._storage, "is_config_seen", return_value=False):
            result1 = cache_handler.apply_policy(
                configs=[config1, config2],
                cache_policy="reuse_cached",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            assert len(result1) == 2

        # Mock storage to return True (configs seen) - should filter
        cache_handler.reset_stats()
        with patch.object(cache_handler._storage, "is_config_seen", return_value=True):
            result2 = cache_handler.apply_policy(
                configs=[config1, config2],
                cache_policy="reuse_cached",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            assert len(result2) == 0  # Filtered (not reused in v1)
            assert cache_handler.configs_deduplicated == 2


class TestStatisticsTracking:
    """Test statistics tracking functionality."""

    def test_configs_deduplicated_counter(self, cache_handler):
        """Test that configs_deduplicated counter increments correctly."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        # Mock storage to mark all configs as seen
        with patch.object(cache_handler._storage, "is_config_seen", return_value=True):
            result = cache_handler.apply_policy(
                configs=[config, config, config],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            assert len(result) == 0
            assert cache_handler.configs_deduplicated == 3

    def test_configs_deduplicated_accumulation(self, cache_handler):
        """Test that configs_deduplicated accumulates across multiple calls."""
        config1 = {"temperature": 0.3, "model": "gpt-3.5"}
        config2 = {"temperature": 0.5, "model": "gpt-4"}

        # Mock storage to mark configs as seen
        with patch.object(cache_handler._storage, "is_config_seen", return_value=True):
            # First call - 2 duplicates
            cache_handler.apply_policy(
                configs=[config1, config2],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            assert cache_handler.configs_deduplicated == 2

            # Second call - 2 more duplicates
            cache_handler.apply_policy(
                configs=[config1, config2],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            assert cache_handler.configs_deduplicated == 4  # Accumulated

    def test_cache_policy_used_tracking(self, cache_handler):
        """Test that cache_policy_used is tracked correctly."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        assert cache_handler.cache_policy_used is None

        cache_handler.apply_policy(
            configs=[config],
            cache_policy="allow_repeats",
            function_name="test_func",
            dataset_name="test_dataset",
        )
        assert cache_handler.cache_policy_used == "allow_repeats"

        cache_handler.reset_stats()
        cache_handler.apply_policy(
            configs=[config],
            cache_policy="prefer_new",
            function_name="test_func",
            dataset_name="test_dataset",
        )
        assert cache_handler.cache_policy_used == "prefer_new"

    def test_reset_stats(self, cache_handler):
        """Test that reset_stats clears statistics."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        # Create some deduplicated configs
        with patch.object(cache_handler._storage, "is_config_seen", return_value=True):
            cache_handler.apply_policy(
                configs=[config, config],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )

        assert cache_handler.configs_deduplicated > 0
        assert cache_handler.cache_policy_used == "prefer_new"

        # Reset
        cache_handler.reset_stats()
        assert cache_handler.configs_deduplicated == 0
        assert cache_handler.cache_policy_used is None

    def test_deduplication_logging(self, cache_handler, caplog):
        """Test that deduplication events are logged."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        # Mock storage to mark configs as seen - should log deduplication
        with caplog.at_level("INFO"):
            with patch.object(
                cache_handler._storage, "is_config_seen", return_value=True
            ):
                cache_handler.apply_policy(
                    configs=[config, config, config],
                    cache_policy="prefer_new",
                    function_name="test_func",
                    dataset_name="test_dataset",
                )

        assert "Deduplicated 3 previously seen configs" in caplog.text


class TestThreadSafety:
    """Test thread-safety of cache policy handler."""

    def test_storage_locking(self, cache_handler):
        """Test that storage locking is used for thread-safe deduplication."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        # Mock the storage to verify locking
        with patch.object(cache_handler._storage, "acquire_lock") as mock_lock:
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock()

            cache_handler.apply_policy(
                configs=[config],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )

            # Verify lock was acquired with correct parameters
            mock_lock.assert_called_once_with(
                "dedup_test_func_test_dataset", timeout=5.0
            )

    def test_lock_name_format(self, cache_handler):
        """Test that lock names are formatted correctly."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        with patch.object(cache_handler._storage, "acquire_lock") as mock_lock:
            mock_lock.return_value.__enter__ = MagicMock()
            mock_lock.return_value.__exit__ = MagicMock()

            # Test with different function/dataset names
            cache_handler.apply_policy(
                configs=[config],
                cache_policy="prefer_new",
                function_name="my_function",
                dataset_name="my_dataset",
            )

            mock_lock.assert_called_with("dedup_my_function_my_dataset", timeout=5.0)


class TestConfigSpaceKeys:
    """Test config space keys handling."""

    def test_with_config_space_keys(self, traigent_config):
        """Test that config space keys are used for hashing when available."""
        optimizer = MockOptimizer(
            config_space={"temperature": [0.3, 0.5], "model": ["gpt-3.5"]}
        )
        handler = CachePolicyHandler(
            traigent_config=traigent_config, optimizer=optimizer
        )

        config_keys = handler._get_config_keys()
        assert config_keys == ["temperature", "model"]

    def test_without_config_space_keys(self, traigent_config):
        """Test behavior when optimizer has no config_space."""
        optimizer = MockOptimizer(config_space=None)
        delattr(optimizer, "config_space")
        handler = CachePolicyHandler(
            traigent_config=traigent_config, optimizer=optimizer
        )

        config_keys = handler._get_config_keys()
        assert config_keys is None

        # Handler should still work without config space keys
        config = {"temperature": 0.5}
        result = handler.apply_policy(
            configs=[config],
            cache_policy="prefer_new",
            function_name="test_func",
            dataset_name="test_dataset",
        )
        assert len(result) == 1


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_empty_config_list(self, cache_handler):
        """Test behavior with empty config list."""
        result = cache_handler.apply_policy(
            configs=[],
            cache_policy="prefer_new",
            function_name="test_func",
            dataset_name="test_dataset",
        )

        assert result == []
        assert cache_handler.configs_deduplicated == 0

    def test_single_config(self, cache_handler):
        """Test behavior with single config."""
        config = {"temperature": 0.5, "model": "gpt-4"}

        result = cache_handler.apply_policy(
            configs=[config],
            cache_policy="prefer_new",
            function_name="test_func",
            dataset_name="test_dataset",
        )

        assert len(result) == 1
        assert result[0] == config

    def test_none_values_in_config(self, cache_handler):
        """Test handling of None values in configurations."""
        config = {"temperature": 0.5, "model": None}

        # Mock storage - first not seen, then seen
        with patch.object(cache_handler._storage, "is_config_seen", return_value=False):
            result1 = cache_handler.apply_policy(
                configs=[config],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            assert len(result1) == 1

        # Second pass with mock marking it as seen
        cache_handler.reset_stats()
        with patch.object(cache_handler._storage, "is_config_seen", return_value=True):
            result2 = cache_handler.apply_policy(
                configs=[config],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            assert len(result2) == 0

    def test_config_with_extra_keys(self, cache_handler):
        """Test that configs with extra keys beyond config_space are handled."""
        config1 = {"temperature": 0.5, "model": "gpt-4", "extra_key": "value1"}
        config2 = {"temperature": 0.5, "model": "gpt-4", "extra_key": "value2"}

        # First pass - not seen
        with patch.object(cache_handler._storage, "is_config_seen", return_value=False):
            result1 = cache_handler.apply_policy(
                configs=[config1],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            assert len(result1) == 1

        # Config2 has different extra_key but same core keys
        # With config_keys, only temperature and model are compared by storage
        cache_handler.reset_stats()
        with patch.object(cache_handler._storage, "is_config_seen", return_value=True):
            result2 = cache_handler.apply_policy(
                configs=[config2],
                cache_policy="prefer_new",
                function_name="test_func",
                dataset_name="test_dataset",
            )
            # Should be filtered if storage marks it as seen
            assert len(result2) == 0
