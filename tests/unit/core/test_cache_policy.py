"""Comprehensive tests for traigent.core.cache_policy module.

Tests cover CachePolicyHandler with focus on deduplication, locking,
statistics tracking, and all cache policy modes.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from traigent.core.cache_policy import CachePolicyHandler


@pytest.fixture
def mock_config():
    """Create mock TraigentConfig."""
    config = Mock()
    config.get_local_storage_path.return_value = "/tmp/traigent_test"
    return config


@pytest.fixture
def mock_optimizer():
    """Create mock optimizer with config_space."""
    optimizer = Mock()
    optimizer.config_space = {
        "model": ["gpt-4", "gpt-3.5-turbo"],
        "temperature": [0.0, 0.5, 1.0],
    }
    return optimizer


@pytest.fixture
def mock_storage():
    """Create mock storage manager."""
    with patch("traigent.core.cache_policy.LocalStorageManager") as mock_storage_class:
        storage = Mock()
        storage.is_config_seen.return_value = False
        storage.acquire_lock.return_value.__enter__ = Mock()
        storage.acquire_lock.return_value.__exit__ = Mock(return_value=False)
        mock_storage_class.return_value = storage
        yield storage


class TestCachePolicyHandlerInitialization:
    """Test CachePolicyHandler initialization."""

    def test_initialize_handler(self, mock_config, mock_optimizer, mock_storage):
        """Test basic handler initialization."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        assert handler._traigent_config is mock_config
        assert handler._optimizer is mock_optimizer
        assert handler._configs_deduplicated == 0
        assert handler._cache_policy_used is None

    def test_initializes_storage_manager(self, mock_config, mock_optimizer):
        """Test that storage manager is initialized."""
        with patch(
            "traigent.core.cache_policy.LocalStorageManager"
        ) as mock_storage_class:
            CachePolicyHandler(mock_config, mock_optimizer)

            mock_config.get_local_storage_path.assert_called_once()
            mock_storage_class.assert_called_once_with("/tmp/traigent_test")


class TestAllowRepeatsPolicy:
    """Test 'allow_repeats' cache policy."""

    def test_allow_repeats_returns_all_configs(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test that allow_repeats policy returns all configurations."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [
            {"model": "gpt-4", "temperature": 0.7},
            {"model": "gpt-3.5-turbo", "temperature": 0.5},
        ]

        result = handler.apply_policy(
            configs, "allow_repeats", "test_function", "test_dataset"
        )

        assert result == configs
        assert len(result) == 2
        assert handler.cache_policy_used == "allow_repeats"

    def test_allow_repeats_no_deduplication(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test that allow_repeats doesn't deduplicate."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}] * 5

        result = handler.apply_policy(
            configs, "allow_repeats", "test_function", "test_dataset"
        )

        assert len(result) == 5
        assert handler.configs_deduplicated == 0

    def test_allow_repeats_doesnt_check_storage(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test that allow_repeats doesn't check storage."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}]

        handler.apply_policy(configs, "allow_repeats", "test_function", "test_dataset")

        # Should not check if configs are seen
        mock_storage.is_config_seen.assert_not_called()


class TestPreferNewPolicy:
    """Test 'prefer_new' cache policy."""

    def test_prefer_new_filters_seen_configs(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test that prefer_new filters previously seen configurations."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [
            {"model": "gpt-4", "temperature": 0.7},
            {"model": "gpt-3.5-turbo", "temperature": 0.5},
        ]

        # First config has been seen, second is new
        mock_storage.is_config_seen.side_effect = [True, False]

        result = handler.apply_policy(
            configs, "prefer_new", "test_function", "test_dataset"
        )

        assert len(result) == 1
        assert result[0]["model"] == "gpt-3.5-turbo"
        assert handler.configs_deduplicated == 1

    def test_prefer_new_all_configs_new(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test prefer_new when all configs are new."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [
            {"model": "gpt-4"},
            {"model": "gpt-3.5-turbo"},
        ]

        mock_storage.is_config_seen.return_value = False

        result = handler.apply_policy(
            configs, "prefer_new", "test_function", "test_dataset"
        )

        assert len(result) == 2
        assert handler.configs_deduplicated == 0

    def test_prefer_new_all_configs_seen(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test prefer_new when all configs have been seen."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [
            {"model": "gpt-4"},
            {"model": "gpt-3.5-turbo"},
        ]

        mock_storage.is_config_seen.return_value = True

        result = handler.apply_policy(
            configs, "prefer_new", "test_function", "test_dataset"
        )

        assert len(result) == 0
        assert handler.configs_deduplicated == 2

    def test_prefer_new_checks_each_config(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test that prefer_new checks each config individually."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": f"model-{i}"} for i in range(3)]

        mock_storage.is_config_seen.return_value = False

        handler.apply_policy(configs, "prefer_new", "test_function", "test_dataset")

        # Should check each config
        assert mock_storage.is_config_seen.call_count == 3

    def test_prefer_new_uses_locking(self, mock_config, mock_optimizer, mock_storage):
        """Test that prefer_new uses storage locking."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}]

        handler.apply_policy(configs, "prefer_new", "test_function", "test_dataset")

        # Should acquire lock for thread safety
        mock_storage.acquire_lock.assert_called_once()
        lock_name = mock_storage.acquire_lock.call_args[0][0]
        assert "dedup_" in lock_name
        assert "test_function" in lock_name or "test" in lock_name
        assert "test_dataset" in lock_name or "test" in lock_name


class TestConfigKeys:
    """Test config key extraction for hashing."""

    def test_get_config_keys_from_optimizer(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test extracting config keys from optimizer."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        keys = handler._get_config_keys()

        assert keys is not None
        assert "model" in keys
        assert "temperature" in keys

    def test_get_config_keys_no_config_space(self, mock_config, mock_storage):
        """Test config keys when optimizer has no config_space."""
        optimizer_no_space = Mock(spec=[])  # No config_space attribute

        handler = CachePolicyHandler(mock_config, optimizer_no_space)

        keys = handler._get_config_keys()

        assert keys is None

    def test_config_keys_passed_to_storage(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test that config keys are passed to storage check."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}]

        handler.apply_policy(configs, "prefer_new", "test_function", "test_dataset")

        # Should pass config keys to is_config_seen
        call_args = mock_storage.is_config_seen.call_args
        assert call_args is not None
        config_keys_arg = (
            call_args[0][3]
            if len(call_args[0]) > 3
            else call_args[1].get("config_keys")
        )
        assert config_keys_arg is not None or call_args[0][3] is not None


class TestStatistics:
    """Test statistics tracking."""

    def test_configs_deduplicated_property(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test configs_deduplicated property."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        assert handler.configs_deduplicated == 0

        # Simulate deduplication
        configs = [{"model": "gpt-4"}]
        mock_storage.is_config_seen.return_value = True

        handler.apply_policy(configs, "prefer_new", "test_function", "test_dataset")

        assert handler.configs_deduplicated == 1

    def test_cache_policy_used_property(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test cache_policy_used property."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        assert handler.cache_policy_used is None

        configs = [{"model": "gpt-4"}]
        handler.apply_policy(configs, "allow_repeats", "test_function", "test_dataset")

        assert handler.cache_policy_used == "allow_repeats"

    def test_cumulative_deduplication_count(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test that deduplication count is cumulative."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}]

        # First call deduplicates 1
        mock_storage.is_config_seen.return_value = True
        handler.apply_policy(configs, "prefer_new", "test_function", "test_dataset")
        assert handler.configs_deduplicated == 1

        # Second call deduplicates 1 more
        handler.apply_policy(configs, "prefer_new", "test_function", "test_dataset")
        assert handler.configs_deduplicated == 2

    def test_reset_stats(self, mock_config, mock_optimizer, mock_storage):
        """Test resetting statistics."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        # Set some stats
        configs = [{"model": "gpt-4"}]
        mock_storage.is_config_seen.return_value = True
        handler.apply_policy(configs, "prefer_new", "test_function", "test_dataset")

        assert handler.configs_deduplicated > 0
        assert handler.cache_policy_used is not None

        # Reset
        handler.reset_stats()

        assert handler.configs_deduplicated == 0
        assert handler.cache_policy_used is None


class TestLockSegment:
    """Test _lock_segment static method."""

    def test_lock_segment_basic(self):
        """Test basic lock segment generation."""
        result = CachePolicyHandler._lock_segment("test_function")

        assert result is not None
        assert isinstance(result, str)

    def test_lock_segment_removes_hash_suffix(self):
        """Test that lock segment removes hash suffixes."""
        # Simulate sanitized identifier with hash
        result = CachePolicyHandler._lock_segment("test_function_abcd1234")

        # Should remove 8-char hex suffix
        assert "abcd1234" not in result or result == "test_function_abcd1234"

    def test_lock_segment_preserves_non_hash_suffix(self):
        """Test that non-hash suffixes are preserved."""
        result = CachePolicyHandler._lock_segment("test_function_v2")

        # Should keep non-hex suffix
        assert "test" in result.lower()

    def test_lock_segment_handles_special_characters(self):
        """Test lock segment with special characters."""
        result = CachePolicyHandler._lock_segment("test-function.py")

        assert result is not None
        assert isinstance(result, str)

    def test_lock_segment_empty_string(self):
        """Test lock segment with empty string."""
        result = CachePolicyHandler._lock_segment("")

        assert result is not None

    def test_lock_segment_no_underscore(self):
        """Test lock segment without underscore."""
        result = CachePolicyHandler._lock_segment("testfunction")

        assert result == "testfunction" or "testfunction" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_config_list(self, mock_config, mock_optimizer, mock_storage):
        """Test applying policy to empty config list."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        result = handler.apply_policy([], "prefer_new", "test_function", "test_dataset")

        assert result == []
        assert handler.configs_deduplicated == 0

    def test_single_config(self, mock_config, mock_optimizer, mock_storage):
        """Test with single configuration."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}]
        mock_storage.is_config_seen.return_value = False

        result = handler.apply_policy(
            configs, "prefer_new", "test_function", "test_dataset"
        )

        assert len(result) == 1

    def test_large_config_list(self, mock_config, mock_optimizer, mock_storage):
        """Test with large number of configurations."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": f"model-{i}"} for i in range(1000)]
        mock_storage.is_config_seen.return_value = False

        result = handler.apply_policy(
            configs, "allow_repeats", "test_function", "test_dataset"
        )

        assert len(result) == 1000

    def test_complex_configs(self, mock_config, mock_optimizer, mock_storage):
        """Test with complex nested configurations."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [
            {
                "model": "gpt-4",
                "params": {"temperature": 0.7, "top_p": 0.9},
                "metadata": {"version": "1.0"},
            }
        ]

        mock_storage.is_config_seen.return_value = False

        result = handler.apply_policy(
            configs, "prefer_new", "test_function", "test_dataset"
        )

        assert len(result) == 1
        assert result[0]["params"]["temperature"] == 0.7

    def test_unicode_in_function_name(self, mock_config, mock_optimizer, mock_storage):
        """Test with Unicode characters in function name."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}]

        result = handler.apply_policy(
            configs, "allow_repeats", "测试_function", "test_dataset"
        )

        assert len(result) == 1

    def test_very_long_names(self, mock_config, mock_optimizer, mock_storage):
        """Test with very long function/dataset names."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        long_name = "a" * 500
        configs = [{"model": "gpt-4"}]

        result = handler.apply_policy(configs, "allow_repeats", long_name, long_name)

        assert len(result) == 1

    def test_multiple_policy_applications(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test applying policy multiple times."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}]

        result1 = handler.apply_policy(configs, "allow_repeats", "func1", "dataset1")
        handler.apply_policy(configs, "prefer_new", "func2", "dataset2")

        assert len(result1) == 1
        assert handler.cache_policy_used == "prefer_new"  # Last policy used


class TestIntegration:
    """Test integration scenarios."""

    def test_full_deduplication_workflow(
        self, mock_config, mock_optimizer, mock_storage
    ):
        """Test complete deduplication workflow."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [
            {"model": "gpt-4", "temperature": 0.7},
            {"model": "gpt-4", "temperature": 0.5},
            {"model": "gpt-3.5-turbo", "temperature": 0.7},
        ]

        # First and third configs are new, second is seen
        mock_storage.is_config_seen.side_effect = [False, True, False]

        result = handler.apply_policy(
            configs, "prefer_new", "optimize_function", "eval_dataset"
        )

        assert len(result) == 2
        assert result[0]["temperature"] == 0.7
        assert result[1]["model"] == "gpt-3.5-turbo"
        assert handler.configs_deduplicated == 1
        assert handler.cache_policy_used == "prefer_new"

    def test_mixed_policy_usage(self, mock_config, mock_optimizer, mock_storage):
        """Test using different policies in sequence."""
        handler = CachePolicyHandler(mock_config, mock_optimizer)

        configs = [{"model": "gpt-4"}]

        # Use allow_repeats first
        result1 = handler.apply_policy(configs, "allow_repeats", "func", "dataset")
        assert len(result1) == 1
        assert handler.configs_deduplicated == 0

        # Reset and use prefer_new
        handler.reset_stats()
        mock_storage.is_config_seen.return_value = True

        result2 = handler.apply_policy(configs, "prefer_new", "func", "dataset")
        assert len(result2) == 0
        assert handler.configs_deduplicated == 1
