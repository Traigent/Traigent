"""Tests for context-based configuration management."""

from traigent.config.context import (
    ConfigurationContext,
    get_config,
    merge_with_context,
    set_config,
)
from traigent.config.types import TraigentConfig


class TestConfigContext:
    """Test suite for configuration context management."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_config()
        assert isinstance(config, TraigentConfig)

    def test_set_and_get_dict_config(self):
        """Test setting and getting dict configuration."""
        test_config = {"model": "GPT-4o", "temperature": 0.7}
        token = set_config(test_config)

        try:
            config = get_config()
            assert config == test_config
        finally:
            # Reset to avoid affecting other tests
            from traigent.config.context import config_context

            config_context.reset(token)

    def test_set_and_get_traigent_config(self):
        """Test setting and getting TraigentConfig."""
        test_config = TraigentConfig(model="GPT-4o", temperature=0.7)
        token = set_config(test_config)

        try:
            config = get_config()
            assert isinstance(config, TraigentConfig)
            assert config.model == "GPT-4o"
            assert config.temperature == 0.7
        finally:
            from traigent.config.context import config_context

            config_context.reset(token)

    def test_configuration_context_manager(self):
        """Test ConfigurationContext context manager."""
        original_config = get_config()
        test_config = {"model": "GPT-4o", "temperature": 0.8}

        with ConfigurationContext(test_config) as context_config:
            # Inside context
            current_config = get_config()
            assert current_config == test_config
            assert context_config == test_config

        # Outside context - should be restored
        restored_config = get_config()
        # Note: We can't directly compare TraigentConfig objects
        if isinstance(original_config, TraigentConfig) and isinstance(
            restored_config, TraigentConfig
        ):
            assert original_config.to_dict() == restored_config.to_dict()
        else:
            assert original_config == restored_config

    def test_nested_configuration_contexts(self):
        """Test nested configuration contexts."""
        config1 = {"model": "o4-mini"}
        config2 = {"model": "GPT-4o", "temperature": 0.7}

        with ConfigurationContext(config1):
            assert get_config() == config1

            with ConfigurationContext(config2):
                assert get_config() == config2

            # Should restore to config1
            assert get_config() == config1

    def test_merge_with_context_dict(self):
        """Test merging override with context (dict)."""
        context_config = {"model": "o4-mini", "temperature": 0.5}
        override_config = {"temperature": 0.8, "max_tokens": 1000}

        with ConfigurationContext(context_config):
            merged = merge_with_context(override_config)

            expected = {
                "model": "o4-mini",
                "temperature": 0.5,  # Context wins
                "max_tokens": 1000,
            }
            assert merged == expected

    def test_merge_with_context_traigent_config(self):
        """Test merging override with TraigentConfig context."""
        context_config = TraigentConfig(model="o4-mini", temperature=0.5)
        override_config = {"temperature": 0.8, "max_tokens": 1000}

        with ConfigurationContext(context_config):
            merged = merge_with_context(override_config)

            assert isinstance(merged, TraigentConfig)
            assert merged.model == "o4-mini"
            assert merged.temperature == 0.5
            assert merged.max_tokens == 1000

    def test_merge_with_context_no_override(self):
        """Test merging with no override returns context."""
        context_config = {"model": "GPT-4o"}

        with ConfigurationContext(context_config):
            merged = merge_with_context(None)
            assert merged == context_config

    def test_context_manager_exception_handling(self):
        """Test context manager properly handles exceptions."""
        original_config = get_config()
        test_config = {"model": "GPT-4o"}

        try:
            with ConfigurationContext(test_config):
                assert get_config() == test_config
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Should still restore original config despite exception
        restored_config = get_config()
        if isinstance(original_config, TraigentConfig) and isinstance(
            restored_config, TraigentConfig
        ):
            assert original_config.to_dict() == restored_config.to_dict()
        else:
            assert original_config == restored_config
