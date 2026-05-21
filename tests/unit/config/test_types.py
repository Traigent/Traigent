"""Tests for TraigentConfig type and execution mode helpers."""

import pytest

from traigent.config.types import (
    ExecutionMode,
    TraigentConfig,
    resolve_execution_mode,
    validate_execution_mode,
)
from traigent.utils.exceptions import ConfigurationError, ValidationError


class TestTraigentConfig:
    """Test suite for TraigentConfig."""

    def test_create_empty_config(self):
        """Test creating empty configuration."""
        config = TraigentConfig()
        assert config.model is None
        assert config.temperature is None
        assert config.custom_params == {}

    def test_create_config_with_values(self):
        """Test creating configuration with values."""
        config = TraigentConfig(model="GPT-4o", temperature=0.7, max_tokens=1000)
        assert config.model == "GPT-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        TraigentConfig(temperature=0.0)
        TraigentConfig(temperature=1.0)
        TraigentConfig(temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValidationError, match="temperature.*below minimum"):
            TraigentConfig(temperature=-0.1)

        with pytest.raises(ValidationError, match="temperature.*exceeds maximum"):
            TraigentConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        TraigentConfig(max_tokens=1)
        TraigentConfig(max_tokens=1000)

        # Invalid max_tokens
        with pytest.raises(ValidationError, match="max_tokens.*positive"):
            TraigentConfig(max_tokens=0)

        with pytest.raises(ValidationError, match="max_tokens.*positive"):
            TraigentConfig(max_tokens=-1)

    def test_top_p_validation(self):
        """Test top_p validation."""
        # Valid top_p
        TraigentConfig(top_p=0.0)
        TraigentConfig(top_p=0.5)
        TraigentConfig(top_p=1.0)

        # Invalid top_p
        with pytest.raises(ValidationError, match="top_p.*below minimum"):
            TraigentConfig(top_p=-0.1)

        with pytest.raises(ValidationError, match="top_p.*exceeds maximum"):
            TraigentConfig(top_p=1.1)

    def test_penalty_validation(self):
        """Test penalty validation."""
        # Valid penalties
        TraigentConfig(frequency_penalty=-2.0)
        TraigentConfig(frequency_penalty=0.0)
        TraigentConfig(frequency_penalty=2.0)
        TraigentConfig(presence_penalty=-2.0)
        TraigentConfig(presence_penalty=2.0)

        # Invalid penalties
        with pytest.raises(ValidationError, match="frequency_penalty.*below minimum"):
            TraigentConfig(frequency_penalty=-2.1)

        with pytest.raises(ValidationError, match="presence_penalty.*exceeds maximum"):
            TraigentConfig(presence_penalty=2.1)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TraigentConfig(
            model="GPT-4o",
            temperature=0.7,
            custom_params={"custom_key": "custom_value"},
        )

        result = config.to_dict()
        expected = {
            "model": "GPT-4o",
            "temperature": 0.7,
            "custom_key": "custom_value",
        }
        assert result == expected

    def test_to_dict_excludes_none_values(self):
        """Test that to_dict excludes None values."""
        config = TraigentConfig(model="GPT-4o")
        result = config.to_dict()

        assert "model" in result
        assert "temperature" not in result
        assert "max_tokens" not in result

    def test_to_dict_excludes_default_execution_mode(self):
        """Default edge_analytics must not leak into partial config merges."""
        assert TraigentConfig().to_dict() == {}
        assert TraigentConfig(model="GPT-4o").to_dict() == {"model": "GPT-4o"}

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "model": "GPT-4o",
            "temperature": 0.7,
            "custom_key": "custom_value",
        }

        config = TraigentConfig.from_dict(config_dict)

        assert config.model == "GPT-4o"
        assert config.temperature == 0.7
        assert config.custom_params == {"custom_key": "custom_value"}

    def test_merge_with_config(self):
        """Test merging with another config."""
        config1 = TraigentConfig(model="gpt-4o-mini", temperature=0.5)
        config2 = TraigentConfig(temperature=0.8, max_tokens=1000)

        merged = config1.merge(config2)

        assert merged.model == "gpt-4o-mini"  # From config1
        assert merged.temperature == 0.8  # From config2 (overrides)
        assert merged.max_tokens == 1000  # From config2

    def test_merge_with_dict(self):
        """Test merging with dictionary."""
        config = TraigentConfig(model="GPT-4o", temperature=0.5)
        override = {"temperature": 0.8, "max_tokens": 1000}

        merged = config.merge(override)

        assert merged.model == "GPT-4o"
        assert merged.temperature == 0.8
        assert merged.max_tokens == 1000

    def test_merge_partial_config_preserves_execution_mode(self):
        """A default-valued override must not reset a hybrid base config."""
        base = TraigentConfig(execution_mode="hybrid", privacy_enabled=True)
        override = TraigentConfig(model="GPT-4o")

        merged = base.merge(override)

        assert merged.model == "GPT-4o"
        assert merged.execution_mode == "hybrid"
        assert merged.privacy_enabled is True

    def test_merge_dict_can_explicitly_reset_execution_mode_to_default(self):
        """Dict overrides preserve explicitly supplied default values."""
        base = TraigentConfig(execution_mode="hybrid", privacy_enabled=True)

        merged = base.merge({"execution_mode": "edge_analytics"})

        assert merged.execution_mode == "edge_analytics"
        assert merged.privacy_enabled is True

    def test_repr(self):
        """Test string representation."""
        config = TraigentConfig(model="GPT-4o", temperature=0.7)
        repr_str = repr(config)

        assert "TraigentConfig" in repr_str
        assert "model='GPT-4o'" in repr_str
        assert "temperature=0.7" in repr_str

    def test_custom_params(self):
        """Test handling of custom parameters."""
        config = TraigentConfig(
            model="GPT-4o",
            custom_params={"custom_setting": True, "api_version": "2023-07-01"},
        )

        result = config.to_dict()
        assert result["custom_setting"] is True
        assert result["api_version"] == "2023-07-01"
        assert result["model"] == "GPT-4o"


class TestValidateExecutionMode:
    """Tests for validate_execution_mode function."""

    def test_resolve_execution_mode_none_defaults_to_edge_analytics(self) -> None:
        """Omitted execution mode follows the public SDK default."""
        assert resolve_execution_mode(None) is ExecutionMode.EDGE_ANALYTICS

    def test_validate_execution_mode_none_defaults_to_edge_analytics(self) -> None:
        """Validation should accept the omitted-mode public default."""
        assert validate_execution_mode(None) is ExecutionMode.EDGE_ANALYTICS

    def test_invalid_mode_string_raises_configuration_error(self) -> None:
        """Invalid mode string should raise ConfigurationError, not ValueError."""
        with pytest.raises(ConfigurationError, match="No such mode 'nonexistent_mode'"):
            validate_execution_mode("nonexistent_mode")

    def test_privacy_alias_validates_as_hybrid(self) -> None:
        """Privacy remains a legacy alias for hybrid."""
        assert validate_execution_mode("privacy") is ExecutionMode.HYBRID

    def test_removed_standard_mode_raises_configuration_error(self) -> None:
        """The removed standard mode is rejected everywhere."""
        with pytest.raises(ConfigurationError, match="No such mode 'standard'"):
            validate_execution_mode("standard")

    def test_reserved_cloud_mode_raises_configuration_error(self) -> None:
        """Cloud remote execution is reserved and fails closed."""
        with pytest.raises(ConfigurationError, match="not available yet"):
            validate_execution_mode("cloud")

    def test_config_privacy_alias_normalizes_to_hybrid(self) -> None:
        """TraigentConfig normalizes the privacy alias and enables privacy."""
        config = TraigentConfig(execution_mode="privacy")

        assert config.execution_mode == ExecutionMode.HYBRID.value
        assert config.privacy_enabled is True

    def test_config_rejects_removed_and_reserved_modes(self) -> None:
        """TraigentConfig follows the same execution-mode contract."""
        with pytest.raises(ConfigurationError, match="No such mode 'standard'"):
            TraigentConfig(execution_mode="standard")

        with pytest.raises(ConfigurationError, match="not available yet"):
            TraigentConfig(execution_mode="cloud")
