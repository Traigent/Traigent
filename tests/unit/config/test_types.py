"""Tests for TraigentConfig type and execution mode helpers."""

import pytest

from traigent.config.types import TraigentConfig, validate_execution_mode
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
            "execution_mode": "edge_analytics",
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

    def test_invalid_mode_string_raises_configuration_error(self) -> None:
        """Invalid mode string should raise ConfigurationError, not ValueError."""
        with pytest.raises(ConfigurationError, match="No such mode 'nonexistent_mode'"):
            validate_execution_mode("nonexistent_mode")
