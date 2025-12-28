"""Comprehensive unit tests for integration configuration system."""

from dataclasses import fields
from unittest.mock import patch

import pytest

# Import the configuration components
from traigent.integrations.config import (
    FrameworkConstraints,
    IntegrationConfig,
    ParameterConstraints,
    configure_integrations,
    integration_config,
)
from traigent.utils.exceptions import ValidationError as ValidationException


@pytest.fixture(autouse=True)
def reset_integration_config():
    """Reset integration_config to defaults before and after each test.

    This ensures test isolation when tests modify the global integration_config.
    """
    # Save original values
    original_values = {
        f.name: getattr(integration_config, f.name) for f in fields(IntegrationConfig)
    }

    yield

    # Restore original values after test
    for name, value in original_values.items():
        setattr(integration_config, name, value)


class TestIntegrationConfig:
    """Test IntegrationConfig dataclass and default values."""

    def test_integration_config_default_values(self):
        """Test that IntegrationConfig has correct default values."""
        config = IntegrationConfig()

        # Discovery settings
        assert config.auto_discover is True
        assert config.discovery_cache_ttl == 3600
        assert config.cache_discovered_classes is True

        # Override behavior
        assert config.strict_mode is False
        assert config.fuzzy_matching_enabled is True
        assert config.fuzzy_matching_threshold == 0.8

        # Validation
        assert config.validate_types is True
        assert config.validate_values is True
        assert config.auto_convert_types is True

        # Compatibility
        assert config.version_check is True
        assert config.warn_on_deprecated is True
        assert config.auto_migrate_parameters is True

        # Performance
        assert config.max_fallback_attempts == 4
        assert config.log_override_details is False

        # Safety
        assert config.allow_unknown_parameters is False
        assert config.sanitize_parameters is True

    def test_integration_config_custom_values(self):
        """Test IntegrationConfig with custom values."""
        config = IntegrationConfig(
            auto_discover=False,
            discovery_cache_ttl=7200,
            strict_mode=True,
            fuzzy_matching_threshold=0.9,
            max_fallback_attempts=10,
        )

        assert config.auto_discover is False
        assert config.discovery_cache_ttl == 7200
        assert config.strict_mode is True
        assert config.fuzzy_matching_threshold == 0.9
        assert config.max_fallback_attempts == 10

        # Unchanged defaults
        assert config.validate_types is True
        assert config.sanitize_parameters is True

    def test_integration_config_dataclass_features(self):
        """Test dataclass features of IntegrationConfig."""
        config1 = IntegrationConfig(strict_mode=True)
        config2 = IntegrationConfig(strict_mode=True)
        config3 = IntegrationConfig(strict_mode=False)

        # Test equality
        assert config1 == config2
        assert config1 != config3

        # Test repr
        repr_str = repr(config1)
        assert "IntegrationConfig" in repr_str
        assert "strict_mode=True" in repr_str


class TestParameterConstraints:
    """Test ParameterConstraints dataclass."""

    def test_parameter_constraints_defaults(self):
        """Test ParameterConstraints default values."""
        constraint = ParameterConstraints()

        assert constraint.type is None
        assert constraint.min_value is None
        assert constraint.max_value is None
        assert constraint.allowed_values is None
        assert constraint.required is False
        assert constraint.deprecated is False
        assert constraint.deprecated_message is None
        assert constraint.aliases == []

    def test_parameter_constraints_with_values(self):
        """Test ParameterConstraints with specific values."""
        constraint = ParameterConstraints(
            type=float,
            min_value=0.0,
            max_value=1.0,
            required=True,
            aliases=["temp", "t"],
        )

        assert constraint.type is float
        assert constraint.min_value == 0.0
        assert constraint.max_value == 1.0
        assert constraint.required is True
        assert constraint.aliases == ["temp", "t"]

    def test_parameter_constraints_allowed_values(self):
        """Test ParameterConstraints with allowed values."""
        allowed_models = {"gpt-4", "gpt-3.5-turbo", "claude-2"}
        constraint = ParameterConstraints(type=str, allowed_values=allowed_models)

        assert constraint.type is str
        assert constraint.allowed_values == allowed_models

    def test_parameter_constraints_deprecation(self):
        """Test ParameterConstraints with deprecation."""
        constraint = ParameterConstraints(
            deprecated=True, deprecated_message="Use 'new_parameter' instead"
        )

        assert constraint.deprecated is True
        assert constraint.deprecated_message == "Use 'new_parameter' instead"


class TestFrameworkConstraintsCommon:
    """Test common framework constraints."""

    def test_common_constraints_exist(self):
        """Test that common constraints are properly defined."""
        common = FrameworkConstraints.COMMON_CONSTRAINTS

        # Check essential parameters exist
        assert "temperature" in common
        assert "top_p" in common
        assert "top_k" in common
        assert "max_tokens" in common
        assert "frequency_penalty" in common
        assert "presence_penalty" in common
        assert "n" in common
        assert "seed" in common

    def test_temperature_constraint(self):
        """Test temperature parameter constraint."""
        temp_constraint = FrameworkConstraints.COMMON_CONSTRAINTS["temperature"]

        assert temp_constraint.type is float
        assert temp_constraint.min_value == 0.0
        assert temp_constraint.max_value == 2.0

    def test_top_p_constraint(self):
        """Test top_p parameter constraint."""
        top_p_constraint = FrameworkConstraints.COMMON_CONSTRAINTS["top_p"]

        assert top_p_constraint.type is float
        assert top_p_constraint.min_value == 0.0
        assert top_p_constraint.max_value == 1.0

    def test_max_tokens_constraint(self):
        """Test max_tokens parameter constraint and aliases."""
        max_tokens_constraint = FrameworkConstraints.COMMON_CONSTRAINTS["max_tokens"]

        assert max_tokens_constraint.type is int
        assert max_tokens_constraint.min_value == 1
        assert "max_length" in max_tokens_constraint.aliases
        assert "max_new_tokens" in max_tokens_constraint.aliases
        assert "max_tokens_to_sample" in max_tokens_constraint.aliases

    def test_penalty_constraints(self):
        """Test frequency and presence penalty constraints."""
        freq_constraint = FrameworkConstraints.COMMON_CONSTRAINTS["frequency_penalty"]
        pres_constraint = FrameworkConstraints.COMMON_CONSTRAINTS["presence_penalty"]

        for constraint in [freq_constraint, pres_constraint]:
            assert constraint.type is float
            assert constraint.min_value == -2.0
            assert constraint.max_value == 2.0

    def test_integer_constraints(self):
        """Test integer parameter constraints."""
        top_k_constraint = FrameworkConstraints.COMMON_CONSTRAINTS["top_k"]
        n_constraint = FrameworkConstraints.COMMON_CONSTRAINTS["n"]

        assert top_k_constraint.type is int
        assert top_k_constraint.min_value == 1

        assert n_constraint.type is int
        assert n_constraint.min_value == 1
        assert "num_completions" in n_constraint.aliases
        assert "best_of" in n_constraint.aliases


class TestFrameworkConstraintsOpenAI:
    """Test OpenAI-specific framework constraints."""

    def test_openai_constraints_include_common(self):
        """Test that OpenAI constraints include all common constraints."""
        openai = FrameworkConstraints.OPENAI_CONSTRAINTS
        common = FrameworkConstraints.COMMON_CONSTRAINTS

        for param_name in common:
            assert param_name in openai
            # Common constraints should be preserved
            assert openai[param_name] == common[param_name]

    def test_openai_model_constraint(self):
        """Test OpenAI model parameter constraint.

        Note: Model validation now uses the model discovery service (dynamic),
        not hardcoded allowed_values. The constraint type should be str.
        """
        model_constraint = FrameworkConstraints.OPENAI_CONSTRAINTS["model"]

        assert model_constraint.type is str
        # allowed_values is now None - validation is done via model discovery service
        assert model_constraint.allowed_values is None

    def test_openai_specific_constraints(self):
        """Test OpenAI-specific constraints."""
        openai = FrameworkConstraints.OPENAI_CONSTRAINTS

        # logit_bias constraint
        assert "logit_bias" in openai
        assert openai["logit_bias"].type is dict

        # response_format constraint
        assert "response_format" in openai
        assert openai["response_format"].type is dict


class TestFrameworkConstraintsAnthropic:
    """Test Anthropic-specific framework constraints."""

    def test_anthropic_constraints_include_common(self):
        """Test that Anthropic constraints include common constraints."""
        anthropic = FrameworkConstraints.ANTHROPIC_CONSTRAINTS

        # Most common constraints should be included
        for param_name in ["temperature", "top_p", "top_k", "seed"]:
            assert param_name in anthropic

    def test_anthropic_model_constraint(self):
        """Test Anthropic model parameter constraint.

        Note: Model validation now uses the model discovery service (dynamic),
        not hardcoded allowed_values. The constraint type should be str.
        """
        model_constraint = FrameworkConstraints.ANTHROPIC_CONSTRAINTS["model"]

        assert model_constraint.type is str
        # allowed_values is now None - validation is done via model discovery service
        assert model_constraint.allowed_values is None

    def test_anthropic_max_tokens_override(self):
        """Test Anthropic max_tokens constraint.

        Note: Anthropic's current API supports up to 200000 tokens for Claude 3.5 models.
        """
        max_tokens_constraint = FrameworkConstraints.ANTHROPIC_CONSTRAINTS["max_tokens"]

        assert max_tokens_constraint.type is int
        assert max_tokens_constraint.min_value == 1
        assert max_tokens_constraint.max_value == 200000  # Current Anthropic API limit
        assert "max_tokens_to_sample" in max_tokens_constraint.aliases


class TestFrameworkConstraintsRetrieval:
    """Test framework constraint retrieval methods."""

    def test_get_constraints_for_openai(self):
        """Test getting constraints for OpenAI framework."""
        constraints = FrameworkConstraints.get_constraints_for_framework("openai")

        assert constraints == FrameworkConstraints.OPENAI_CONSTRAINTS
        assert "model" in constraints
        # Model validation is now done via discovery service, not allowed_values
        assert constraints["model"].type is str
        assert constraints["model"].allowed_values is None

    def test_get_constraints_for_anthropic(self):
        """Test getting constraints for Anthropic framework."""
        constraints = FrameworkConstraints.get_constraints_for_framework("anthropic")

        assert constraints == FrameworkConstraints.ANTHROPIC_CONSTRAINTS
        assert "model" in constraints
        # Model validation is now done via discovery service, not allowed_values
        assert constraints["model"].type is str
        assert constraints["model"].allowed_values is None

    def test_get_constraints_case_insensitive(self):
        """Test that framework name matching is case-insensitive."""
        openai_upper = FrameworkConstraints.get_constraints_for_framework("OPENAI")
        openai_mixed = FrameworkConstraints.get_constraints_for_framework("OpenAI")
        openai_lower = FrameworkConstraints.get_constraints_for_framework("openai")

        assert openai_upper == openai_lower == openai_mixed
        assert openai_upper == FrameworkConstraints.OPENAI_CONSTRAINTS

    def test_get_constraints_unknown_framework(self):
        """Test getting constraints for unknown framework returns common constraints."""
        unknown_constraints = FrameworkConstraints.get_constraints_for_framework(
            "unknown_framework"
        )

        assert unknown_constraints == FrameworkConstraints.COMMON_CONSTRAINTS

        # Test with completely made-up framework
        made_up_constraints = FrameworkConstraints.get_constraints_for_framework(
            "fictitious_llm"
        )
        assert made_up_constraints == FrameworkConstraints.COMMON_CONSTRAINTS


class TestParameterValidation:
    """Test parameter validation functionality."""

    def test_validate_parameter_valid_values(self):
        """Test validation of valid parameter values."""
        # Valid temperature
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", 0.7)
        assert issues == []

        # Valid max_tokens
        issues = FrameworkConstraints.validate_parameter("openai", "max_tokens", 100)
        assert issues == []

        # Valid model
        issues = FrameworkConstraints.validate_parameter("openai", "model", "gpt-4")
        assert issues == []

    def test_validate_parameter_type_errors(self):
        """Test validation with type errors."""
        # Wrong type for temperature (should be float)
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", "0.7")
        assert len(issues) == 1
        assert "expected type float" in issues[0]
        assert "got str" in issues[0]

        # Wrong type for max_tokens (should be int)
        issues = FrameworkConstraints.validate_parameter("openai", "max_tokens", 100.5)
        assert len(issues) == 1
        assert "expected type int" in issues[0]
        assert "got float" in issues[0]

    def test_validate_parameter_range_errors(self):
        """Test validation with range errors."""
        # Temperature too low
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", -0.5)
        assert len(issues) == 1
        assert "below minimum" in issues[0]

        # Temperature too high
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", 2.5)
        assert len(issues) == 1
        assert "above maximum" in issues[0]

        # max_tokens too low
        issues = FrameworkConstraints.validate_parameter("openai", "max_tokens", 0)
        assert len(issues) == 1
        assert "below minimum" in issues[0]

    def test_validate_parameter_allowed_values_error(self):
        """Test validation with disallowed values.

        Note: Model validation is now done via the model discovery service,
        not hardcoded allowed_values. Since allowed_values is None,
        any string model value passes basic FrameworkConstraints validation.
        The actual model validation happens at the plugin level.
        """
        # With dynamic model validation, FrameworkConstraints doesn't validate model names
        issues = FrameworkConstraints.validate_parameter(
            "openai", "model", "invalid-model"
        )
        assert len(issues) == 0

    def test_validate_parameter_with_aliases(self):
        """Test validation using parameter aliases."""
        # Test max_length alias for max_tokens
        issues = FrameworkConstraints.validate_parameter("openai", "max_length", 100)
        assert issues == []  # Should be valid

        # Test invalid value through alias
        issues = FrameworkConstraints.validate_parameter("openai", "max_length", 0)
        assert len(issues) == 1
        assert "below minimum" in issues[0]
        assert "max_tokens" in issues[0]  # Should refer to canonical name

    def test_validate_parameter_unknown_parameter(self):
        """Test validation of unknown parameters."""
        # Unknown parameter should return no issues
        issues = FrameworkConstraints.validate_parameter(
            "openai", "unknown_param", "any_value"
        )
        assert issues == []

    def test_validate_parameter_deprecation_warning(self):
        """Test that deprecation warnings are logged."""
        # Create a deprecated constraint for testing
        with patch.object(
            FrameworkConstraints, "get_constraints_for_framework"
        ) as mock_get:
            deprecated_constraint = ParameterConstraints(
                type=str, deprecated=True, deprecated_message="Use new_param instead"
            )
            mock_constraints = {"old_param": deprecated_constraint}
            mock_get.return_value = mock_constraints

            with patch("traigent.integrations.config.logger") as mock_logger:
                issues = FrameworkConstraints.validate_parameter(
                    "test", "old_param", "value"
                )

                # Should not have validation issues
                assert issues == []

                # Should log deprecation warning
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "Use new_param instead" in warning_call

    def test_validate_parameter_multiple_issues(self):
        """Test validation with multiple issues (early return on type error)."""
        # Type error should prevent range checking
        issues = FrameworkConstraints.validate_parameter(
            "openai", "temperature", "invalid"
        )
        assert len(issues) == 1  # Only type error, no range error
        assert "expected type float" in issues[0]


class TestGlobalConfiguration:
    """Test global configuration management."""

    def test_global_integration_config_exists(self):
        """Test that global integration_config exists and is correct type."""
        assert integration_config is not None
        assert isinstance(integration_config, IntegrationConfig)

    def test_configure_integrations_valid_options(self):
        """Test configuring integrations with valid options."""
        original_auto_discover = integration_config.auto_discover
        original_strict_mode = integration_config.strict_mode

        # Configure some options
        configure_integrations(
            auto_discover=not original_auto_discover,
            strict_mode=not original_strict_mode,
            discovery_cache_ttl=7200,
        )

        # Check that values were updated
        assert integration_config.auto_discover == (not original_auto_discover)
        assert integration_config.strict_mode == (not original_strict_mode)
        assert integration_config.discovery_cache_ttl == 7200
        # Cleanup handled by reset_integration_config fixture

    def test_configure_integrations_invalid_options(self):
        """Test configuring integrations with invalid options."""
        with patch("traigent.integrations.config.logger") as mock_logger:
            configure_integrations(invalid_option="should_warn", another_invalid=123)

            # Should warn about invalid options (at least one warning per invalid option)
            assert (
                mock_logger.warning.call_count >= 2
            ), f"Expected at least 2 warnings, got {mock_logger.warning.call_count}"
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any(
                "invalid_option" in call for call in warning_calls
            ), f"Expected warning about 'invalid_option' in: {warning_calls}"
            assert any(
                "another_invalid" in call for call in warning_calls
            ), f"Expected warning about 'another_invalid' in: {warning_calls}"

    def test_configure_integrations_type_safety(self):
        """Test that configuration maintains type safety."""
        # Valid type change
        configure_integrations(fuzzy_matching_threshold=0.9)
        assert integration_config.fuzzy_matching_threshold == 0.9

        # Invalid type should raise and not change the value
        with pytest.raises(ValidationException):
            configure_integrations(fuzzy_matching_threshold="invalid")
        assert integration_config.fuzzy_matching_threshold == 0.9
        # Cleanup handled by reset_integration_config fixture

    def test_configure_integrations_rejects_negative_cache_ttl(self):
        """Negative discovery_cache_ttl should raise an error."""
        with pytest.raises(ValidationException):
            configure_integrations(discovery_cache_ttl=-10)


class TestIntegrationConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_configuration_workflow(self):
        """Test complete configuration workflow."""
        # Start with default config
        config = IntegrationConfig()

        # Test validation with default settings
        assert config.validate_types is True
        assert config.strict_mode is False

        # Change to strict mode
        config.strict_mode = True
        assert config.strict_mode is True

        # Test that fuzzy matching can be disabled
        config.fuzzy_matching_enabled = False
        assert config.fuzzy_matching_enabled is False

    def test_framework_constraint_integration(self):
        """Test integration between configuration and framework constraints."""
        # Get constraints for different frameworks
        FrameworkConstraints.get_constraints_for_framework("openai")
        FrameworkConstraints.get_constraints_for_framework("anthropic")

        # Test validation across frameworks
        valid_temp = 0.7

        # Should be valid for both frameworks
        openai_issues = FrameworkConstraints.validate_parameter(
            "openai", "temperature", valid_temp
        )
        anthropic_issues = FrameworkConstraints.validate_parameter(
            "anthropic", "temperature", valid_temp
        )

        assert openai_issues == []
        assert anthropic_issues == []

        # Test framework-specific differences
        anthropic_max_tokens = 4000  # Within Anthropic limit
        openai_max_tokens = (
            4000  # Within OpenAI limit (no specific limit in common constraints)
        )

        anthropic_issues = FrameworkConstraints.validate_parameter(
            "anthropic", "max_tokens", anthropic_max_tokens
        )
        openai_issues = FrameworkConstraints.validate_parameter(
            "openai", "max_tokens", openai_max_tokens
        )

        assert anthropic_issues == []  # Should be valid
        assert openai_issues == []  # Should be valid (no max limit in common)

    def test_configuration_edge_cases(self):
        """Test configuration system edge cases."""
        # Test with boundary values
        config = IntegrationConfig(
            fuzzy_matching_threshold=0.0,  # Minimum
            max_fallback_attempts=0,  # Minimum
            discovery_cache_ttl=1,  # Very low
        )

        assert config.fuzzy_matching_threshold == 0.0
        assert config.max_fallback_attempts == 0
        assert config.discovery_cache_ttl == 1

        # Test with extreme values
        config_extreme = IntegrationConfig(
            fuzzy_matching_threshold=1.0,  # Maximum
            max_fallback_attempts=1000,  # Very high
            discovery_cache_ttl=86400,  # 24 hours
        )

        assert config_extreme.fuzzy_matching_threshold == 1.0
        assert config_extreme.max_fallback_attempts == 1000
        assert config_extreme.discovery_cache_ttl == 86400


if __name__ == "__main__":
    pytest.main([__file__])
