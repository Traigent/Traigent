"""Comprehensive tests for traigent.integrations.config module.

Tests cover IntegrationConfig, ParameterConstraints, FrameworkConstraints,
and the configure_integrations function with extensive edge case coverage.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

import traigent.integrations.config as config_module
from traigent.integrations.config import (
    FrameworkConstraints,
    IntegrationConfig,
    ParameterConstraints,
    configure_integrations,
    integration_config,
)
from traigent.utils.exceptions import ValidationError


@pytest.fixture(autouse=True)
def reset_integration_config():
    """Reset integration_config to defaults before and after each test.

    This ensures test isolation when tests modify the global integration_config.
    Handles both property modifications and complete object replacement.
    """
    # Save the original object reference and its values
    original_config = config_module.integration_config
    original_values = {
        f.name: getattr(original_config, f.name) for f in fields(IntegrationConfig)
    }

    yield

    # Restore original object reference if it was replaced
    config_module.integration_config = original_config

    # Restore original values
    for name, value in original_values.items():
        setattr(config_module.integration_config, name, value)


class TestIntegrationConfig:
    """Test IntegrationConfig dataclass and default values."""

    def test_default_values(self):
        """Test that IntegrationConfig has sensible defaults."""
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

    def test_custom_values(self):
        """Test creating IntegrationConfig with custom values."""
        config = IntegrationConfig(
            auto_discover=False,
            strict_mode=True,
            fuzzy_matching_threshold=0.9,
            max_fallback_attempts=10,
        )

        assert config.auto_discover is False
        assert config.strict_mode is True
        assert config.fuzzy_matching_threshold == 0.9
        assert config.max_fallback_attempts == 10

    def test_all_boolean_fields(self):
        """Test all boolean configuration fields."""
        config = IntegrationConfig(
            auto_discover=False,
            cache_discovered_classes=False,
            strict_mode=True,
            fuzzy_matching_enabled=False,
            validate_types=False,
            validate_values=False,
            auto_convert_types=False,
            version_check=False,
            warn_on_deprecated=False,
            auto_migrate_parameters=False,
            log_override_details=True,
            allow_unknown_parameters=True,
            sanitize_parameters=False,
        )

        assert config.auto_discover is False
        assert config.cache_discovered_classes is False
        assert config.strict_mode is True
        assert config.log_override_details is True


class TestParameterConstraints:
    """Test ParameterConstraints dataclass."""

    def test_default_values(self):
        """Test ParameterConstraints with default values."""
        constraint = ParameterConstraints()

        assert constraint.type is None
        assert constraint.min_value is None
        assert constraint.max_value is None
        assert constraint.allowed_values is None
        assert constraint.required is False
        assert constraint.deprecated is False
        assert constraint.deprecated_message is None
        assert constraint.aliases == []

    def test_with_type_constraint(self):
        """Test ParameterConstraints with type constraint."""
        constraint = ParameterConstraints(type=float)
        assert constraint.type is float

    def test_with_range_constraints(self):
        """Test ParameterConstraints with min/max values."""
        constraint = ParameterConstraints(
            type=float,
            min_value=0.0,
            max_value=2.0,
        )

        assert constraint.min_value == 0.0
        assert constraint.max_value == 2.0

    def test_with_allowed_values(self):
        """Test ParameterConstraints with allowed values."""
        allowed = {"gpt-4", "gpt-3.5-turbo"}
        constraint = ParameterConstraints(
            type=str,
            allowed_values=allowed,
        )

        assert constraint.allowed_values == allowed

    def test_with_aliases(self):
        """Test ParameterConstraints with parameter aliases."""
        constraint = ParameterConstraints(
            type=int,
            aliases=["max_length", "max_new_tokens"],
        )

        assert "max_length" in constraint.aliases
        assert "max_new_tokens" in constraint.aliases

    def test_deprecated_parameter(self):
        """Test ParameterConstraints for deprecated parameter."""
        constraint = ParameterConstraints(
            deprecated=True,
            deprecated_message="Use 'new_param' instead",
        )

        assert constraint.deprecated is True
        assert "new_param" in constraint.deprecated_message

    def test_required_parameter(self):
        """Test ParameterConstraints for required parameter."""
        constraint = ParameterConstraints(required=True)
        assert constraint.required is True


class TestFrameworkConstraintsCommon:
    """Test FrameworkConstraints common constraints."""

    def test_common_constraints_temperature(self):
        """Test temperature constraint."""
        constraint = FrameworkConstraints.COMMON_CONSTRAINTS["temperature"]

        assert constraint.type is float
        assert constraint.min_value == 0.0
        assert constraint.max_value == 2.0

    def test_common_constraints_top_p(self):
        """Test top_p constraint."""
        constraint = FrameworkConstraints.COMMON_CONSTRAINTS["top_p"]

        assert constraint.type is float
        assert constraint.min_value == 0.0
        assert constraint.max_value == 1.0

    def test_common_constraints_max_tokens(self):
        """Test max_tokens constraint and aliases."""
        constraint = FrameworkConstraints.COMMON_CONSTRAINTS["max_tokens"]

        assert constraint.type is int
        assert constraint.min_value == 1
        assert "max_length" in constraint.aliases
        assert "max_new_tokens" in constraint.aliases
        assert "max_tokens_to_sample" in constraint.aliases

    def test_common_constraints_penalties(self):
        """Test frequency and presence penalty constraints."""
        freq = FrameworkConstraints.COMMON_CONSTRAINTS["frequency_penalty"]
        pres = FrameworkConstraints.COMMON_CONSTRAINTS["presence_penalty"]

        assert freq.type is float
        assert freq.min_value == -2.0
        assert freq.max_value == 2.0

        assert pres.type is float
        assert pres.min_value == -2.0
        assert pres.max_value == 2.0

    def test_common_constraints_top_k(self):
        """Test top_k constraint."""
        constraint = FrameworkConstraints.COMMON_CONSTRAINTS["top_k"]

        assert constraint.type is int
        assert constraint.min_value == 1

    def test_common_constraints_n_completions(self):
        """Test n completions constraint."""
        constraint = FrameworkConstraints.COMMON_CONSTRAINTS["n"]

        assert constraint.type is int
        assert constraint.min_value == 1
        assert "num_completions" in constraint.aliases

    def test_common_constraints_seed(self):
        """Test seed constraint."""
        constraint = FrameworkConstraints.COMMON_CONSTRAINTS["seed"]

        assert constraint.type is int
        assert "random_seed" in constraint.aliases


class TestFrameworkConstraintsOpenAI:
    """Test OpenAI-specific constraints."""

    def test_openai_inherits_common(self):
        """Test that OpenAI constraints include common constraints."""
        assert "temperature" in FrameworkConstraints.OPENAI_CONSTRAINTS
        assert "top_p" in FrameworkConstraints.OPENAI_CONSTRAINTS
        assert "max_tokens" in FrameworkConstraints.OPENAI_CONSTRAINTS

    def test_openai_model_constraint(self):
        """Test OpenAI model constraint.

        Note: Model validation now uses the model discovery service (dynamic),
        not hardcoded allowed_values. The constraint type should be str.
        """
        constraint = FrameworkConstraints.OPENAI_CONSTRAINTS["model"]

        assert constraint.type is str
        # allowed_values is now None - validation is done via model discovery service
        assert constraint.allowed_values is None

    def test_openai_specific_parameters(self):
        """Test OpenAI-specific parameters."""
        assert "logit_bias" in FrameworkConstraints.OPENAI_CONSTRAINTS
        assert "response_format" in FrameworkConstraints.OPENAI_CONSTRAINTS

        logit_constraint = FrameworkConstraints.OPENAI_CONSTRAINTS["logit_bias"]
        assert logit_constraint.type is dict


class TestFrameworkConstraintsAnthropic:
    """Test Anthropic-specific constraints."""

    def test_anthropic_inherits_common(self):
        """Test that Anthropic constraints include common constraints."""
        assert "temperature" in FrameworkConstraints.ANTHROPIC_CONSTRAINTS
        assert "top_p" in FrameworkConstraints.ANTHROPIC_CONSTRAINTS

    def test_anthropic_model_constraint(self):
        """Test Anthropic model constraint.

        Note: Model validation now uses the model discovery service (dynamic),
        not hardcoded allowed_values. The constraint type should be str.
        """
        constraint = FrameworkConstraints.ANTHROPIC_CONSTRAINTS["model"]

        assert constraint.type is str
        # allowed_values is now None - validation is done via model discovery service
        assert constraint.allowed_values is None

    def test_anthropic_max_tokens_override(self):
        """Test Anthropic max_tokens constraint.

        Note: Anthropic's current API supports up to 200000 tokens for Claude 3.5 models.
        """
        constraint = FrameworkConstraints.ANTHROPIC_CONSTRAINTS["max_tokens"]

        assert constraint.type is int
        assert constraint.min_value == 1
        assert constraint.max_value == 200000  # Current Anthropic API limit
        assert "max_tokens_to_sample" in constraint.aliases


class TestGetConstraintsForFramework:
    """Test get_constraints_for_framework method."""

    def test_get_openai_constraints(self):
        """Test getting OpenAI constraints."""
        constraints = FrameworkConstraints.get_constraints_for_framework("openai")

        assert "temperature" in constraints
        assert "model" in constraints
        assert constraints == FrameworkConstraints.OPENAI_CONSTRAINTS

    def test_get_openai_constraints_case_insensitive(self):
        """Test framework name is case-insensitive."""
        constraints = FrameworkConstraints.get_constraints_for_framework("OpenAI")
        assert constraints == FrameworkConstraints.OPENAI_CONSTRAINTS

    def test_get_anthropic_constraints(self):
        """Test getting Anthropic constraints."""
        constraints = FrameworkConstraints.get_constraints_for_framework("anthropic")

        assert "temperature" in constraints
        assert "model" in constraints
        assert constraints == FrameworkConstraints.ANTHROPIC_CONSTRAINTS

    def test_get_anthropic_constraints_case_insensitive(self):
        """Test Anthropic framework name is case-insensitive."""
        constraints = FrameworkConstraints.get_constraints_for_framework("ANTHROPIC")
        assert constraints == FrameworkConstraints.ANTHROPIC_CONSTRAINTS

    def test_get_unknown_framework_returns_common(self):
        """Test unknown framework returns common constraints."""
        constraints = FrameworkConstraints.get_constraints_for_framework("unknown")

        assert constraints == FrameworkConstraints.COMMON_CONSTRAINTS
        assert "temperature" in constraints

    def test_get_empty_framework_name(self):
        """Test empty framework name returns common constraints."""
        constraints = FrameworkConstraints.get_constraints_for_framework("")
        assert constraints == FrameworkConstraints.COMMON_CONSTRAINTS


class TestValidateParameter:
    """Test validate_parameter method."""

    def test_validate_valid_temperature(self):
        """Test validating valid temperature value."""
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", 0.7)
        assert issues == []

    def test_validate_temperature_below_min(self):
        """Test temperature below minimum."""
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", -0.1)

        assert len(issues) == 1
        assert "below minimum" in issues[0]

    def test_validate_temperature_above_max(self):
        """Test temperature above maximum."""
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", 2.5)

        assert len(issues) == 1
        assert "above maximum" in issues[0]

    def test_validate_wrong_type(self):
        """Test parameter with wrong type."""
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", "hot")

        assert len(issues) == 1
        assert "expected type float" in issues[0]
        assert "got str" in issues[0]

    def test_validate_model_allowed_value(self):
        """Test model with allowed value."""
        issues = FrameworkConstraints.validate_parameter("openai", "model", "gpt-4o")
        assert issues == []

    def test_validate_model_disallowed_value(self):
        """Test model validation.

        Note: Model validation is now done via the model discovery service,
        not hardcoded allowed_values. Since allowed_values is None,
        any string model value passes basic FrameworkConstraints validation.
        The actual model validation happens at the plugin level via custom_validator.
        """
        issues = FrameworkConstraints.validate_parameter(
            "openai", "model", "invalid-model"
        )

        # With dynamic model validation, FrameworkConstraints doesn't validate model names
        # (that's done by the plugin's model discovery service)
        assert len(issues) == 0

    def test_validate_parameter_alias(self):
        """Test validating parameter by alias."""
        issues = FrameworkConstraints.validate_parameter("openai", "max_length", 100)
        assert issues == []

    def test_validate_max_tokens_alias(self):
        """Test max_tokens alias validation."""
        issues = FrameworkConstraints.validate_parameter(
            "openai", "max_new_tokens", 500
        )
        assert issues == []

    def test_validate_unknown_parameter(self):
        """Test validating unknown parameter returns no issues."""
        issues = FrameworkConstraints.validate_parameter(
            "openai", "unknown_param", "value"
        )
        assert issues == []

    def test_validate_top_p_in_range(self):
        """Test top_p within valid range."""
        issues = FrameworkConstraints.validate_parameter("openai", "top_p", 0.9)
        assert issues == []

    def test_validate_top_p_out_of_range(self):
        """Test top_p out of valid range."""
        issues = FrameworkConstraints.validate_parameter("openai", "top_p", 1.5)

        assert len(issues) == 1
        assert "above maximum" in issues[0]

    def test_validate_max_tokens_below_min(self):
        """Test max_tokens below minimum."""
        issues = FrameworkConstraints.validate_parameter("openai", "max_tokens", 0)

        assert len(issues) == 1
        assert "below minimum" in issues[0]

    def test_validate_anthropic_max_tokens_above_limit(self):
        """Test Anthropic max_tokens above limit.

        Note: Anthropic's current API supports up to 200000 tokens for Claude 3.5 models.
        """
        issues = FrameworkConstraints.validate_parameter(
            "anthropic", "max_tokens", 250000  # Above the 200000 limit
        )

        assert len(issues) == 1
        assert "above maximum" in issues[0]

    def test_validate_anthropic_max_tokens_within_limit(self):
        """Test Anthropic max_tokens within limit."""
        issues = FrameworkConstraints.validate_parameter(
            "anthropic", "max_tokens", 4000
        )
        assert issues == []

    def test_validate_frequency_penalty_range(self):
        """Test frequency_penalty validation."""
        # Valid
        assert (
            FrameworkConstraints.validate_parameter("openai", "frequency_penalty", 0.5)
            == []
        )
        assert (
            FrameworkConstraints.validate_parameter("openai", "frequency_penalty", -1.0)
            == []
        )

        # Invalid
        issues = FrameworkConstraints.validate_parameter(
            "openai", "frequency_penalty", -3.0
        )
        assert len(issues) == 1
        assert "below minimum" in issues[0]

    def test_validate_edge_case_boundary_values(self):
        """Test validation at exact boundary values."""
        # Exact min
        assert (
            FrameworkConstraints.validate_parameter("openai", "temperature", 0.0) == []
        )
        # Exact max
        assert (
            FrameworkConstraints.validate_parameter("openai", "temperature", 2.0) == []
        )
        # Just below min
        issues = FrameworkConstraints.validate_parameter(
            "openai", "temperature", -0.0001
        )
        assert len(issues) == 1


class TestConfigureIntegrations:
    """Test configure_integrations function."""

    def setup_method(self):
        """Reset global config before each test."""
        config_module.integration_config = IntegrationConfig()

    def test_configure_single_option(self):
        """Test configuring a single option."""
        configure_integrations(auto_discover=False)
        assert integration_config.auto_discover is False

    def test_configure_multiple_options(self):
        """Test configuring multiple options."""
        configure_integrations(
            auto_discover=False,
            strict_mode=True,
            fuzzy_matching_threshold=0.9,
        )

        assert integration_config.auto_discover is False
        assert integration_config.strict_mode is True
        assert integration_config.fuzzy_matching_threshold == 0.9

    def test_configure_boolean_options(self):
        """Test configuring boolean options."""
        configure_integrations(
            validate_types=False,
            sanitize_parameters=False,
            log_override_details=True,
        )

        assert integration_config.validate_types is False
        assert integration_config.sanitize_parameters is False
        assert integration_config.log_override_details is True

    def test_configure_numeric_options(self):
        """Test configuring numeric options."""
        configure_integrations(
            discovery_cache_ttl=7200,
            max_fallback_attempts=10,
        )

        assert integration_config.discovery_cache_ttl == 7200
        assert integration_config.max_fallback_attempts == 10

    def test_configure_unknown_option_warns(self):
        """Test configuring unknown option logs warning."""
        # Should not raise, just warn
        configure_integrations(unknown_option="value")
        # Config unchanged
        assert not hasattr(integration_config, "unknown_option")

    def test_configure_invalid_boolean_type(self):
        """Test configuring boolean with invalid type raises error."""
        with pytest.raises(ValidationError):
            configure_integrations(auto_discover="not_a_bool")

    def test_configure_invalid_threshold_range(self):
        """Test fuzzy_matching_threshold out of range raises error."""
        with pytest.raises(ValidationError):
            configure_integrations(fuzzy_matching_threshold=1.5)

    def test_configure_negative_threshold(self):
        """Test negative fuzzy_matching_threshold raises error."""
        with pytest.raises(ValidationError):
            configure_integrations(fuzzy_matching_threshold=-0.1)

    def test_configure_invalid_cache_ttl(self):
        """Test invalid cache_ttl raises error."""
        with pytest.raises(ValidationError):
            configure_integrations(discovery_cache_ttl=-100)

    def test_configure_zero_cache_ttl_invalid(self):
        """Test zero cache_ttl is invalid."""
        with pytest.raises(ValidationError):
            configure_integrations(discovery_cache_ttl=0)

    def test_configure_invalid_max_fallback_attempts(self):
        """Test invalid max_fallback_attempts raises error."""
        with pytest.raises(ValidationError):
            configure_integrations(max_fallback_attempts=-5)

    def test_configure_preserves_unmodified_options(self):
        """Test that unmodified options remain unchanged."""
        original_ttl = integration_config.discovery_cache_ttl
        original_strict = integration_config.strict_mode

        configure_integrations(auto_discover=False)

        assert integration_config.auto_discover is False
        assert integration_config.discovery_cache_ttl == original_ttl
        assert integration_config.strict_mode == original_strict

    def test_configure_valid_threshold_boundary(self):
        """Test threshold at exact boundaries."""
        configure_integrations(fuzzy_matching_threshold=0.0)
        assert integration_config.fuzzy_matching_threshold == 0.0

        configure_integrations(fuzzy_matching_threshold=1.0)
        assert integration_config.fuzzy_matching_threshold == 1.0

    def test_configure_all_options(self):
        """Test configuring all available options."""
        configure_integrations(
            auto_discover=False,
            discovery_cache_ttl=7200,
            cache_discovered_classes=False,
            strict_mode=True,
            fuzzy_matching_enabled=False,
            fuzzy_matching_threshold=0.9,
            validate_types=False,
            validate_values=False,
            auto_convert_types=False,
            version_check=False,
            warn_on_deprecated=False,
            auto_migrate_parameters=False,
            max_fallback_attempts=10,
            log_override_details=True,
            allow_unknown_parameters=True,
            sanitize_parameters=False,
        )

        # Verify all were set
        assert integration_config.auto_discover is False
        assert integration_config.discovery_cache_ttl == 7200
        assert integration_config.strict_mode is True
        assert integration_config.max_fallback_attempts == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_parameter_with_no_constraints(self):
        """Test parameter that has no constraints defined."""
        issues = FrameworkConstraints.validate_parameter(
            "openai", "custom_param", "value"
        )
        assert issues == []

    def test_validate_with_none_value(self):
        """Test validating None value."""
        # Should fail type check
        issues = FrameworkConstraints.validate_parameter("openai", "temperature", None)
        assert len(issues) > 0

    def test_multiple_validation_issues(self):
        """Test parameter can have multiple validation issues."""
        # Wrong type prevents range check, so only one issue
        issues = FrameworkConstraints.validate_parameter(
            "openai", "temperature", "invalid"
        )
        assert len(issues) == 1

    def test_empty_framework_name_validation(self):
        """Test validation with empty framework name."""
        issues = FrameworkConstraints.validate_parameter("", "temperature", 0.7)
        assert issues == []

    def test_config_with_extreme_values(self):
        """Test configuration with extreme but valid values."""
        config = IntegrationConfig(
            discovery_cache_ttl=999999,
            max_fallback_attempts=1000,
            fuzzy_matching_threshold=0.999,
        )

        assert config.discovery_cache_ttl == 999999
        assert config.max_fallback_attempts == 1000
        assert config.fuzzy_matching_threshold == 0.999
