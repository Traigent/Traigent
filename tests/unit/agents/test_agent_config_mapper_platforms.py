"""Tests for configuration mapper functionality with new platforms (Anthropic, Cohere, HuggingFace).

This module tests the configuration mapping system for the newly added platforms,
ensuring proper parameter mapping, validation, and compatibility checking.
"""

import pytest

from traigent.agents.config_mapper import (
    ConfigurationMapper,
    ParameterMapping,
    PlatformMapping,
    apply_config_to_agent,
    get_supported_platforms,
    register_platform_mapping,
    validate_config_compatibility,
)
from traigent.cloud.models import AgentSpecification
from traigent.utils.exceptions import ConfigurationError


class TestConfigMapperPlatforms:
    """Test configuration mapper functionality with new platforms."""

    @pytest.fixture
    def config_mapper(self):
        """Create a fresh configuration mapper."""
        return ConfigurationMapper()

    @pytest.fixture
    def anthropic_agent_spec(self):
        """Create an Anthropic agent specification."""
        return AgentSpecification(
            id="test-anthropic-agent",
            name="Test Anthropic Agent",
            agent_platform="anthropic",
            model_parameters={
                "model": "claude-3-opus-20240229",
                "temperature": 1.0,
                "max_tokens_to_sample": 1000,
                "top_p": 1.0,
                "top_k": 100,
                "stop_sequences": [],
            },
        )

    @pytest.fixture
    def cohere_agent_spec(self):
        """Create a Cohere agent specification."""
        return AgentSpecification(
            id="test-cohere-agent",
            name="Test Cohere Agent",
            agent_platform="cohere",
            model_parameters={
                "model": "command-r",
                "temperature": 0.5,
                "max_tokens": 500,
                "p": 0.9,
                "k": 50,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
        )

    @pytest.fixture
    def huggingface_agent_spec(self):
        """Create a HuggingFace agent specification."""
        return AgentSpecification(
            id="test-hf-agent",
            name="Test HuggingFace Agent",
            agent_platform="huggingface",
            model_parameters={
                "model_id": "gpt2",
                "temperature": 0.8,
                "max_new_tokens": 100,
                "top_p": 0.95,
                "top_k": 40,
                "stop": None,
            },
        )

    def test_anthropic_platform_mapping_exists(self, config_mapper):
        """Test that Anthropic platform mapping is registered by default."""
        platforms = config_mapper.get_supported_platforms()
        assert "anthropic" in platforms

        mapping = config_mapper.get_platform_mapping("anthropic")
        assert mapping is not None
        assert mapping.platform == "anthropic"

    def test_cohere_platform_mapping_exists(self, config_mapper):
        """Test that Cohere platform mapping is registered by default."""
        platforms = config_mapper.get_supported_platforms()
        assert "cohere" in platforms

        mapping = config_mapper.get_platform_mapping("cohere")
        assert mapping is not None
        assert mapping.platform == "cohere"

    def test_huggingface_platform_mapping_exists(self, config_mapper):
        """Test that HuggingFace platform mapping is registered by default."""
        platforms = config_mapper.get_supported_platforms()
        assert "huggingface" in platforms

        mapping = config_mapper.get_platform_mapping("huggingface")
        assert mapping is not None
        assert mapping.platform == "huggingface"

    def test_anthropic_parameter_mapping(self, config_mapper, anthropic_agent_spec):
        """Test Anthropic parameter mapping functionality."""
        config = {
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.95,
            "top_k": 50,
            "stop_sequences": ["END", "STOP"],
        }

        updated_spec = config_mapper.apply_configuration(anthropic_agent_spec, config)

        # Verify parameters were mapped correctly
        assert updated_spec.model_parameters["model"] == "claude-3-sonnet-20240229"
        assert updated_spec.model_parameters["temperature"] == 0.7
        assert (
            updated_spec.model_parameters["max_tokens_to_sample"] == 2000
        )  # Note the mapping
        assert updated_spec.model_parameters["top_p"] == 0.95
        assert updated_spec.model_parameters["top_k"] == 50
        assert updated_spec.model_parameters["stop_sequences"] == ["END", "STOP"]

    def test_cohere_parameter_mapping(self, config_mapper, cohere_agent_spec):
        """Test Cohere parameter mapping functionality."""
        config = {
            "model": "command-r-plus",
            "temperature": 0.3,
            "max_tokens": 1500,
            "top_p": 0.85,
            "top_k": 30,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.2,
            "seed": 12345,
        }

        updated_spec = config_mapper.apply_configuration(cohere_agent_spec, config)

        # Verify parameters were mapped correctly
        assert updated_spec.model_parameters["model"] == "command-r-plus"
        assert updated_spec.model_parameters["temperature"] == 0.3
        assert updated_spec.model_parameters["max_tokens"] == 1500
        assert updated_spec.model_parameters["p"] == 0.85  # Note: top_p -> p
        assert updated_spec.model_parameters["k"] == 30  # Note: top_k -> k
        assert updated_spec.model_parameters["frequency_penalty"] == 0.5
        assert updated_spec.model_parameters["presence_penalty"] == 0.2
        assert updated_spec.model_parameters["seed"] == 12345

    def test_huggingface_parameter_mapping(self, config_mapper, huggingface_agent_spec):
        """Test HuggingFace parameter mapping functionality."""
        config = {
            "model": "meta-llama/Llama-2-7b-hf",
            "temperature": 0.6,
            "max_tokens": 256,
            "top_p": 0.9,
            "top_k": 20,
            "stop_sequences": ["\\n", "END"],
            "seed": 42,
        }

        updated_spec = config_mapper.apply_configuration(huggingface_agent_spec, config)

        # Verify parameters were mapped correctly
        assert (
            updated_spec.model_parameters["model_id"] == "meta-llama/Llama-2-7b-hf"
        )  # Note: model -> model_id
        assert updated_spec.model_parameters["temperature"] == 0.6
        assert (
            updated_spec.model_parameters["max_new_tokens"] == 256
        )  # Note: max_tokens -> max_new_tokens
        assert updated_spec.model_parameters["top_p"] == 0.9
        assert updated_spec.model_parameters["top_k"] == 20
        assert updated_spec.model_parameters["stop"] == [
            "\\n",
            "END",
        ]  # Note: stop_sequences -> stop
        assert updated_spec.model_parameters["seed"] == 42

    def test_anthropic_parameter_validation(self, config_mapper, anthropic_agent_spec):
        """Test Anthropic parameter validation."""
        # Test invalid temperature
        config = {"temperature": 3.0}  # Out of range
        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(anthropic_agent_spec, config)
        assert "Temperature must be between 0 and 2" in str(exc_info.value)

        # Test invalid top_p
        config = {"top_p": 1.5}  # Out of range
        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(anthropic_agent_spec, config)
        assert "Probability must be between 0 and 1" in str(exc_info.value)

        # Test invalid top_k
        config = {"top_k": -5}  # Negative value
        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(anthropic_agent_spec, config)
        assert "Value must be a positive integer" in str(exc_info.value)

    def test_cohere_parameter_validation(self, config_mapper, cohere_agent_spec):
        """Test Cohere parameter validation."""
        # Test invalid frequency_penalty
        config = {"frequency_penalty": 3.0}  # Out of range
        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(cohere_agent_spec, config)
        assert "Penalty must be between -2 and 2" in str(exc_info.value)

        # Test invalid presence_penalty
        config = {"presence_penalty": -3.0}  # Out of range
        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(cohere_agent_spec, config)
        assert "Penalty must be between -2 and 2" in str(exc_info.value)

        # Test invalid seed
        config = {"seed": -100}  # Negative value
        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(cohere_agent_spec, config)
        assert "Value must be a positive integer" in str(exc_info.value)

    def test_huggingface_parameter_validation(
        self, config_mapper, huggingface_agent_spec
    ):
        """Test HuggingFace parameter validation."""
        # Test invalid max_tokens
        config = {"max_tokens": 0}  # Zero value
        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(huggingface_agent_spec, config)
        assert "Value must be a positive integer" in str(exc_info.value)

        # Test invalid temperature
        config = {"temperature": -0.5}  # Negative value
        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(huggingface_agent_spec, config)
        assert "Temperature must be between 0 and 2" in str(exc_info.value)

    def test_validate_configuration_compatibility_anthropic(
        self, config_mapper, anthropic_agent_spec
    ):
        """Test configuration compatibility validation for Anthropic."""
        config_space = {
            "model": ["claude-3-opus", "claude-3-sonnet"],
            "temperature": [0.0, 0.5, 1.0],
            "max_tokens": [500, 1000, 2000],
            "top_p": [0.9, 0.95, 1.0],
            "top_k": [10, 50, 100],
            "unknown_param": [1, 2, 3],  # Unknown parameter
        }

        result = config_mapper.validate_configuration_compatibility(
            anthropic_agent_spec, config_space
        )

        assert result["compatible"] is True
        assert len(result["errors"]) == 0
        assert "model" in result["supported_parameters"]
        assert "temperature" in result["supported_parameters"]
        assert "max_tokens" in result["supported_parameters"]
        assert "top_p" in result["supported_parameters"]
        assert "top_k" in result["supported_parameters"]
        assert "unknown_param" not in result["supported_parameters"]
        assert any("unknown_param" in warning for warning in result["warnings"])

    def test_validate_configuration_compatibility_cohere(
        self, config_mapper, cohere_agent_spec
    ):
        """Test configuration compatibility validation for Cohere."""
        config_space = {
            "model": ["command", "command-r"],
            "temperature": [0.0, 0.5, 1.0],
            "max_tokens": [100, 500, 1000],
            "top_p": [0.8, 0.9, 1.0],
            "top_k": [10, 50],
            "frequency_penalty": [-1.0, 0.0, 1.0],
            "presence_penalty": [-1.0, 0.0, 1.0],
            "seed": [42, 123],
        }

        result = config_mapper.validate_configuration_compatibility(
            cohere_agent_spec, config_space
        )

        assert result["compatible"] is True
        assert len(result["errors"]) == 0
        assert all(
            param in result["supported_parameters"] for param in config_space.keys()
        )

    def test_validate_configuration_compatibility_huggingface(
        self, config_mapper, huggingface_agent_spec
    ):
        """Test configuration compatibility validation for HuggingFace."""
        config_space = {
            "model": ["gpt2", "gpt2-medium", "gpt2-large"],
            "temperature": [0.5, 0.8, 1.0],
            "max_tokens": [50, 100, 200],
            "top_p": [0.9, 0.95],
            "top_k": [20, 40, 50],
            "stop_sequences": [["\\n"], ["\\n\\n"], ["END"]],
            "seed": [42],
        }

        result = config_mapper.validate_configuration_compatibility(
            huggingface_agent_spec, config_space
        )

        assert result["compatible"] is True
        assert len(result["errors"]) == 0
        assert all(
            param in result["supported_parameters"] for param in config_space.keys()
        )

    def test_parameter_transformation_functions(self, config_mapper):
        """Test parameter transformation functions for different platforms."""
        # Test OpenAI stop_sequences transformation
        openai_mapping = config_mapper.get_platform_mapping("openai")
        stop_mapping = next(
            m
            for m in openai_mapping.parameter_mappings
            if m.source_key == "stop_sequences"
        )

        # Test list input (should remain as list)
        assert stop_mapping.transform(["END", "STOP"]) == ["END", "STOP"]

        # Test single string input (should be converted to list)
        assert stop_mapping.transform("END") == ["END"]

    def test_platform_specific_validation_rules(self, config_mapper):
        """Test platform-specific validation rules."""
        # Create specs with invalid configurations
        invalid_anthropic_spec = AgentSpecification(
            id="test", name="Test", agent_platform="anthropic", model_parameters={}
        )

        # Apply a configuration that should pass individual parameter validation
        # but might fail platform-specific rules
        config = {"temperature": 0.5, "max_tokens": 1000}

        # Should succeed as these are valid parameters
        result = config_mapper.apply_configuration(invalid_anthropic_spec, config)
        assert result.model_parameters["temperature"] == 0.5
        assert result.model_parameters["max_tokens_to_sample"] == 1000

    def test_custom_platform_registration(self, config_mapper):
        """Test registering a custom platform mapping."""
        # Create custom platform mapping
        custom_mapping = PlatformMapping(
            platform="custom_llm",
            parameter_mappings=[
                ParameterMapping(
                    source_key="model",
                    target_key="model_identifier",
                    description="Custom LLM model identifier",
                ),
                ParameterMapping(
                    source_key="temperature",
                    target_key="sampling_temp",
                    validation=lambda x: (
                        x if 0 <= x <= 1 else ValueError("Must be 0-1")
                    ),
                    description="Sampling temperature (0-1)",
                ),
                ParameterMapping(
                    source_key="max_tokens",
                    target_key="output_length",
                    transform=lambda x: min(x, 512),  # Cap at 512
                    description="Maximum output length",
                ),
            ],
            validation_rules=[
                lambda spec: spec.model_parameters.get("model_identifier") is not None
                or ValueError("Model required")
            ],
        )

        # Register the mapping
        config_mapper.register_platform_mapping(custom_mapping)

        # Verify it was registered
        assert "custom_llm" in config_mapper.get_supported_platforms()
        assert config_mapper.get_platform_mapping("custom_llm") is not None

        # Test using the custom mapping
        custom_spec = AgentSpecification(
            id="custom-agent",
            name="Custom Agent",
            agent_platform="custom_llm",
            model_parameters={},
        )

        config = {
            "model": "custom-model-v1",
            "temperature": 0.8,
            "max_tokens": 1000,  # Should be capped at 512
        }

        updated_spec = config_mapper.apply_configuration(custom_spec, config)

        assert updated_spec.model_parameters["model_identifier"] == "custom-model-v1"
        assert updated_spec.model_parameters["sampling_temp"] == 0.8
        assert (
            updated_spec.model_parameters["output_length"] == 512
        )  # Capped by transform

    def test_missing_platform_mapping(self, config_mapper):
        """Test behavior when platform mapping is not found."""
        unknown_spec = AgentSpecification(
            id="unknown",
            name="Unknown",
            agent_platform="unknown_platform",
            model_parameters={},
        )

        config = {"temperature": 0.5}

        # Should return original spec unchanged
        result = config_mapper.apply_configuration(unknown_spec, config)
        assert result.model_parameters == {}

        # Validation should indicate incompatibility
        validation_result = config_mapper.validate_configuration_compatibility(
            unknown_spec, {"temperature": [0.5]}
        )
        assert validation_result["compatible"] is False
        assert (
            "No mapping available for platform: unknown_platform"
            in validation_result["errors"][0]
        )

    def test_partial_configuration_mapping(self, config_mapper, anthropic_agent_spec):
        """Test mapping with partial configuration (some parameters missing)."""
        # Only provide some parameters
        config = {
            "temperature": 0.7,
            "top_k": 25,
            # max_tokens, top_p, etc. are missing
        }

        updated_spec = config_mapper.apply_configuration(anthropic_agent_spec, config)

        # Verify only provided parameters were updated
        assert updated_spec.model_parameters["temperature"] == 0.7
        assert updated_spec.model_parameters["top_k"] == 25
        # Original values should be preserved
        assert updated_spec.model_parameters["model"] == "claude-3-opus-20240229"
        assert updated_spec.model_parameters["max_tokens_to_sample"] == 1000
        assert updated_spec.model_parameters["top_p"] == 1.0

    def test_preserve_original_spec(self, config_mapper, cohere_agent_spec):
        """Test that original spec is preserved when preserve_original=True."""
        config = {"temperature": 0.1}

        # Apply configuration with preserve_original=True (default)
        updated_spec = config_mapper.apply_configuration(
            cohere_agent_spec, config, preserve_original=True
        )

        # Verify original spec is unchanged
        assert cohere_agent_spec.model_parameters["temperature"] == 0.5
        # Updated spec should have new value
        assert updated_spec.model_parameters["temperature"] == 0.1

        # Apply configuration with preserve_original=False
        config_mapper.apply_configuration(
            cohere_agent_spec, config, preserve_original=False
        )
        # Original spec should now be modified
        assert cohere_agent_spec.model_parameters["temperature"] == 0.1

    def test_configuration_error_handling(self, config_mapper, huggingface_agent_spec):
        """Test proper error handling and error messages."""
        # Test with invalid configuration that causes exception
        config = {"temperature": "invalid_value"}  # Should be numeric

        with pytest.raises(ConfigurationError) as exc_info:
            config_mapper.apply_configuration(huggingface_agent_spec, config)

        error = exc_info.value
        assert "Failed to apply configuration for platform huggingface" in str(error)
        assert error.details["platform"] == "huggingface"
        assert error.details["config"] == config

    def test_default_value_handling(self, config_mapper):
        """Test handling of default values in parameter mappings."""
        # Create a custom mapping with default values
        custom_mapping = PlatformMapping(
            platform="test_defaults",
            parameter_mappings=[
                ParameterMapping(
                    source_key="temperature", target_key="temp", default_value=0.7
                ),
                ParameterMapping(
                    source_key="max_tokens", target_key="max_len", default_value=100
                ),
            ],
        )

        config_mapper.register_platform_mapping(custom_mapping)

        spec = AgentSpecification(
            id="test", name="Test", agent_platform="test_defaults", model_parameters={}
        )

        # Apply empty configuration - should use defaults
        updated_spec = config_mapper.apply_configuration(spec, {})

        # Default values should be applied when source key is missing
        assert updated_spec.model_parameters["temp"] == 0.7
        assert updated_spec.model_parameters["max_len"] == 100

        # Apply configuration with one parameter
        config = {"temperature": 0.5}
        updated_spec = config_mapper.apply_configuration(spec, config)

        # Specified value should be used, not default
        assert updated_spec.model_parameters["temp"] == 0.5
        # Other parameter uses default since key is not in config
        assert updated_spec.model_parameters["max_len"] == 100

    def test_global_functions(self):
        """Test global configuration mapper functions."""
        # Test apply_config_to_agent
        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_platform="anthropic",
            model_parameters={"temperature": 1.0},
        )
        config = {"temperature": 0.5}

        updated_spec = apply_config_to_agent(spec, config)
        assert updated_spec.model_parameters["temperature"] == 0.5

        # Test validate_config_compatibility
        validation_result = validate_config_compatibility(
            spec, {"temperature": [0.5, 1.0]}
        )
        assert validation_result["compatible"] is True

        # Test get_supported_platforms
        platforms = get_supported_platforms()
        assert "anthropic" in platforms
        assert "cohere" in platforms
        assert "huggingface" in platforms
        assert "openai" in platforms
        assert "langchain" in platforms

        # Test register_platform_mapping
        test_mapping = PlatformMapping(platform="test_global")
        register_platform_mapping(test_mapping)
        assert "test_global" in get_supported_platforms()
