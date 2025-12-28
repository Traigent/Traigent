"""Unit tests for agent configuration mapper."""

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


def _validate_temp_range(value):
    """Test validation function for temperature range."""
    if not 0 <= value <= 1:
        raise ValueError("Invalid temperature")


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification."""
    return AgentSpecification(
        id="test-agent",
        name="Test Agent",
        agent_type="conversational",
        agent_platform="openai",
        prompt_template="Answer this question: {question}",
        model_parameters={"model": "o4-mini", "temperature": 0.7},
        persona="helpful assistant",
        guidelines=["Be concise", "Be accurate"],
    )


@pytest.fixture
def sample_config():
    """Create a sample Traigent configuration."""
    return {
        "model": "GPT-4o",
        "temperature": 0.9,
        "max_tokens": 500,
        "top_p": 0.8,
        "frequency_penalty": 0.1,
    }


@pytest.fixture
def custom_platform_mapping():
    """Create a custom platform mapping for testing."""
    return PlatformMapping(
        platform="custom",
        parameter_mappings=[
            ParameterMapping(
                source_key="model_name",
                target_key="model",
                description="Custom model parameter",
            ),
            ParameterMapping(
                source_key="temp",
                target_key="temperature",
                transform=lambda x: float(x),
                validation=_validate_temp_range,
                description="Temperature with transform",
            ),
            ParameterMapping(
                source_key="system_prompt",
                target_key="persona",
                target_section="persona",
                description="System prompt mapping",
            ),
        ],
        template_mappings={
            "context": "context_data",
            "instructions": "task_instructions",
        },
    )


class TestParameterMapping:
    """Test ParameterMapping dataclass."""

    def test_parameter_mapping_creation(self):
        """Test creating parameter mapping."""
        mapping = ParameterMapping(
            source_key="temp",
            target_key="temperature",
            target_section="model_parameters",
            description="Temperature parameter",
        )

        assert mapping.source_key == "temp"
        assert mapping.target_key == "temperature"
        assert mapping.target_section == "model_parameters"
        assert mapping.transform is None
        assert mapping.default_value is None
        assert mapping.validation is None

    def test_parameter_mapping_with_transform(self):
        """Test parameter mapping with transformation."""

        def transform_func(x):
            return str(x).upper()

        mapping = ParameterMapping(
            source_key="model", target_key="model_name", transform=transform_func
        )

        assert mapping.transform == transform_func
        assert mapping.transform("GPT-4o") == "GPT-4O"


class TestPlatformMapping:
    """Test PlatformMapping dataclass."""

    def test_platform_mapping_creation(self):
        """Test creating platform mapping."""
        param_mapping = ParameterMapping("source", "target")
        mapping = PlatformMapping(
            platform="test",
            parameter_mappings=[param_mapping],
            template_mappings={"var": "config_key"},
        )

        assert mapping.platform == "test"
        assert len(mapping.parameter_mappings) == 1
        assert mapping.parameter_mappings[0] == param_mapping
        assert mapping.template_mappings["var"] == "config_key"
        assert len(mapping.custom_transformers) == 0
        assert len(mapping.validation_rules) == 0


class TestConfigurationMapper:
    """Test ConfigurationMapper class."""

    def test_initialization(self):
        """Test mapper initialization."""
        mapper = ConfigurationMapper()

        # Should have default mappings registered
        platforms = mapper.get_supported_platforms()
        assert "langchain" in platforms
        assert "openai" in platforms

    def test_register_platform_mapping(self, custom_platform_mapping):
        """Test registering custom platform mapping."""
        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(custom_platform_mapping)

        platforms = mapper.get_supported_platforms()
        assert "custom" in platforms

        retrieved = mapper.get_platform_mapping("custom")
        assert retrieved == custom_platform_mapping

    def test_apply_configuration_openai(self, sample_agent_spec, sample_config):
        """Test applying configuration to OpenAI agent."""
        mapper = ConfigurationMapper()

        updated_spec = mapper.apply_configuration(
            sample_agent_spec, sample_config, preserve_original=True
        )

        # Original should be unchanged
        assert sample_agent_spec.model_parameters["model"] == "o4-mini"
        assert sample_agent_spec.model_parameters["temperature"] == 0.7

        # Updated should have new values
        assert updated_spec.model_parameters["model"] == "GPT-4o"
        assert updated_spec.model_parameters["temperature"] == 0.9
        assert updated_spec.model_parameters["max_tokens"] == 500
        assert updated_spec.model_parameters["top_p"] == 0.8
        assert updated_spec.model_parameters["frequency_penalty"] == 0.1

    def test_apply_configuration_preserve_false(self, sample_agent_spec, sample_config):
        """Test applying configuration without preserving original."""
        mapper = ConfigurationMapper()

        updated_spec = mapper.apply_configuration(
            sample_agent_spec, sample_config, preserve_original=False
        )

        # Should be the same object
        assert updated_spec is sample_agent_spec
        assert sample_agent_spec.model_parameters["model"] == "GPT-4o"

    def test_apply_configuration_unknown_platform(self, sample_config):
        """Test applying configuration to unknown platform."""
        mapper = ConfigurationMapper()

        unknown_spec = AgentSpecification(
            id="unknown",
            name="Unknown",
            agent_type="task",
            agent_platform="unknown_platform",
            prompt_template="Test",
            model_parameters={},
        )

        # Should return unchanged spec with warning log
        updated_spec = mapper.apply_configuration(unknown_spec, sample_config)
        assert updated_spec.model_parameters == {}

    def test_apply_configuration_with_transforms(self, custom_platform_mapping):
        """Test applying configuration with transforms."""
        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(custom_platform_mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="custom",
            prompt_template="Test",
            model_parameters={},
        )

        config = {
            "model_name": "GPT-4o",
            "temp": "0.8",  # String that should be converted to float
            "system_prompt": "You are helpful",
        }

        updated_spec = mapper.apply_configuration(spec, config)

        assert updated_spec.model_parameters["model"] == "GPT-4o"
        assert updated_spec.model_parameters["temperature"] == 0.8
        assert isinstance(updated_spec.model_parameters["temperature"], float)
        assert updated_spec.persona == "You are helpful"

    def test_apply_configuration_validation_error(self, custom_platform_mapping):
        """Test configuration application with validation error."""
        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(custom_platform_mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="custom",
            prompt_template="Test",
            model_parameters={},
        )

        # Invalid temperature (> 1)
        config = {"temp": "1.5"}

        with pytest.raises(ConfigurationError, match="Validation failed"):
            mapper.apply_configuration(spec, config)

    def test_apply_template_mappings(self):
        """Test applying template mappings."""
        mapper = ConfigurationMapper()

        # Create mapping with template variables
        mapping = PlatformMapping(
            platform="template_test",
            template_mappings={"context": "context_data", "task": "task_type"},
        )
        mapper.register_platform_mapping(mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="template_test",
            prompt_template="Context: {context}\nTask: {task}\nQuestion: {question}",
            model_parameters={},
        )

        config = {"context_data": "AI research", "task_type": "analysis"}

        updated_spec = mapper.apply_configuration(spec, config)

        expected_template = "Context: AI research\nTask: analysis\nQuestion: {question}"
        assert updated_spec.prompt_template == expected_template

    def test_validate_configuration_compatibility(self, sample_agent_spec):
        """Test configuration compatibility validation."""
        mapper = ConfigurationMapper()

        # Compatible configuration
        config_space = {
            "model": ["o4-mini", "GPT-4o"],
            "temperature": (0.0, 1.0),
            "max_tokens": [100, 500, 1000],
        }

        result = mapper.validate_configuration_compatibility(
            sample_agent_spec, config_space
        )

        assert result["compatible"] is True
        assert len(result["errors"]) == 0
        assert "model" in result["supported_parameters"]
        assert "temperature" in result["supported_parameters"]
        assert "max_tokens" in result["supported_parameters"]

    def test_validate_configuration_compatibility_unknown_platform(self):
        """Test compatibility validation for unknown platform."""
        mapper = ConfigurationMapper()

        unknown_spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="unknown",
            prompt_template="Test",
            model_parameters={},
        )

        result = mapper.validate_configuration_compatibility(
            unknown_spec, {"model": "GPT-4o"}
        )

        assert result["compatible"] is False
        assert "No mapping available" in result["errors"][0]

    def test_validate_configuration_compatibility_warnings(self, sample_agent_spec):
        """Test compatibility validation with warnings."""
        mapper = ConfigurationMapper()

        config_space = {
            "model": ["GPT-4o"],
            "unknown_param": ["value1", "value2"],  # Should generate warning
        }

        result = mapper.validate_configuration_compatibility(
            sample_agent_spec, config_space
        )

        assert result["compatible"] is True  # No errors, just warnings
        assert len(result["warnings"]) == 1
        assert "unknown_param" in result["warnings"][0]

    def test_validation_functions(self):
        """Test built-in validation functions."""
        mapper = ConfigurationMapper()

        # Temperature validation
        mapper._validate_temperature(0.5)  # Should not raise
        mapper._validate_temperature(0.0)  # Should not raise
        mapper._validate_temperature(2.0)  # Should not raise

        with pytest.raises(ValueError):
            mapper._validate_temperature(-0.1)

        with pytest.raises(ValueError):
            mapper._validate_temperature(2.1)

        # Probability validation
        mapper._validate_probability(0.5)  # Should not raise

        with pytest.raises(ValueError):
            mapper._validate_probability(-0.1)

        with pytest.raises(ValueError):
            mapper._validate_probability(1.1)

        # Penalty validation
        mapper._validate_penalty(0.0)  # Should not raise
        mapper._validate_penalty(-2.0)  # Should not raise
        mapper._validate_penalty(2.0)  # Should not raise

        with pytest.raises(ValueError):
            mapper._validate_penalty(-2.1)

        with pytest.raises(ValueError):
            mapper._validate_penalty(2.1)

        # Positive integer validation
        mapper._validate_positive_int(100)  # Should not raise

        with pytest.raises(ValueError):
            mapper._validate_positive_int(0)

        with pytest.raises(ValueError):
            mapper._validate_positive_int(-5)


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_apply_config_to_agent(self, sample_agent_spec, sample_config):
        """Test apply_config_to_agent function."""
        updated_spec = apply_config_to_agent(sample_agent_spec, sample_config)

        assert updated_spec.model_parameters["model"] == "GPT-4o"
        assert updated_spec.model_parameters["temperature"] == 0.9

        # Original should be unchanged (preserve_original=True by default)
        assert sample_agent_spec.model_parameters["model"] == "o4-mini"

    def test_validate_config_compatibility(self, sample_agent_spec):
        """Test validate_config_compatibility function."""
        config_space = {"model": ["GPT-4o"], "temperature": (0.0, 1.0)}

        result = validate_config_compatibility(sample_agent_spec, config_space)

        assert result["compatible"] is True
        assert len(result["supported_parameters"]) >= 2

    def test_register_platform_mapping_function(self, custom_platform_mapping):
        """Test register_platform_mapping function."""
        # Register the mapping
        register_platform_mapping(custom_platform_mapping)

        # Should be available in supported platforms
        platforms = get_supported_platforms()
        assert "custom" in platforms

    def test_get_supported_platforms(self):
        """Test get_supported_platforms function."""
        platforms = get_supported_platforms()

        assert isinstance(platforms, list)
        assert "langchain" in platforms
        assert "openai" in platforms


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_configuration_error_on_transform_failure(self):
        """Test handling of transform failures."""

        # Create mapping with failing transform
        def failing_transform(x):
            raise ValueError("Transform failed")

        mapping = PlatformMapping(
            platform="fail_test",
            parameter_mappings=[
                ParameterMapping(
                    source_key="test_param",
                    target_key="test_target",
                    transform=failing_transform,
                )
            ],
        )

        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="fail_test",
            prompt_template="Test",
            model_parameters={},
        )

        config = {"test_param": "value"}

        with pytest.raises(
            ConfigurationError,
            match="Transform 'failing_transform' for parameter 'test_param'",
        ):
            mapper.apply_configuration(spec, config)

    def test_configuration_error_on_validation_failure(self):
        """Test configuration error on validation failure."""

        def failing_validation(x):
            raise ValueError("Validation failed")

        mapping = PlatformMapping(
            platform="validation_fail",
            parameter_mappings=[
                ParameterMapping(
                    source_key="test_param",
                    target_key="test_target",
                    validation=failing_validation,
                )
            ],
        )

        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="validation_fail",
            prompt_template="Test",
            model_parameters={},
        )

        config = {"test_param": "value"}

        with pytest.raises(ConfigurationError, match="Validation failed"):
            mapper.apply_configuration(spec, config)


class TestCustomTransformers:
    """Test custom transformation functionality."""

    def test_custom_transformers(self):
        """Test applying custom transformers."""

        def custom_transformer(spec, config):
            # Add a custom field based on config
            if "custom_field" in config:
                spec.model_parameters["transformed_field"] = (
                    f"custom_{config['custom_field']}"
                )
            return spec

        mapping = PlatformMapping(
            platform="custom_transform",
            custom_transformers={"custom": custom_transformer},
        )

        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="custom_transform",
            prompt_template="Test",
            model_parameters={},
        )

        config = {"custom_field": "value"}

        updated_spec = mapper.apply_configuration(spec, config)
        assert updated_spec.model_parameters["transformed_field"] == "custom_value"

    def test_custom_transformer_failure_handling(self):
        """Test handling of custom transformer failures."""

        def failing_transformer(spec, config):
            raise RuntimeError("Transformer failed")

        mapping = PlatformMapping(
            platform="failing_transform",
            custom_transformers={"failing": failing_transformer},
        )

        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="failing_transform",
            prompt_template="Test",
            model_parameters={},
        )

        with pytest.raises(
            ConfigurationError, match="Custom transformer 'failing' failed"
        ):
            mapper.apply_configuration(spec, {})

    def test_transform_failure_raises_configuration_error(self):
        """Transform errors should surface as configuration errors."""

        def failing_transform(value):
            raise RuntimeError("boom")

        mapping = PlatformMapping(
            platform="transform_fail",
            parameter_mappings=[
                ParameterMapping(
                    source_key="temp",
                    target_key="temperature",
                    transform=failing_transform,
                )
            ],
        )

        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="transform_fail",
            prompt_template="Test",
            model_parameters={},
        )

        with pytest.raises(
            ConfigurationError, match="Transform 'failing_transform' raised"
        ):
            mapper.apply_configuration(spec, {"temp": "value"})


class TestValidationRules:
    """Test validation rule functionality."""

    def test_validation_rules(self):
        """Test applying validation rules."""

        def require_model_param(spec):
            if not spec.model_parameters.get("model"):
                raise ValueError("Model parameter is required")

        mapping = PlatformMapping(
            platform="validation_test", validation_rules=[require_model_param]
        )

        mapper = ConfigurationMapper()
        mapper.register_platform_mapping(mapping)

        spec = AgentSpecification(
            id="test",
            name="Test",
            agent_type="task",
            agent_platform="validation_test",
            prompt_template="Test",
            model_parameters={},
        )

        config = {"other_param": "value"}

        with pytest.raises(ConfigurationError, match="Model parameter is required"):
            mapper.apply_configuration(spec, config)

        # Should work with model parameter
        config_with_model = {"model": "GPT-4o"}

        # Need to add parameter mapping for this to work
        mapping.parameter_mappings.append(
            ParameterMapping(source_key="model", target_key="model")
        )

        updated_spec = mapper.apply_configuration(spec, config_with_model)
        assert updated_spec.model_parameters["model"] == "GPT-4o"
