"""Unit tests for traigent.core.config_builder.

Tests for configuration builder patterns and utilities for constructing and
validating configuration objects for the optimization system.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.config.types import ExecutionMode
from traigent.core.config_builder import (
    ConfigurationSpaceBuilder,
    OptimizedFunctionConfig,
    create_advanced_config_space,
    create_simple_config_space,
)
from traigent.core.constants import (
    DEFAULT_EXECUTION_MODE,
    DEFAULT_MODEL,
)
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.types import ParameterType


class TestOptimizedFunctionConfig:
    """Tests for OptimizedFunctionConfig class."""

    @pytest.fixture
    def sample_func(self) -> MagicMock:
        """Create a sample function for testing."""
        mock_func = MagicMock()
        mock_func.__name__ = "test_function"
        return mock_func

    @pytest.fixture
    def sample_objectives(self) -> ObjectiveSchema:
        """Create sample objectives for testing."""
        return ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition(
                    name="accuracy", orientation="maximize", weight=0.7
                ),
                ObjectiveDefinition(name="cost", orientation="minimize", weight=0.3),
            ]
        )

    @pytest.fixture
    def sample_config_space(self) -> dict[str, Any]:
        """Create sample configuration space for testing."""
        return {
            "temperature": (0.0, 1.0),
            "max_tokens": (100, 2000),
            "model": ["gpt-4o", "gpt-4o-mini"],
        }

    @pytest.fixture
    def sample_default_config(self) -> dict[str, Any]:
        """Create sample default configuration for testing."""
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "gpt-4o",
        }

    @pytest.fixture
    def basic_config(
        self,
        sample_func: MagicMock,
        sample_objectives: ObjectiveSchema,
        sample_config_space: dict[str, Any],
    ) -> OptimizedFunctionConfig:
        """Create a basic configuration instance for testing."""
        return OptimizedFunctionConfig(
            func=sample_func,
            objectives=sample_objectives,
            configuration_space=sample_config_space,
        )

    # Happy path tests
    def test_init_with_minimal_params(self) -> None:
        """Test initialization with minimal parameters."""
        config = OptimizedFunctionConfig()

        assert config.func is None
        assert config.eval_dataset is None
        assert config.configuration_space == {}
        assert config.default_config == {}
        assert config.constraints == []
        assert config.injection_mode == "context"
        assert config.config_param is None
        assert config.auto_override_frameworks is False
        assert config.framework_targets == []
        assert config.execution_mode == DEFAULT_EXECUTION_MODE
        assert config.local_storage_path is None
        assert config.minimal_logging is True
        assert config.custom_evaluator is None
        assert config.scoring_function is None
        assert config.metric_functions == {}

    def test_init_with_all_params(
        self,
        sample_func: MagicMock,
        sample_objectives: ObjectiveSchema,
        sample_config_space: dict[str, Any],
        sample_default_config: dict[str, Any],
    ) -> None:
        """Test initialization with all parameters."""
        eval_dataset = [{"input": "test", "output": "result"}]

        def constraint(x):
            return x["temperature"] > 0

        custom_evaluator = MagicMock()
        scoring_function = MagicMock()
        metric_functions = {"accuracy": MagicMock()}

        config = OptimizedFunctionConfig(
            func=sample_func,
            eval_dataset=eval_dataset,
            objectives=sample_objectives,
            configuration_space=sample_config_space,
            default_config=sample_default_config,
            constraints=[constraint],
            injection_mode="parameter",
            config_param="llm_config",
            auto_override_frameworks=True,
            framework_targets=["langchain"],
            execution_mode="cloud",
            local_storage_path="/tmp/traigent",
            minimal_logging=False,
            custom_evaluator=custom_evaluator,
            scoring_function=scoring_function,
            metric_functions=metric_functions,
            custom_key="custom_value",
        )

        assert config.func == sample_func
        assert config.eval_dataset == eval_dataset
        assert config.configuration_space == sample_config_space
        assert config.default_config == sample_default_config
        assert len(config.constraints) == 1
        assert config.injection_mode == "parameter"
        assert config.config_param == "llm_config"
        assert config.auto_override_frameworks is True
        assert config.framework_targets == ["langchain"]
        assert config.execution_mode == "cloud"
        assert config.local_storage_path == "/tmp/traigent"
        assert config.minimal_logging is False
        assert config.custom_evaluator == custom_evaluator
        assert config.scoring_function == scoring_function
        assert config.metric_functions == metric_functions
        assert config.extra_config["custom_key"] == "custom_value"

    def test_init_with_objectives_as_list(self) -> None:
        """Test initialization with objectives as a list of strings."""
        config = OptimizedFunctionConfig(objectives=["accuracy", "cost"])

        assert len(config.objective_schema.objectives) == 2
        assert config.objectives == ["accuracy", "cost"]

    def test_init_with_objectives_as_schema(
        self, sample_objectives: ObjectiveSchema
    ) -> None:
        """Test initialization with objectives as ObjectiveSchema."""
        config = OptimizedFunctionConfig(objectives=sample_objectives)

        assert config.objective_schema == sample_objectives
        assert len(config.objectives) == 2

    def test_init_with_none_objectives_creates_default(self) -> None:
        """Test initialization with None objectives creates default."""
        config = OptimizedFunctionConfig(objectives=None)

        assert len(config.objective_schema.objectives) >= 1
        assert "accuracy" in config.objectives

    # Class methods tests
    def test_from_legacy_params_with_config_space(self) -> None:
        """Test from_legacy_params with legacy 'config_space' parameter."""
        legacy_config = OptimizedFunctionConfig.from_legacy_params(
            config_space={"temperature": (0.0, 1.0)}
        )

        assert legacy_config.configuration_space == {"temperature": (0.0, 1.0)}

    def test_from_legacy_params_with_configuration_space(self) -> None:
        """Test from_legacy_params with 'configuration_space' parameter."""
        legacy_config = OptimizedFunctionConfig.from_legacy_params(
            configuration_space={"temperature": (0.0, 1.0)}
        )

        assert legacy_config.configuration_space == {"temperature": (0.0, 1.0)}

    def test_from_legacy_params_prefers_config_space(self) -> None:
        """Test from_legacy_params prefers 'config_space' over 'configuration_space'."""
        legacy_config = OptimizedFunctionConfig.from_legacy_params(
            config_space={"temperature": (0.0, 1.0)},
            configuration_space={"model": ["gpt-4o"]},
        )

        assert legacy_config.configuration_space == {"temperature": (0.0, 1.0)}

    def test_from_dict(
        self, sample_config_space: dict[str, Any], sample_objectives: ObjectiveSchema
    ) -> None:
        """Test creating configuration from dictionary."""
        config_dict = {
            "configuration_space": sample_config_space,
            "objectives": sample_objectives,
            "execution_mode": "cloud",
        }

        config = OptimizedFunctionConfig.from_dict(config_dict)

        assert config.configuration_space == sample_config_space
        assert config.objective_schema == sample_objectives
        assert config.execution_mode == "cloud"

    def test_to_dict(
        self,
        sample_func: MagicMock,
        sample_objectives: ObjectiveSchema,
        sample_config_space: dict[str, Any],
    ) -> None:
        """Test converting configuration to dictionary."""
        config = OptimizedFunctionConfig(
            func=sample_func,
            objectives=sample_objectives,
            configuration_space=sample_config_space,
            execution_mode="edge_analytics",
        )

        config_dict = config.to_dict()

        assert config_dict["func"] == sample_func
        assert config_dict["objectives"] == sample_objectives
        assert config_dict["configuration_space"] == sample_config_space
        assert config_dict["execution_mode"] == "edge_analytics"
        assert "default_config" in config_dict
        assert "constraints" in config_dict

    def test_to_dict_preserves_extra_config(self) -> None:
        """Test to_dict preserves extra configuration."""
        config = OptimizedFunctionConfig(custom_key="custom_value", other="data")

        config_dict = config.to_dict()

        assert config_dict["custom_key"] == "custom_value"
        assert config_dict["other"] == "data"

    # Validation tests
    def test_validate_valid_config(self, basic_config: OptimizedFunctionConfig) -> None:
        """Test validation passes for valid configuration."""
        result = basic_config.validate()

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_invalid_injection_mode(self) -> None:
        """Test validation fails for invalid injection mode."""
        config = OptimizedFunctionConfig(injection_mode="invalid")

        result = config.validate()

        assert result["is_valid"] is False
        assert any("injection_mode" in error for error in result["errors"])

    def test_validate_invalid_execution_mode(self) -> None:
        """Test validation fails for invalid execution mode."""
        config = OptimizedFunctionConfig(execution_mode="invalid_mode")

        result = config.validate()

        assert result["is_valid"] is False
        assert any("execution_mode" in error for error in result["errors"])

    def test_validate_valid_injection_modes(self) -> None:
        """Test validation passes for all valid injection modes."""
        for mode in ["context", "parameter", "decorator"]:
            config = OptimizedFunctionConfig(injection_mode=mode)
            result = config.validate()
            assert result["is_valid"] is True, f"Failed for mode: {mode}"

    def test_validate_valid_execution_modes(self) -> None:
        """Test validation passes for edge_analytics execution mode.

        Note: Only edge_analytics is currently supported. cloud/hybrid raise
        ConfigurationError (not yet supported), privacy/standard were removed.
        """
        config = OptimizedFunctionConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value
        )
        result = config.validate()
        assert result["is_valid"] is True

    def test_validate_empty_objectives(self) -> None:
        """Test validation fails when objectives are empty."""
        # Empty objectives list will create a default "accuracy" objective
        # So we need to create an ObjectiveSchema with empty objectives directly
        # which will fail during schema construction itself
        with pytest.raises(ValueError, match="At least one objective"):
            ObjectiveSchema.from_objectives([])

    def test_validate_invalid_configuration_space_type(self) -> None:
        """Test validation fails when configuration_space is not a dict."""
        config = OptimizedFunctionConfig(configuration_space="not_a_dict")  # type: ignore

        result = config.validate()

        assert result["is_valid"] is False
        assert any("configuration space" in error.lower() for error in result["errors"])

    def test_validate_invalid_parameter_name_type(self) -> None:
        """Test validation fails when parameter name is not a string."""
        config = OptimizedFunctionConfig(configuration_space={123: (0.0, 1.0)})  # type: ignore

        result = config.validate()

        assert result["is_valid"] is False
        assert any("parameter name" in error.lower() for error in result["errors"])

    def test_validate_invalid_default_config_type(self) -> None:
        """Test validation fails when default_config is not a dict."""
        config = OptimizedFunctionConfig(default_config="not_a_dict")  # type: ignore

        result = config.validate()

        assert result["is_valid"] is False
        assert any("default config" in error.lower() for error in result["errors"])

    def test_validate_invalid_constraints_type(self) -> None:
        """Test validation fails when constraints is not a list."""
        config = OptimizedFunctionConfig(constraints="not_a_list")  # type: ignore

        result = config.validate()

        assert result["is_valid"] is False
        assert any("constraints" in error.lower() for error in result["errors"])

    def test_validate_non_callable_constraint(self) -> None:
        """Test validation fails when constraint is not callable."""
        config = OptimizedFunctionConfig(constraints=["not_callable"])  # type: ignore

        result = config.validate()

        assert result["is_valid"] is False
        assert any("callable" in error.lower() for error in result["errors"])

    def test_validate_invalid_framework_targets_type(self) -> None:
        """Test validation fails when framework_targets is not a list."""
        config = OptimizedFunctionConfig(framework_targets="not_a_list")  # type: ignore

        result = config.validate()

        assert result["is_valid"] is False
        assert any("framework targets" in error.lower() for error in result["errors"])

    def test_validate_non_string_framework_target(self) -> None:
        """Test validation fails when framework target is not a string."""
        config = OptimizedFunctionConfig(framework_targets=[123, "valid"])  # type: ignore

        result = config.validate()

        assert result["is_valid"] is False
        assert any("framework targets" in error.lower() for error in result["errors"])

    def test_validate_warning_both_evaluators(self) -> None:
        """Test validation warns when both custom_evaluator and scoring_function are set."""
        config = OptimizedFunctionConfig(
            custom_evaluator=MagicMock(), scoring_function=MagicMock()
        )

        result = config.validate()

        assert result["is_valid"] is True
        assert len(result["warnings"]) > 0
        assert any(
            "custom_evaluator" in warning.lower() for warning in result["warnings"]
        )

    # Merge tests
    def test_merge_configurations(self) -> None:
        """Test merging two configurations."""
        config1 = OptimizedFunctionConfig(
            configuration_space={"temperature": (0.0, 1.0)}, execution_mode="cloud"
        )

        config2 = OptimizedFunctionConfig(
            configuration_space={"max_tokens": (100, 2000)},
            execution_mode="edge_analytics",
        )

        merged = config1.merge(config2)

        assert "temperature" in merged.configuration_space
        assert "max_tokens" in merged.configuration_space
        assert merged.execution_mode == "edge_analytics"

    def test_merge_deep_merges_dicts(self) -> None:
        """Test merge performs deep merge for dictionary fields."""
        config1 = OptimizedFunctionConfig(
            default_config={"temperature": 0.7, "model": "gpt-4o"}
        )

        config2 = OptimizedFunctionConfig(default_config={"max_tokens": 1000})

        merged = config1.merge(config2)

        assert merged.default_config["temperature"] == 0.7
        assert merged.default_config["model"] == "gpt-4o"
        assert merged.default_config["max_tokens"] == 1000

    def test_merge_overwrites_non_dict_values(self) -> None:
        """Test merge overwrites non-dictionary values."""
        config1 = OptimizedFunctionConfig(injection_mode="context")
        config2 = OptimizedFunctionConfig(injection_mode="parameter")

        merged = config1.merge(config2)

        assert merged.injection_mode == "parameter"

    # Helper method tests
    def test_get_objective_schema(self, sample_objectives: ObjectiveSchema) -> None:
        """Test getting objective schema."""
        config = OptimizedFunctionConfig(objectives=sample_objectives)

        schema = config.get_objective_schema()

        assert schema == sample_objectives
        assert len(schema.objectives) == 2

    def test_get_parameter_space(self, sample_config_space: dict[str, Any]) -> None:
        """Test getting parameter space."""
        config = OptimizedFunctionConfig(configuration_space=sample_config_space)

        space = config.get_parameter_space()

        assert space == sample_config_space

    def test_clone(
        self,
        sample_func: MagicMock,
        sample_objectives: ObjectiveSchema,
        sample_config_space: dict[str, Any],
    ) -> None:
        """Test cloning configuration."""
        original = OptimizedFunctionConfig(
            func=sample_func,
            objectives=sample_objectives,
            configuration_space=sample_config_space,
            execution_mode="cloud",
        )

        cloned = original.clone()

        # Verify it's a different instance
        assert cloned is not original

        # Verify all attributes are equal
        assert cloned.func == original.func
        assert cloned.objective_schema == original.objective_schema
        assert cloned.configuration_space == original.configuration_space
        assert cloned.execution_mode == original.execution_mode

    def test_clone_deep_copies_mutable_fields(self) -> None:
        """Test clone performs deep copy of mutable fields."""
        original = OptimizedFunctionConfig(
            configuration_space={"temperature": (0.0, 1.0)},
            default_config={"model": "gpt-4o"},
        )

        cloned = original.clone()

        # Modify the clone
        cloned.configuration_space["max_tokens"] = (100, 2000)
        cloned.default_config["temperature"] = 0.5

        # Original should not be affected
        assert "max_tokens" not in original.configuration_space
        assert "temperature" not in original.default_config

    # Edge cases
    def test_empty_configuration_space(self) -> None:
        """Test configuration with empty configuration space."""
        config = OptimizedFunctionConfig(configuration_space={})

        result = config.validate()
        assert result["is_valid"] is True

    def test_empty_default_config(self) -> None:
        """Test configuration with empty default config."""
        config = OptimizedFunctionConfig(default_config={})

        result = config.validate()
        assert result["is_valid"] is True

    def test_empty_constraints(self) -> None:
        """Test configuration with empty constraints."""
        config = OptimizedFunctionConfig(constraints=[])

        result = config.validate()
        assert result["is_valid"] is True

    def test_valid_callable_constraints(self) -> None:
        """Test configuration with valid callable constraints."""

        def constraint1(x):
            return x["temperature"] > 0

        constraint2 = MagicMock()

        config = OptimizedFunctionConfig(constraints=[constraint1, constraint2])

        result = config.validate()
        assert result["is_valid"] is True


class TestConfigurationSpaceBuilder:
    """Tests for ConfigurationSpaceBuilder class."""

    @pytest.fixture
    def builder(self) -> ConfigurationSpaceBuilder:
        """Create a fresh builder instance for testing."""
        return ConfigurationSpaceBuilder()

    # Initialization tests
    def test_init(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test builder initialization."""
        assert builder.parameters == []
        assert builder.constraints == []
        assert builder.name is None
        assert builder.description is None

    # Float parameter tests
    def test_add_float_parameter(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test adding a float parameter."""
        result = builder.add_float_parameter(
            "temperature", (0.0, 1.0), default=0.7, description="LLM temperature"
        )

        assert result is builder  # Method chaining
        assert len(builder.parameters) == 1
        param = builder.parameters[0]
        assert param.name == "temperature"
        assert param.type == ParameterType.FLOAT
        assert param.bounds == (0.0, 1.0)
        assert param.default == 0.7
        assert param.description == "LLM temperature"

    def test_add_float_parameter_without_default(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test adding a float parameter without default."""
        builder.add_float_parameter("learning_rate", (0.001, 0.1))

        param = builder.parameters[0]
        assert param.default is None

    # Integer parameter tests
    def test_add_integer_parameter(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test adding an integer parameter."""
        result = builder.add_integer_parameter(
            "max_tokens", (100, 2000), default=1000, description="Maximum tokens"
        )

        assert result is builder  # Method chaining
        assert len(builder.parameters) == 1
        param = builder.parameters[0]
        assert param.name == "max_tokens"
        assert param.type == ParameterType.INTEGER
        assert param.bounds == (100, 2000)
        assert param.default == 1000
        assert param.description == "Maximum tokens"

    def test_add_integer_parameter_without_default(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test adding an integer parameter without default."""
        builder.add_integer_parameter("batch_size", (1, 128))

        param = builder.parameters[0]
        assert param.default is None

    # Categorical parameter tests
    def test_add_categorical_parameter(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test adding a categorical parameter."""
        result = builder.add_categorical_parameter(
            "model",
            ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            default="gpt-4o",
            description="Model name",
        )

        assert result is builder  # Method chaining
        assert len(builder.parameters) == 1
        param = builder.parameters[0]
        assert param.name == "model"
        assert param.type == ParameterType.CATEGORICAL
        assert param.bounds == ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        assert param.default == "gpt-4o"
        assert param.description == "Model name"

    def test_add_categorical_parameter_with_mixed_types(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test adding a categorical parameter with mixed types."""
        builder.add_categorical_parameter("param", [0.0, 0.3, 0.7, 1.0])

        param = builder.parameters[0]
        assert param.bounds == [0.0, 0.3, 0.7, 1.0]

    # Boolean parameter tests
    def test_add_boolean_parameter(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test adding a boolean parameter."""
        result = builder.add_boolean_parameter(
            "use_cache", default=True, description="Enable caching"
        )

        assert result is builder  # Method chaining
        assert len(builder.parameters) == 1
        param = builder.parameters[0]
        assert param.name == "use_cache"
        assert param.type == ParameterType.BOOLEAN
        assert param.bounds == [True, False]
        assert param.default is True
        assert param.description == "Enable caching"

    def test_add_boolean_parameter_without_default(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test adding a boolean parameter without default."""
        builder.add_boolean_parameter("verbose")

        param = builder.parameters[0]
        assert param.default is None

    # Constraint tests
    def test_add_constraint(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test adding a constraint function."""

        def constraint(x):
            return x["temperature"] < x["top_p"]

        result = builder.add_constraint(constraint)

        assert result is builder  # Method chaining
        assert len(builder.constraints) == 1
        assert builder.constraints[0] == constraint

    def test_add_multiple_constraints(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test adding multiple constraints."""

        def constraint1(x):
            return x["a"] < x["b"]

        def constraint2(x):
            return x["b"] < x["c"]

        builder.add_constraint(constraint1).add_constraint(constraint2)

        assert len(builder.constraints) == 2

    # Name and description tests
    def test_set_name(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test setting configuration space name."""
        result = builder.set_name("Test Configuration")

        assert result is builder  # Method chaining
        assert builder.name == "Test Configuration"

    def test_set_description(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test setting configuration space description."""
        result = builder.set_description("A test configuration space")

        assert result is builder  # Method chaining
        assert builder.description == "A test configuration space"

    # Build tests
    def test_build_empty(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test building an empty configuration space."""
        space = builder.build()

        assert space == {}

    def test_build_with_float_parameters(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test building configuration space with float parameters."""
        builder.add_float_parameter("temperature", (0.0, 1.0))
        builder.add_float_parameter("top_p", (0.0, 1.0))

        space = builder.build()

        assert space == {
            "temperature": (0.0, 1.0),
            "top_p": (0.0, 1.0),
        }

    def test_build_with_integer_parameters(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test building configuration space with integer parameters."""
        builder.add_integer_parameter("max_tokens", (100, 2000))
        builder.add_integer_parameter("batch_size", (1, 128))

        space = builder.build()

        assert space == {
            "max_tokens": (100, 2000),
            "batch_size": (1, 128),
        }

    def test_build_with_categorical_parameters(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test building configuration space with categorical parameters."""
        builder.add_categorical_parameter("model", ["gpt-4o", "gpt-4o-mini"])
        builder.add_categorical_parameter("prompt_style", ["direct", "step-by-step"])

        space = builder.build()

        assert space == {
            "model": ["gpt-4o", "gpt-4o-mini"],
            "prompt_style": ["direct", "step-by-step"],
        }

    def test_build_with_boolean_parameters(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test building configuration space with boolean parameters."""
        builder.add_boolean_parameter("use_cache")
        builder.add_boolean_parameter("verbose")

        space = builder.build()

        assert space == {
            "use_cache": [True, False],
            "verbose": [True, False],
        }

    def test_build_with_mixed_parameters(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test building configuration space with mixed parameter types."""
        builder.add_float_parameter("temperature", (0.0, 1.0))
        builder.add_integer_parameter("max_tokens", (100, 2000))
        builder.add_categorical_parameter("model", ["gpt-4o", "gpt-4o-mini"])
        builder.add_boolean_parameter("use_cache")

        space = builder.build()

        assert len(space) == 4
        assert space["temperature"] == (0.0, 1.0)
        assert space["max_tokens"] == (100, 2000)
        assert space["model"] == ["gpt-4o", "gpt-4o-mini"]
        assert space["use_cache"] == [True, False]

    def test_build_typed(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test building a typed ConfigurationSpace object."""
        builder.add_float_parameter("temperature", (0.0, 1.0))
        builder.add_constraint(lambda x: x["temperature"] > 0)
        builder.set_name("Test Space")
        builder.set_description("Test description")

        space = builder.build_typed()

        assert len(space.parameters) == 1
        assert space.constraints is not None
        assert len(space.constraints) == 1
        assert space.name == "Test Space"
        assert space.description == "Test description"

    def test_build_typed_without_constraints(
        self, builder: ConfigurationSpaceBuilder
    ) -> None:
        """Test build_typed without constraints."""
        builder.add_float_parameter("temperature", (0.0, 1.0))

        space = builder.build_typed()

        assert len(space.parameters) == 1
        assert space.constraints is None

    # Method chaining tests
    def test_method_chaining(self, builder: ConfigurationSpaceBuilder) -> None:
        """Test fluent interface with method chaining."""
        space = (
            builder.add_float_parameter("temperature", (0.0, 1.0))
            .add_integer_parameter("max_tokens", (100, 2000))
            .add_categorical_parameter("model", ["gpt-4o", "gpt-4o-mini"])
            .add_boolean_parameter("use_cache")
            .add_constraint(lambda x: x["temperature"] > 0)
            .set_name("Chained Config")
            .set_description("Built with chaining")
            .build()
        )

        assert len(space) == 4
        assert builder.name == "Chained Config"
        assert builder.description == "Built with chaining"
        assert len(builder.constraints) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions for common configurations."""

    # create_simple_config_space tests
    def test_create_simple_config_space_default(self) -> None:
        """Test create_simple_config_space with default parameters."""
        space = create_simple_config_space()

        assert "temperature" in space
        assert "max_tokens" in space
        assert space["temperature"] == [0.0, 0.3, 0.7]
        assert space["max_tokens"] == [500, 1000, 2000]

    def test_create_simple_config_space_with_model_choices(self) -> None:
        """Test create_simple_config_space with model choices."""
        space = create_simple_config_space(model_choices=["gpt-4o", "gpt-4o-mini"])

        assert "model" in space
        assert space["model"] == ["gpt-4o", "gpt-4o-mini"]

    def test_create_simple_config_space_with_temperature_range(self) -> None:
        """Test create_simple_config_space with temperature range."""
        space = create_simple_config_space(temperature_range=(0.0, 2.0))

        assert "temperature" in space
        assert space["temperature"] == (0.0, 2.0)

    def test_create_simple_config_space_with_max_tokens_range(self) -> None:
        """Test create_simple_config_space with max_tokens range."""
        space = create_simple_config_space(max_tokens_range=(50, 4000))

        assert "max_tokens" in space
        assert space["max_tokens"] == (50, 4000)

    def test_create_simple_config_space_with_all_params(self) -> None:
        """Test create_simple_config_space with all parameters."""
        space = create_simple_config_space(
            model_choices=["gpt-4o", "gpt-4o-mini"],
            temperature_range=(0.0, 2.0),
            max_tokens_range=(50, 4000),
        )

        assert len(space) == 3
        assert space["model"] == ["gpt-4o", "gpt-4o-mini"]
        assert space["temperature"] == (0.0, 2.0)
        assert space["max_tokens"] == (50, 4000)

    # create_advanced_config_space tests
    def test_create_advanced_config_space_default(self) -> None:
        """Test create_advanced_config_space with default parameters."""
        space = create_advanced_config_space()

        assert "model" in space
        assert "temperature" in space
        assert "max_tokens" in space
        assert "prompt_style" in space

        # Check default model choices
        assert DEFAULT_MODEL in space["model"]
        assert "gpt-4o-mini" in space["model"]

        # Check temperature choices
        assert space["temperature"] == [0.0, 0.1, 0.3, 0.5]

        # Check max_tokens choices
        assert space["max_tokens"] == [500, 1000, 1500, 2000]

        # Check prompt_style choices
        assert space["prompt_style"] == ["direct", "step-by-step", "teach"]

    def test_create_advanced_config_space_without_prompt_styles(self) -> None:
        """Test create_advanced_config_space without prompt styles."""
        space = create_advanced_config_space(include_prompt_styles=False)

        assert "model" in space
        assert "temperature" in space
        assert "max_tokens" in space
        assert "prompt_style" not in space

    def test_create_advanced_config_space_with_model_choices(self) -> None:
        """Test create_advanced_config_space with custom model choices."""
        custom_models = ["claude-3-opus", "claude-3-sonnet"]
        space = create_advanced_config_space(model_choices=custom_models)

        assert space["model"] == custom_models

    def test_create_advanced_config_space_with_all_params(self) -> None:
        """Test create_advanced_config_space with all parameters."""
        custom_models = ["gpt-4o", "claude-3-opus"]
        space = create_advanced_config_space(
            include_prompt_styles=True, model_choices=custom_models
        )

        assert len(space) == 4
        assert space["model"] == custom_models
        assert "temperature" in space
        assert "max_tokens" in space
        assert "prompt_style" in space

    def test_create_advanced_config_space_returns_dict(self) -> None:
        """Test create_advanced_config_space returns a dictionary."""
        space = create_advanced_config_space()

        assert isinstance(space, dict)

    # Edge cases
    def test_create_simple_config_space_with_empty_model_choices(self) -> None:
        """Test create_simple_config_space with empty model choices."""
        # Empty list evaluates to False, so model parameter won't be added
        space = create_simple_config_space(model_choices=[])

        # Model should not be in the space since empty list is falsy
        assert "model" not in space
        # But temperature and max_tokens should still be there
        assert "temperature" in space
        assert "max_tokens" in space

    def test_create_advanced_config_space_with_empty_model_choices(self) -> None:
        """Test create_advanced_config_space with empty model choices."""
        # Empty list evaluates to False, so it uses default model choices
        space = create_advanced_config_space(model_choices=[])

        # When model_choices is empty (falsy), the function uses default models
        assert "model" in space
        assert DEFAULT_MODEL in space["model"]
        assert "gpt-4o-mini" in space["model"]

    def test_create_simple_config_space_uses_defaults(self) -> None:
        """Test create_simple_config_space uses DEFAULT constants."""
        space = create_simple_config_space(
            model_choices=["gpt-4o"],
            temperature_range=(0.0, 1.0),
            max_tokens_range=(100, 2000),
        )

        # The defaults should be used in builder, but not visible in dict output
        assert "model" in space
        assert "temperature" in space
        assert "max_tokens" in space
