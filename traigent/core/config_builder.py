"""Configuration builder for TraiGent optimization system.

This module provides builder patterns and utilities for constructing and
validating configuration objects for the optimization system.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

from traigent.config.types import ExecutionMode
from traigent.core.constants import (
    DEFAULT_EXECUTION_MODE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT_STYLE,
    DEFAULT_TEMPERATURE,
)
from traigent.core.objectives import (
    ObjectiveSchema,
    create_default_objectives,
    normalize_objectives,
    schema_to_objective_names,
)
from traigent.core.types import Parameter, ParameterType
from traigent.core.types_ext import ValidationResult
from traigent.core.utils import create_validation_result
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizedFunctionConfig:
    """Configuration builder for OptimizedFunction.

    This class provides a builder pattern for constructing OptimizedFunction
    configurations with validation and backward compatibility support.
    """

    def __init__(
        self,
        func: Callable[..., Any] | None = None,
        eval_dataset: Any | None = None,
        objectives: list[str] | ObjectiveSchema | None = None,
        configuration_space: dict[str, Any] | None = None,
        default_config: dict[str, Any] | None = None,
        constraints: list[Callable[..., Any]] | None = None,
        injection_mode: str = "context",
        config_param: str | None = None,
        auto_override_frameworks: bool = False,
        framework_targets: list[str] | None = None,
        execution_mode: str = DEFAULT_EXECUTION_MODE,
        local_storage_path: str | None = None,
        minimal_logging: bool = True,
        custom_evaluator: Callable[..., Any] | None = None,
        scoring_function: Callable[..., Any] | None = None,
        metric_functions: dict[str, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize configuration with validation.

        Args:
            func: Function to optimize
            eval_dataset: Evaluation dataset
            objectives: Optimization objectives
            configuration_space: Parameter search space
            default_config: Default parameter values
            constraints: Parameter constraints
            injection_mode: Configuration injection method
            config_param: Parameter name for injection
            auto_override_frameworks: Auto-override frameworks
            framework_targets: Framework target classes
            execution_mode: Execution environment mode
            local_storage_path: Local storage directory
            minimal_logging: Use minimal logging
            custom_evaluator: Custom evaluation function
            scoring_function: Custom scoring function
            metric_functions: Custom metric functions
            **kwargs: Additional configuration
        """
        self.func = func
        self.eval_dataset = eval_dataset
        self.configuration_space = configuration_space or {}
        self.default_config = default_config or {}
        self.constraints = constraints or []
        self.injection_mode = injection_mode
        self.config_param = config_param
        self.auto_override_frameworks = auto_override_frameworks
        self.framework_targets = framework_targets or []
        self.execution_mode = execution_mode
        self.local_storage_path = local_storage_path
        self.minimal_logging = minimal_logging
        schema = normalize_objectives(objectives)
        if schema is None:
            schema = create_default_objectives(["accuracy"])
        self.objective_schema = schema
        self.objectives = schema_to_objective_names(schema)
        self.custom_evaluator = custom_evaluator
        self.scoring_function = scoring_function
        self.metric_functions = metric_functions or {}
        self.extra_config = kwargs

        # Validate configuration
        validation = self.validate()
        if not validation["is_valid"]:
            logger.warning(f"Configuration validation issues: {validation['errors']}")

    @classmethod
    def from_legacy_params(cls, **kwargs: Any) -> OptimizedFunctionConfig:
        """Create configuration from legacy parameter format.

        Args:
            **kwargs: Legacy parameters

        Returns:
            OptimizedFunctionConfig instance
        """
        # Handle backward compatibility for old parameter names
        config_space = kwargs.pop("config_space", kwargs.pop("configuration_space", {}))

        return cls(configuration_space=config_space, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> OptimizedFunctionConfig:
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            OptimizedFunctionConfig instance
        """
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "func": self.func,
            "eval_dataset": self.eval_dataset,
            "objectives": self.objective_schema,
            "configuration_space": self.configuration_space,
            "default_config": self.default_config,
            "constraints": self.constraints,
            "injection_mode": self.injection_mode,
            "config_param": self.config_param,
            "auto_override_frameworks": self.auto_override_frameworks,
            "framework_targets": self.framework_targets,
            "execution_mode": self.execution_mode,
            "local_storage_path": self.local_storage_path,
            "minimal_logging": self.minimal_logging,
            "custom_evaluator": self.custom_evaluator,
            "scoring_function": self.scoring_function,
            "metric_functions": self.metric_functions,
            **self.extra_config,
        }

    def validate(self) -> ValidationResult:
        """Validate the configuration.

        Returns:
            ValidationResult with validation status and errors
        """
        errors = []
        warnings = []

        # Validate injection mode
        valid_injection_modes = ["context", "parameter", "decorator"]
        if self.injection_mode not in valid_injection_modes:
            errors.append(
                f"Invalid injection_mode: {self.injection_mode}. Must be one of {valid_injection_modes}"
            )

        # Validate execution mode
        valid_execution_modes = [
            ExecutionMode.EDGE_ANALYTICS.value,
            ExecutionMode.HYBRID.value,
            ExecutionMode.PRIVACY.value,
            ExecutionMode.STANDARD.value,
            ExecutionMode.CLOUD.value,
        ]
        if self.execution_mode not in valid_execution_modes:
            errors.append(
                f"Invalid execution_mode: {self.execution_mode}. Must be one of {valid_execution_modes}"
            )

        # Validate objectives
        if not self.objective_schema.objectives:
            errors.append("At least one objective must be specified")

        # Validate configuration space
        if self.configuration_space:
            if not isinstance(self.configuration_space, dict):
                errors.append("Configuration space must be a dictionary")
            else:
                # Validate parameter definitions
                for param_name, _param_def in self.configuration_space.items():
                    if not isinstance(param_name, str):
                        errors.append(
                            f"Parameter name must be string, got {type(param_name)}"
                        )

        # Validate default config
        if self.default_config:
            if not isinstance(self.default_config, dict):
                errors.append("Default config must be a dictionary")

        # Validate constraints
        if self.constraints:
            if not isinstance(self.constraints, list):
                errors.append("Constraints must be a list")
            else:
                for i, constraint in enumerate(self.constraints):
                    if not callable(constraint):
                        errors.append(f"Constraint {i} must be callable")

        # Validate framework targets
        if self.framework_targets:
            if not isinstance(self.framework_targets, list):
                errors.append("Framework targets must be a list")
            else:
                if not all(
                    isinstance(target, str) for target in self.framework_targets
                ):
                    errors.append("All framework targets must be strings")

        # Warnings for potentially problematic configurations
        if self.custom_evaluator and self.scoring_function:
            warnings.append(
                "Both custom_evaluator and scoring_function specified - custom_evaluator takes precedence"
            )

        return cast(
            ValidationResult,
            create_validation_result(
                is_valid=len(errors) == 0, errors=errors, warnings=warnings
            ),
        )

    def merge(self, other: OptimizedFunctionConfig) -> OptimizedFunctionConfig:
        """Merge with another configuration.

        Args:
            other: Configuration to merge with

        Returns:
            New merged configuration
        """
        merged_dict = self.to_dict()

        # Deep merge dictionaries
        for key, value in other.to_dict().items():
            if (
                key in merged_dict
                and isinstance(merged_dict[key], dict)
                and isinstance(value, dict)
            ):
                merged_dict[key] = {**merged_dict[key], **value}
            else:
                merged_dict[key] = value

        return OptimizedFunctionConfig.from_dict(merged_dict)

    def get_objective_schema(self) -> ObjectiveSchema:
        """Get or create objective schema from configuration.

        Returns:
            ObjectiveSchema instance
        """
        return self.objective_schema

    def get_parameter_space(self) -> dict[str, Any]:
        """Get validated parameter space.

        Returns:
            Parameter space dictionary
        """
        return self.configuration_space

    def clone(self) -> OptimizedFunctionConfig:
        """Create a deep copy of the configuration.

        Returns:
            New configuration instance
        """
        return OptimizedFunctionConfig.from_dict(copy.deepcopy(self.to_dict()))


class ConfigurationSpaceBuilder:
    """Builder for creating configuration spaces with validation."""

    def __init__(self) -> None:
        """Initialize configuration space builder."""
        self.parameters: list[Parameter] = []
        self.constraints: list[Callable[..., Any]] = []
        self.name: str | None = None
        self.description: str | None = None

    def add_float_parameter(
        self,
        name: str,
        bounds: tuple[float, float],
        default: float | None = None,
        description: str | None = None,
    ) -> ConfigurationSpaceBuilder:
        """Add a float parameter.

        Args:
            name: Parameter name
            bounds: (min, max) bounds
            default: Default value
            description: Parameter description

        Returns:
            Self for method chaining
        """
        param = Parameter(
            name=name,
            type=ParameterType.FLOAT,
            bounds=bounds,
            default=default,
            description=description,
        )
        self.parameters.append(param)
        return self

    def add_integer_parameter(
        self,
        name: str,
        bounds: tuple[int, int],
        default: int | None = None,
        description: str | None = None,
    ) -> ConfigurationSpaceBuilder:
        """Add an integer parameter.

        Args:
            name: Parameter name
            bounds: (min, max) bounds
            default: Default value
            description: Parameter description

        Returns:
            Self for method chaining
        """
        param = Parameter(
            name=name,
            type=ParameterType.INTEGER,
            bounds=bounds,
            default=default,
            description=description,
        )
        self.parameters.append(param)
        return self

    def add_categorical_parameter(
        self,
        name: str,
        choices: list[Any],
        default: Any | None = None,
        description: str | None = None,
    ) -> ConfigurationSpaceBuilder:
        """Add a categorical parameter.

        Args:
            name: Parameter name
            choices: List of possible values
            default: Default value
            description: Parameter description

        Returns:
            Self for method chaining
        """
        param = Parameter(
            name=name,
            type=ParameterType.CATEGORICAL,
            bounds=choices,
            default=default,
            description=description,
        )
        self.parameters.append(param)
        return self

    def add_boolean_parameter(
        self, name: str, default: bool | None = None, description: str | None = None
    ) -> ConfigurationSpaceBuilder:
        """Add a boolean parameter.

        Args:
            name: Parameter name
            default: Default value
            description: Parameter description

        Returns:
            Self for method chaining
        """
        param = Parameter(
            name=name,
            type=ParameterType.BOOLEAN,
            bounds=[True, False],
            default=default,
            description=description,
        )
        self.parameters.append(param)
        return self

    def add_constraint(
        self, constraint: Callable[..., Any]
    ) -> ConfigurationSpaceBuilder:
        """Add a constraint function.

        Args:
            constraint: Constraint function

        Returns:
            Self for method chaining
        """
        self.constraints.append(constraint)
        return self

    def set_name(self, name: str) -> ConfigurationSpaceBuilder:
        """Set configuration space name.

        Args:
            name: Configuration space name

        Returns:
            Self for method chaining
        """
        self.name = name
        return self

    def set_description(self, description: str) -> ConfigurationSpaceBuilder:
        """Set configuration space description.

        Args:
            description: Configuration space description

        Returns:
            Self for method chaining
        """
        self.description = description
        return self

    def build(self) -> dict[str, Any]:
        """Build the configuration space dictionary.

        Returns:
            Configuration space dictionary
        """
        space_dict = {}

        for param in self.parameters:
            if param.type in (ParameterType.FLOAT, ParameterType.INTEGER):
                space_dict[param.name] = param.bounds
            elif param.type == ParameterType.CATEGORICAL:
                space_dict[param.name] = param.bounds
            elif param.type == ParameterType.BOOLEAN:
                space_dict[param.name] = [True, False]

        return space_dict

    def build_typed(self) -> Any:
        """Build a typed ConfigurationSpace object.

        Returns:
            ConfigurationSpace instance
        """
        # Import here to avoid circular imports
        from traigent.core.types import ConfigurationSpace

        space = ConfigurationSpace(
            parameters=self.parameters,
            constraints=self.constraints if self.constraints else None,
            name=self.name,
            description=self.description,
        )

        return space


# Convenience functions for common configurations
def create_simple_config_space(
    model_choices: list[str] | None = None,
    temperature_range: tuple[float, float] | None = None,
    max_tokens_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Create a simple configuration space for common LLM parameters.

    Args:
        model_choices: List of model choices
        temperature_range: Temperature bounds
        max_tokens_range: Max tokens bounds

    Returns:
        Configuration space dictionary
    """
    builder = ConfigurationSpaceBuilder()

    if model_choices:
        builder.add_categorical_parameter("model", model_choices, DEFAULT_MODEL)

    if temperature_range:
        builder.add_float_parameter(
            "temperature", temperature_range, DEFAULT_TEMPERATURE
        )
    else:
        builder.add_categorical_parameter(
            "temperature", [0.0, 0.3, 0.7], DEFAULT_TEMPERATURE
        )

    if max_tokens_range:
        builder.add_integer_parameter(
            "max_tokens", max_tokens_range, DEFAULT_MAX_TOKENS
        )
    else:
        builder.add_categorical_parameter(
            "max_tokens", [500, 1000, 2000], DEFAULT_MAX_TOKENS
        )

    return builder.build()


def create_advanced_config_space(
    include_prompt_styles: bool = True, model_choices: list[str] | None = None
) -> dict[str, Any]:
    """Create an advanced configuration space with more parameters.

    Args:
        include_prompt_styles: Whether to include prompt style parameters
        model_choices: List of model choices

    Returns:
        Configuration space dictionary
    """
    builder = ConfigurationSpaceBuilder()

    # Model selection
    if model_choices:
        builder.add_categorical_parameter("model", model_choices, DEFAULT_MODEL)
    else:
        builder.add_categorical_parameter(
            "model", [DEFAULT_MODEL, "gpt-4o-mini"], DEFAULT_MODEL
        )

    # Temperature
    builder.add_categorical_parameter(
        "temperature", [0.0, 0.1, 0.3, 0.5], DEFAULT_TEMPERATURE
    )

    # Max tokens
    builder.add_categorical_parameter(
        "max_tokens", [500, 1000, 1500, 2000], DEFAULT_MAX_TOKENS
    )

    # Prompt styles (if requested)
    if include_prompt_styles:
        builder.add_categorical_parameter(
            "prompt_style", ["direct", "step-by-step", "teach"], DEFAULT_PROMPT_STYLE
        )

    builder.set_name("Advanced LLM Configuration")
    builder.set_description("Comprehensive configuration space for LLM optimization")

    return builder.build()
