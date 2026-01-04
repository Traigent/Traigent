"""Configuration mapper for agent specifications.

This module provides utilities to map Traigent configuration space
to platform-specific agent parameters and merge them with agent
specifications during optimization.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-AGENTS REQ-AGNT-013

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from traigent.utils.exceptions import ConfigurationError
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.cloud.models import AgentSpecification

logger = get_logger(__name__)


@dataclass
class ParameterMapping:
    """Mapping configuration for a single parameter."""

    source_key: str  # Key in Traigent config space
    target_key: str  # Key in agent specification
    target_section: str = "model_parameters"  # Section to place parameter
    transform: Callable[..., Any] | None = None  # Optional transformation function
    default_value: Any | None = None  # Default value if source missing
    validation: Callable[..., Any] | None = None  # Optional validation function
    description: str = ""  # Human-readable description


@dataclass
class PlatformMapping:
    """Complete mapping configuration for a platform."""

    platform: str
    parameter_mappings: list[ParameterMapping] = field(default_factory=list)
    template_mappings: dict[str, str] = field(
        default_factory=dict
    )  # Prompt template variable mappings
    custom_transformers: dict[str, Callable[..., Any]] = field(
        default_factory=dict
    )  # Custom transformation functions
    validation_rules: list[Callable[..., Any]] = field(
        default_factory=list
    )  # Platform-specific validation


class ConfigurationMapper:
    """Maps Traigent configurations to agent specifications."""

    def __init__(self) -> None:
        """Initialize configuration mapper."""
        self._platform_mappings: dict[str, PlatformMapping] = {}
        self._register_default_mappings()

    def register_platform_mapping(self, mapping: PlatformMapping) -> None:
        """Register a platform mapping configuration.

        Args:
            mapping: Platform mapping configuration
        """
        self._platform_mappings[mapping.platform.lower()] = mapping
        logger.info(f"Registered mapping for platform: {mapping.platform}")

    def apply_configuration(
        self,
        agent_spec: AgentSpecification,
        config: dict[str, Any],
        preserve_original: bool = True,
    ) -> AgentSpecification:
        """Apply Traigent configuration to agent specification.

        Args:
            agent_spec: Original agent specification
            config: Traigent configuration to apply
            preserve_original: Whether to preserve the original spec

        Returns:
            Updated agent specification

        Raises:
            ConfigurationError: If mapping fails
        """
        if preserve_original:
            spec = copy.deepcopy(agent_spec)
        else:
            spec = agent_spec

        platform = (spec.agent_platform or "").lower()

        if platform not in self._platform_mappings:
            logger.warning(f"No mapping registered for platform: {platform}")
            return spec

        mapping = self._platform_mappings[platform]

        try:
            # Apply parameter mappings
            spec = self._apply_parameter_mappings(spec, config, mapping)

            # Apply template mappings
            spec = self._apply_template_mappings(spec, config, mapping)

            # Apply custom transformations
            spec = self._apply_custom_transformations(spec, config, mapping)

            # Validate result
            self._validate_mapped_specification(spec, mapping)

            return spec

        except Exception as e:
            raise ConfigurationError(
                f"Failed to apply configuration for platform {platform}: {e}",
                {"platform": platform, "config": config},
            ) from e

    def get_supported_platforms(self) -> list[str]:
        """Get list of supported platforms.

        Returns:
            List of platform names
        """
        return list(self._platform_mappings.keys())

    def get_platform_mapping(self, platform: str) -> PlatformMapping | None:
        """Get mapping configuration for a platform.

        Args:
            platform: Platform name

        Returns:
            Platform mapping or None if not found
        """
        return self._platform_mappings.get(platform.lower())

    def validate_configuration_compatibility(  # noqa: C901
        self, agent_spec: AgentSpecification, config_space: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that configuration space is compatible with agent.

        Args:
            agent_spec: Agent specification
            config_space: Traigent configuration space

        Returns:
            Validation results
        """
        platform = (
            agent_spec.agent_platform.lower() if agent_spec.agent_platform else ""
        )
        mapping = self._platform_mappings.get(platform)

        if not mapping:
            return {
                "compatible": False,
                "errors": [f"No mapping available for platform: {platform}"],
                "warnings": [],
                "supported_parameters": [],
            }

        errors: list[str] = []
        warnings = []
        supported_parameters = []

        # Check if config space parameters can be mapped
        for param_name in config_space.keys():
            mapped = False
            for param_mapping in mapping.parameter_mappings:
                if param_mapping.source_key == param_name:
                    mapped = True
                    supported_parameters.append(param_name)

                    # Validate parameter space definition if validation function exists
                    if param_mapping.validation:
                        try:
                            # For config space validation, check if it's a list or tuple
                            param_space = config_space[param_name]
                            if isinstance(param_space, (list, tuple)):
                                # Validate each possible value
                                for value in param_space:
                                    param_mapping.validation(value)
                            else:
                                # Single value validation
                                param_mapping.validation(param_space)
                        except (ValueError, TypeError) as e:
                            warnings.append(f"Parameter {param_name}: {e}")
                    break

            if not mapped:
                warnings.append(
                    f"Parameter '{param_name}' has no mapping for platform '{platform}'"
                )

        return {
            "compatible": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "supported_parameters": supported_parameters,
        }

    def _apply_parameter_mappings(  # noqa: C901
        self, spec: AgentSpecification, config: dict[str, Any], mapping: PlatformMapping
    ) -> AgentSpecification:
        """Apply parameter mappings to specification."""
        for param_mapping in mapping.parameter_mappings:
            source_value = config.get(
                param_mapping.source_key, param_mapping.default_value
            )

            if source_value is None:
                continue

            # Apply transformation if specified
            if param_mapping.transform:
                transform_name = getattr(
                    param_mapping.transform,
                    "__name__",
                    param_mapping.transform.__class__.__name__,
                )
                try:
                    target_value = param_mapping.transform(source_value)
                except ConfigurationError:
                    raise
                except ValueError as err:
                    raise ConfigurationError(
                        (
                            f"Transform '{transform_name}' for parameter "
                            f"'{param_mapping.source_key}' produced an invalid value: {err}"
                        ),
                        {
                            "source_key": param_mapping.source_key,
                            "target_key": param_mapping.target_key,
                            "transform": transform_name,
                        },
                    ) from err
                except Exception as e:
                    raise ConfigurationError(
                        (
                            f"Transform '{transform_name}' raised an unexpected error for "
                            f"'{param_mapping.source_key}': {e}"
                        ),
                        {
                            "source_key": param_mapping.source_key,
                            "target_key": param_mapping.target_key,
                            "transform": transform_name,
                        },
                    ) from e
            else:
                target_value = source_value

            # Apply validation if specified
            if param_mapping.validation:
                try:
                    param_mapping.validation(target_value)
                except ValueError as e:
                    raise ConfigurationError(
                        f"Validation failed for {param_mapping.target_key}: {e}"
                    ) from e

            # Set the parameter in the appropriate section
            if param_mapping.target_section == "model_parameters":
                if spec.model_parameters is None:
                    spec.model_parameters = {}
                spec.model_parameters[param_mapping.target_key] = target_value
            elif param_mapping.target_section == "custom_tools":
                # Ensure custom_tools is a list of strings
                if spec.custom_tools is None:
                    spec.custom_tools = []
                if isinstance(target_value, list):
                    spec.custom_tools.extend([str(v) for v in target_value])
                else:
                    spec.custom_tools.append(str(target_value))
            else:
                # Set as direct attribute
                setattr(spec, param_mapping.target_key, target_value)

        return spec

    def _apply_template_mappings(
        self, spec: AgentSpecification, config: dict[str, Any], mapping: PlatformMapping
    ) -> AgentSpecification:
        """Apply template variable mappings."""
        if not mapping.template_mappings or not spec.prompt_template:
            return spec

        template = spec.prompt_template

        # Replace template variables based on config
        for template_var, config_key in mapping.template_mappings.items():
            if config_key in config:
                placeholder = f"{{{template_var}}}"
                if placeholder in template:
                    template = template.replace(placeholder, str(config[config_key]))

        spec.prompt_template = template
        return spec

    def _apply_custom_transformations(
        self, spec: AgentSpecification, config: dict[str, Any], mapping: PlatformMapping
    ) -> AgentSpecification:
        """Apply custom transformation functions."""
        for transform_name, transform_func in mapping.custom_transformers.items():
            try:
                spec = transform_func(spec, config)
            except ConfigurationError:
                raise
            except Exception as e:
                transformer_label = transform_name or getattr(
                    transform_func, "__name__", transform_func.__class__.__name__
                )
                raise ConfigurationError(
                    f"Custom transformer '{transformer_label}' failed: {e}",
                    {"transformer": transformer_label},
                ) from e

        return spec

    def _validate_mapped_specification(
        self, spec: AgentSpecification, mapping: PlatformMapping
    ) -> None:
        """Validate the mapped specification."""
        for validation_rule in mapping.validation_rules:
            try:
                validation_rule(spec)
            except ValueError as e:
                raise ConfigurationError(f"Validation failed: {e}") from e

    def _register_default_mappings(self) -> None:
        """Register default platform mappings."""
        # LangChain mapping
        langchain_mapping = PlatformMapping(
            platform="langchain",
            parameter_mappings=[
                ParameterMapping(
                    source_key="model", target_key="model", description="LLM model name"
                ),
                ParameterMapping(
                    source_key="temperature",
                    target_key="temperature",
                    validation=self._validate_temperature,
                    description="Sampling temperature",
                ),
                ParameterMapping(
                    source_key="max_tokens",
                    target_key="max_tokens",
                    validation=self._validate_positive_int,
                    description="Maximum output tokens",
                ),
                ParameterMapping(
                    source_key="top_p",
                    target_key="top_p",
                    validation=self._validate_probability,
                    description="Nucleus sampling parameter",
                ),
                ParameterMapping(
                    source_key="frequency_penalty",
                    target_key="frequency_penalty",
                    validation=self._validate_penalty,
                    description="Frequency penalty",
                ),
                ParameterMapping(
                    source_key="presence_penalty",
                    target_key="presence_penalty",
                    validation=self._validate_penalty,
                    description="Presence penalty",
                ),
            ],
            template_mappings={
                "system_message": "system_prompt",
                "context": "context_data",
            },
        )

        # OpenAI mapping
        openai_mapping = PlatformMapping(
            platform="openai",
            parameter_mappings=[
                ParameterMapping(
                    source_key="model",
                    target_key="model",
                    description="OpenAI model name",
                ),
                ParameterMapping(
                    source_key="temperature",
                    target_key="temperature",
                    validation=self._validate_temperature,
                    description="Sampling temperature",
                ),
                ParameterMapping(
                    source_key="max_tokens",
                    target_key="max_tokens",
                    validation=self._validate_positive_int,
                    description="Maximum output tokens",
                ),
                ParameterMapping(
                    source_key="top_p",
                    target_key="top_p",
                    validation=self._validate_probability,
                    description="Top-p sampling",
                ),
                ParameterMapping(
                    source_key="frequency_penalty",
                    target_key="frequency_penalty",
                    validation=self._validate_penalty,
                    description="Frequency penalty",
                ),
                ParameterMapping(
                    source_key="presence_penalty",
                    target_key="presence_penalty",
                    validation=self._validate_penalty,
                    description="Presence penalty",
                ),
                ParameterMapping(
                    source_key="stop_sequences",
                    target_key="stop",
                    transform=lambda x: x if isinstance(x, list) else [x],
                    description="Stop sequences",
                ),
            ],
        )

        self.register_platform_mapping(langchain_mapping)
        self.register_platform_mapping(openai_mapping)

        # Anthropic mapping
        anthropic_mapping = PlatformMapping(
            platform="anthropic",
            parameter_mappings=[
                ParameterMapping(
                    source_key="model",
                    target_key="model",
                    description="Anthropic model name (claude-3-opus, claude-3-sonnet, claude-3-haiku)",
                ),
                ParameterMapping(
                    source_key="temperature",
                    target_key="temperature",
                    validation=self._validate_temperature,
                    description="Sampling temperature",
                ),
                ParameterMapping(
                    source_key="max_tokens",
                    target_key="max_tokens_to_sample",
                    validation=self._validate_positive_int,
                    description="Maximum output tokens",
                ),
                ParameterMapping(
                    source_key="top_p",
                    target_key="top_p",
                    validation=self._validate_probability,
                    description="Nucleus sampling parameter",
                ),
                ParameterMapping(
                    source_key="top_k",
                    target_key="top_k",
                    validation=self._validate_positive_int,
                    description="Top-k sampling parameter",
                ),
                ParameterMapping(
                    source_key="stop_sequences",
                    target_key="stop_sequences",
                    description="List of stop sequences",
                ),
            ],
        )

        # Cohere mapping
        cohere_mapping = PlatformMapping(
            platform="cohere",
            parameter_mappings=[
                ParameterMapping(
                    source_key="model",
                    target_key="model",
                    description="Cohere model name (command, command-r, etc.)",
                ),
                ParameterMapping(
                    source_key="temperature",
                    target_key="temperature",
                    validation=self._validate_temperature,
                    description="Sampling temperature",
                ),
                ParameterMapping(
                    source_key="max_tokens",
                    target_key="max_tokens",
                    validation=self._validate_positive_int,
                    description="Maximum output tokens",
                ),
                ParameterMapping(
                    source_key="top_p",
                    target_key="p",
                    validation=self._validate_probability,
                    description="Nucleus sampling parameter",
                ),
                ParameterMapping(
                    source_key="top_k",
                    target_key="k",
                    validation=self._validate_positive_int,
                    description="Top-k sampling parameter",
                ),
                ParameterMapping(
                    source_key="frequency_penalty",
                    target_key="frequency_penalty",
                    validation=self._validate_penalty,
                    description="Frequency penalty",
                ),
                ParameterMapping(
                    source_key="presence_penalty",
                    target_key="presence_penalty",
                    validation=self._validate_penalty,
                    description="Presence penalty",
                ),
                ParameterMapping(
                    source_key="seed",
                    target_key="seed",
                    validation=self._validate_positive_int,
                    description="Random seed",
                ),
            ],
        )

        # HuggingFace mapping
        huggingface_mapping = PlatformMapping(
            platform="huggingface",
            parameter_mappings=[
                ParameterMapping(
                    source_key="model",
                    target_key="model_id",
                    description="HuggingFace model ID",
                ),
                ParameterMapping(
                    source_key="temperature",
                    target_key="temperature",
                    validation=self._validate_temperature,
                    description="Sampling temperature",
                ),
                ParameterMapping(
                    source_key="max_tokens",
                    target_key="max_new_tokens",
                    validation=self._validate_positive_int,
                    description="Maximum new tokens to generate",
                ),
                ParameterMapping(
                    source_key="top_p",
                    target_key="top_p",
                    validation=self._validate_probability,
                    description="Nucleus sampling parameter",
                ),
                ParameterMapping(
                    source_key="top_k",
                    target_key="top_k",
                    validation=self._validate_positive_int,
                    description="Top-k sampling parameter",
                ),
                ParameterMapping(
                    source_key="stop_sequences",
                    target_key="stop",
                    description="List of stop sequences",
                ),
                ParameterMapping(
                    source_key="seed",
                    target_key="seed",
                    validation=self._validate_positive_int,
                    description="Random seed",
                ),
            ],
        )

        # Register new platform mappings
        self.register_platform_mapping(anthropic_mapping)
        self.register_platform_mapping(cohere_mapping)
        self.register_platform_mapping(huggingface_mapping)

    def _validate_temperature(self, value: float) -> None:
        """Validate temperature parameter."""
        if not isinstance(value, (int, float)) or not 0 <= value <= 2:
            raise ValueError("Temperature must be between 0 and 2")

    def _validate_probability(self, value: float) -> None:
        """Validate probability parameter."""
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("Probability must be between 0 and 1")

    def _validate_penalty(self, value: float) -> None:
        """Validate penalty parameter."""
        if not isinstance(value, (int, float)) or not -2 <= value <= 2:
            raise ValueError("Penalty must be between -2 and 2")

    def _validate_positive_int(self, value: int) -> None:
        """Validate positive integer parameter."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Value must be a positive integer")


# Global configuration mapper instance
config_mapper = ConfigurationMapper()


def apply_config_to_agent(
    agent_spec: AgentSpecification,
    config: dict[str, Any],
    preserve_original: bool = True,
) -> AgentSpecification:
    """Apply Traigent configuration to agent specification.

    Args:
        agent_spec: Original agent specification
        config: Traigent configuration to apply
        preserve_original: Whether to preserve the original spec

    Returns:
        Updated agent specification
    """
    return config_mapper.apply_configuration(agent_spec, config, preserve_original)


def validate_config_compatibility(
    agent_spec: AgentSpecification, config_space: dict[str, Any]
) -> dict[str, Any]:
    """Validate configuration space compatibility with agent.

    Args:
        agent_spec: Agent specification
        config_space: Traigent configuration space

    Returns:
        Validation results
    """
    return config_mapper.validate_configuration_compatibility(agent_spec, config_space)


def register_platform_mapping(mapping: PlatformMapping) -> None:
    """Register a new platform mapping.

    Args:
        mapping: Platform mapping configuration
    """
    config_mapper.register_platform_mapping(mapping)


def get_supported_platforms() -> list[str]:
    """Get list of supported platforms.

    Returns:
        List of platform names
    """
    return config_mapper.get_supported_platforms()
