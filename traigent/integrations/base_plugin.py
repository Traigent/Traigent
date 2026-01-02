"""Base plugin architecture for Traigent integrations.

This module provides the abstract base class and utilities for creating
integration plugins that handle parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Maintainability FUNC-INTEGRATIONS FUNC-INVOKERS REQ-INT-008 REQ-INJ-002 SYNC-IntegrationHook

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from traigent.utils.exceptions import TraigentError

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig

logger = logging.getLogger(__name__)


class IntegrationPriority(Enum):
    """Priority levels for integration plugins."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class PluginMetadata:
    """Metadata for an integration plugin."""

    name: str
    version: str
    supported_packages: list[str]
    priority: IntegrationPriority = IntegrationPriority.NORMAL
    description: str = ""
    author: str = ""
    requires_packages: list[str] = field(default_factory=list)
    supports_versions: dict[str, str] = field(
        default_factory=dict
    )  # package: version_spec


@dataclass
class ValidationRule:
    """Validation rule for a parameter."""

    required: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any | None] | None = None
    pattern: str | None = None
    custom_validator: str | None = None  # Name of custom validation method


class IntegrationPlugin(ABC):
    """Abstract base class for all integration plugins.

    This class defines the interface that all integration plugins must implement
    to provide parameter mappings, validation, and override functionality.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the plugin with optional configuration override.

        Args:
            config_path: Optional path to YAML/JSON configuration file for overrides
        """
        self.metadata = self._get_metadata()
        self._parameter_mappings = self._get_default_mappings()
        self._validation_rules = self._get_validation_rules()
        self._enabled = True
        self._config_path = config_path

        # Load configuration overrides if provided
        if config_path:
            self._load_config_overrides(config_path)

    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """Return metadata about this plugin.

        Returns:
            PluginMetadata object describing the plugin
        """
        raise NotImplementedError

    @abstractmethod
    def _get_default_mappings(self) -> dict[str, str]:
        """Return default parameter mappings.

        Returns:
            Dict mapping Traigent parameter names to framework-specific names
        """
        raise NotImplementedError

    @abstractmethod
    def _get_validation_rules(self) -> dict[str, ValidationRule]:
        """Return validation rules for parameters.

        Returns:
            Dict mapping parameter names to their validation rules
        """
        raise NotImplementedError

    @abstractmethod
    def get_target_classes(self) -> list[str]:
        """Return list of class names this plugin should override.

        Returns:
            List of fully qualified class names (e.g., 'openai.ChatCompletion')
        """
        raise NotImplementedError

    @abstractmethod
    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of class names to method names to override.

        Returns:
            Dict mapping class names to lists of method names
        """
        raise NotImplementedError

    def _normalize_config(
        self, config: "TraigentConfig | dict[str, Any] | None"
    ) -> "TraigentConfig":
        """Convert supported config payloads into a TraigentConfig instance."""
        from traigent.config.types import TraigentConfig as ConfigType

        if isinstance(config, ConfigType):
            return config

        if isinstance(config, dict):
            config_copy = dict(config)
            custom_params_section = config_copy.pop("custom_params", None)
            if isinstance(custom_params_section, Mapping):
                for key, value in custom_params_section.items():
                    config_copy.setdefault(key, value)
            try:
                return ConfigType.from_dict(config_copy)
            except Exception as exc:  # pragma: no cover - defensive
                raise TraigentError(
                    f"Invalid configuration payload provided to {self.__class__.__name__}: {exc}"
                ) from exc

        if config is None:
            raise TraigentError(
                f"{self.__class__.__name__} requires a configuration object to apply overrides"
            )

        raise TraigentError(
            f"Unsupported configuration type for {self.__class__.__name__}: "
            f"{type(config).__name__}"
        )

    def get_parameter_mappings(self) -> dict[str, str]:
        """Get the current parameter mappings.

        Returns:
            Dict of Traigent param -> framework param mappings
        """
        return self._parameter_mappings.copy()

    def validate_config(self, config: "TraigentConfig") -> bool:
        """Validate a TraigentConfig against this plugin's rules.

        Args:
            config: TraigentConfig to validate

        Returns:
            True if config is valid, False otherwise

        Raises:
            TraigentError: If validation fails with details
        """
        errors = []

        for param_name, rule in self._validation_rules.items():
            # Get value from both direct attributes and custom_params
            value = None
            if hasattr(config, param_name):
                value = getattr(config, param_name)
            if (
                value is None
                and hasattr(config, "custom_params")
                and config.custom_params
            ):
                value = config.custom_params.get(param_name)

            # Check required parameters
            if rule.required and value is None:
                errors.append(f"Required parameter '{param_name}' is missing")
                continue

            if value is None:
                continue

            # Validate min/max values (ensure comparable types)
            try:
                if rule.min_value is not None and value < rule.min_value:
                    errors.append(
                        f"Parameter '{param_name}' value {value} is below minimum {rule.min_value}"
                    )
            except TypeError:
                # Cannot compare types - treat as invalid
                errors.append(
                    f"Parameter '{param_name}' value {value} has incompatible type (expected numeric)"
                )

            try:
                if rule.max_value is not None and value > rule.max_value:
                    errors.append(
                        f"Parameter '{param_name}' value {value} is above maximum {rule.max_value}"
                    )
            except TypeError:
                # Cannot compare types - treat as invalid
                errors.append(
                    f"Parameter '{param_name}' value {value} has incompatible type (expected numeric)"
                )

            # Validate allowed values
            if rule.allowed_values is not None and value not in rule.allowed_values:
                errors.append(
                    f"Parameter '{param_name}' value {value} not in allowed values: {rule.allowed_values}"
                )

            # Custom validation
            if rule.custom_validator:
                custom_method = getattr(self, rule.custom_validator, None)
                if custom_method:
                    custom_errors = custom_method(param_name, value)
                    if custom_errors:
                        errors.extend(custom_errors)

        if errors:
            raise TraigentError(
                f"Validation failed for {self.metadata.name}: " + "; ".join(errors)
            )

        return True

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply parameter overrides based on TraigentConfig.

        Args:
            kwargs: Original keyword arguments
            config: TraigentConfig or dict with override parameters

        Returns:
            Modified kwargs with overrides applied
        """
        if not self._enabled:
            return kwargs

        config_obj = self._normalize_config(config)

        # Validate config first
        self.validate_config(config_obj)

        # Apply parameter mappings
        overridden = kwargs.copy()

        # Get all config values (both direct attributes and custom_params)
        config_dict = {}

        # Add direct TraigentConfig attributes
        for attr in [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
            "seed",
        ]:
            if hasattr(config_obj, attr):
                value = getattr(config_obj, attr)
                if value is not None:
                    config_dict[attr] = value

        # Add custom params
        if hasattr(config_obj, "custom_params") and config_obj.custom_params:
            config_dict.update(config_obj.custom_params)

        # Apply mappings
        mapped_params: set[str] = set()
        for traigent_param, framework_param in self._parameter_mappings.items():
            if traigent_param in config_dict:
                # Only override if user didn't explicitly provide the framework parameter
                if framework_param not in kwargs:
                    overridden[framework_param] = config_dict[traigent_param]
                mapped_params.add(traigent_param)

        # Pass through custom_params that aren't in mappings (framework-native params)
        if hasattr(config_obj, "custom_params") and config_obj.custom_params:
            for param_name, value in config_obj.custom_params.items():
                if param_name not in mapped_params and param_name not in overridden:
                    overridden[param_name] = value

        return overridden

    def _load_config_overrides(self, config_path: Path) -> None:
        """Load configuration overrides from YAML or JSON file.

        Args:
            config_path: Path to configuration file
        """
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return

        try:
            with open(config_path) as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    config_data = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {config_path}")
                    return

            # Override parameter mappings
            if "mappings" in config_data:
                self._parameter_mappings.update(config_data["mappings"])

            # Override validation rules
            if "validation" in config_data:
                for param_name, rule_data in config_data["validation"].items():
                    if param_name not in self._validation_rules:
                        self._validation_rules[param_name] = ValidationRule()

                    rule = self._validation_rules[param_name]
                    for key, value in rule_data.items():
                        if hasattr(rule, key):
                            setattr(rule, key, value)

            # Override metadata if specified
            if "metadata" in config_data:
                for key, value in config_data["metadata"].items():
                    if hasattr(self.metadata, key):
                        setattr(self.metadata, key, value)

            logger.info(f"Loaded configuration overrides from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")

    def is_compatible(self, package_name: str, version: str | None = None) -> bool:
        """Check if this plugin is compatible with a given package/version.

        Args:
            package_name: Name of the package
            version: Optional version string

        Returns:
            True if compatible, False otherwise
        """
        if package_name not in self.metadata.supported_packages:
            return False

        if version and package_name in self.metadata.supports_versions:
            return self._is_version_supported(
                package_name,
                version,
                self.metadata.supports_versions[package_name],
            )

        return True

    def _is_version_supported(
        self,
        package_name: str,
        version: str,
        supported_spec: str | Sequence[str],
    ) -> bool:
        """Check if a version satisfies the plugin's supported specifiers."""

        if isinstance(supported_spec, str):
            specifiers: Sequence[str] = [supported_spec]
        else:
            specifiers = list(supported_spec)

        try:
            parsed_version = Version(version)
        except InvalidVersion:
            logger.warning(
                "Invalid version '%s' for package '%s' while checking compatibility",
                version,
                package_name,
            )
            return False

        matched_specifier = False
        for spec in specifiers:
            try:
                specifier = SpecifierSet(spec)
            except InvalidSpecifier:
                # Fall back to prefix/string comparison for legacy metadata entries
                logger.debug(
                    "Unsupported version specifier '%s' for package '%s'; falling back to prefix match",
                    spec,
                    package_name,
                )
                if parsed_version.public.startswith(spec):
                    return True
                continue

            matched_specifier = True
            if parsed_version in specifier:
                return True

        # If we never parsed a valid specifier, fall back to strict equality
        if not matched_specifier:
            return any(parsed_version.public == spec for spec in specifiers)

        logger.debug(
            "Version %s for package '%s' does not meet supported specifiers %s",
            version,
            package_name,
            specifiers,
        )
        return False

    def enable(self) -> None:
        """Enable this plugin."""
        self._enabled = True

    def disable(self) -> None:
        """Disable this plugin."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.metadata.name}, version={self.metadata.version})"
