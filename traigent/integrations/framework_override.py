"""Automatic Framework Parameter Override System.

This module provides the core functionality for seamless framework optimization
where Traigent automatically overrides framework parameters during optimization
without requiring any changes to user code.
# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Maintainability FUNC-INTEGRATIONS FUNC-INVOKERS REQ-INT-008 REQ-INJ-002 SYNC-IntegrationHook

Key Features:
- Automatic monkey patching of framework constructors and methods
- Parameter mapping from Traigent config to framework parameters
- Support for multiple frameworks (OpenAI, LangChain, Anthropic, Cohere, HuggingFace)
- Streaming and tool/function calling support
- Method-level parameter injection for completion calls
- Nested framework call handling
- Context-aware override management
- Dynamic parameter discovery and validation (when available)
- Version compatibility management
- Resilient override strategies with fallbacks
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, cast

from ..config.context import get_config
from ..config.types import TraigentConfig
from ..utils.logging import get_logger

# Import BaseOverrideManager from the canonical location
from .base import BaseOverrideManager

# Import static mappings from dedicated module
from .mappings import METHOD_MAPPINGS, PARAMETER_MAPPINGS

logger = get_logger(__name__)

# Backwards compatibility alias
LegacyBaseOverrideManager = BaseOverrideManager

# Import enhanced capabilities if available
try:
    from .utils.discovery import ParameterDiscovery
    from .utils.validation import ParameterValidator
    from .utils.version_compat import VersionCompatibilityManager

    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False


class FrameworkOverrideManager(BaseOverrideManager):
    """Manages automatic framework parameter overrides during optimization."""

    def __init__(self) -> None:
        """Initialize the framework override manager."""
        # Initialize base class if available
        super().__init__()
        self._use_enhanced_features = ENHANCED_FEATURES_AVAILABLE

        # Declare types for optional enhanced components
        self.discovery: ParameterDiscovery | None
        self.validator: ParameterValidator | None
        self.version_manager: VersionCompatibilityManager | None
        self._discovered_mappings: dict[str, Any] = {}

        if ENHANCED_FEATURES_AVAILABLE:
            # Initialize enhanced components
            self.discovery = ParameterDiscovery()
            self.validator = ParameterValidator()
            self.version_manager = VersionCompatibilityManager()
        else:
            self.discovery = None
            self.validator = None
            self.version_manager = None

        self._parameter_mappings = self._init_parameter_mappings()
        self._method_mappings = self._init_method_mappings()

    def _init_parameter_mappings(self) -> dict[str, dict[str, str]]:
        """Initialize parameter mappings for different frameworks.

        Uses static mappings from mappings.py as the baseline.
        These serve as fallback when no plugin is registered for a framework.
        Plugin mappings (via LLMPlugin._get_default_mappings) take precedence.
        """
        # Deep copy to allow instance-level modifications without affecting global mappings
        return {k: dict(v) for k, v in PARAMETER_MAPPINGS.items()}

    def _init_method_mappings(self) -> dict[str, dict[str, list[str]]]:
        """Initialize method mappings for different frameworks.

        Uses static mappings from mappings.py as the baseline.
        Maps class names to methods that should be overridden for parameter injection.
        These serve as fallback when no plugin is registered for a framework.
        """
        # Deep copy to allow instance-level modifications without affecting global mappings
        # Inner lists also need to be copied
        return {
            k: {method: list(params) for method, params in v.items()}
            for k, v in METHOD_MAPPINGS.items()
        }

    def register_framework_target(
        self, target_class: str | type, parameter_mapping: dict[str, str] | None = None
    ) -> None:
        """Register a framework class for automatic parameter override.

        Args:
            target_class: Framework class name or class object to override
            parameter_mapping: Optional custom parameter mapping
        """
        if isinstance(target_class, type):
            class_name = f"{target_class.__module__}.{target_class.__name__}"
        else:
            class_name = target_class

        if parameter_mapping:
            self._parameter_mappings[class_name] = parameter_mapping

    def _create_override_constructor(
        self, original_constructor: Callable[..., Any], class_name: str
    ) -> Callable[..., Any]:
        """Create an overridden constructor that injects Traigent parameters.

        Args:
            original_constructor: Original class constructor
            class_name: Name of the class being overridden

        Returns:
            Overridden constructor with parameter injection
        """
        parameter_mapping = self._parameter_mappings.get(class_name, {})
        override_active = self._override_active  # Capture in closure

        @functools.wraps(original_constructor)
        def override_constructor(*args, **kwargs):
            # Check if we're in an optimization context
            if not getattr(override_active, "enabled", False):
                return original_constructor(*args, **kwargs)

            # Get current Traigent configuration
            config = get_config()
            if not config:
                return original_constructor(*args, **kwargs)

            # Extract configuration values
            config_dict = {}
            if isinstance(config, TraigentConfig):
                config_dict = config.to_dict()
                config_dict.update(config.custom_params)
            elif isinstance(config, dict):
                config_dict = config
            else:
                return original_constructor(*args, **kwargs)

            # Get current configuration space from context to determine what to override
            from traigent.config.context import get_config_space

            config_space = get_config_space()

            # Override parameters based on mapping
            overridden_kwargs = kwargs.copy()
            overrides_applied = []

            for traigent_param, framework_param in parameter_mapping.items():
                if traigent_param in config_dict:
                    # Only override if parameter is in configuration space (being optimized)
                    # or if no configuration space is set (not in optimization)
                    if config_space is None or traigent_param in config_space:
                        override_value = config_dict[traigent_param]
                        original_value = overridden_kwargs.get(
                            framework_param, "not_set"
                        )
                        overridden_kwargs[framework_param] = override_value
                        overrides_applied.append(
                            f"{framework_param}: {original_value} -> {override_value}"
                        )

            # Record what overrides were applied
            # (logging handled separately if needed)

            # Call original constructor with overridden parameters
            return original_constructor(*args, **overridden_kwargs)

        return override_constructor

    def _create_override_method(
        self, original_method: Callable[..., Any], class_name: str, method_name: str
    ) -> Callable[..., Any]:
        """Create an overridden method that injects Traigent parameters.

        Args:
            original_method: Original method
            class_name: Name of the class being overridden
            method_name: Name of the method being overridden

        Returns:
            Overridden method with parameter injection
        """
        parameter_mapping = self._parameter_mappings.get(class_name, {})
        supported_params = self._method_mappings.get(class_name, {}).get(
            method_name, []
        )
        override_active = self._override_active  # Capture in closure

        @functools.wraps(original_method)
        def override_method(instance, *args, **kwargs):
            # Check if we're in an optimization context
            if not getattr(override_active, "enabled", False):
                return original_method(instance, *args, **kwargs)

            # Get current Traigent configuration
            config = get_config()
            if not config:
                return original_method(instance, *args, **kwargs)

            # Extract configuration values
            config_dict = {}
            if isinstance(config, TraigentConfig):
                config_dict = config.to_dict()
                config_dict.update(config.custom_params)
            elif isinstance(config, dict):
                config_dict = config
            else:
                return original_method(instance, *args, **kwargs)

            # Get current configuration space from context to determine what to override
            from traigent.config.context import get_config_space

            config_space = get_config_space()

            # Override parameters based on mapping and supported params
            overridden_kwargs = kwargs.copy()
            overrides_applied = []

            for traigent_param, framework_param in parameter_mapping.items():
                if traigent_param in config_dict and traigent_param in supported_params:
                    # Only override if parameter is in configuration space (being optimized)
                    # or if no configuration space is set (not in optimization)
                    if config_space is None or traigent_param in config_space:
                        override_value = config_dict[traigent_param]
                        original_value = overridden_kwargs.get(
                            framework_param, "not_set"
                        )
                        overridden_kwargs[framework_param] = override_value
                        overrides_applied.append(
                            f"{framework_param}: {original_value} -> {override_value}"
                        )

            # Record what overrides were applied
            # (logging handled separately if needed)

            # Call original method with overridden parameters
            return original_method(instance, *args, **overridden_kwargs)

        @functools.wraps(original_method)
        async def async_override_method(instance, *args, **kwargs):
            # Check if we're in an optimization context
            if not getattr(override_active, "enabled", False):
                return await original_method(instance, *args, **kwargs)

            # Get current Traigent configuration
            config = get_config()
            if not config:
                return await original_method(instance, *args, **kwargs)

            # Extract configuration values
            config_dict = {}
            if isinstance(config, TraigentConfig):
                config_dict = config.to_dict()
                config_dict.update(config.custom_params)
            elif isinstance(config, dict):
                config_dict = config
            else:
                return await original_method(instance, *args, **kwargs)

            # Get current configuration space from context to determine what to override
            from traigent.config.context import get_config_space

            config_space = get_config_space()

            # Override parameters based on mapping and supported params
            overridden_kwargs = kwargs.copy()
            overrides_applied = []

            for traigent_param, framework_param in parameter_mapping.items():
                if traigent_param in config_dict and traigent_param in supported_params:
                    # Only override if parameter is in configuration space (being optimized)
                    # or if no configuration space is set (not in optimization)
                    if config_space is None or traigent_param in config_space:
                        override_value = config_dict[traigent_param]
                        original_value = overridden_kwargs.get(
                            framework_param, "not_set"
                        )
                        overridden_kwargs[framework_param] = override_value
                        overrides_applied.append(
                            f"{framework_param}: {original_value} -> {override_value}"
                        )

            # Record what overrides were applied
            # (logging handled separately if needed)

            # Call original method with overridden parameters
            return await original_method(instance, *args, **overridden_kwargs)

        # Return async version if original is async
        import inspect

        if inspect.iscoroutinefunction(original_method):
            return async_override_method
        else:
            return override_method

    def _apply_method_override(
        self, target_class: type, class_name: str, method_path: str
    ) -> None:
        """Apply method override to a specific method path.

        Args:
            target_class: The class to override
            class_name: Name of the class
            method_path: Dot-separated path to the method (e.g., "messages.create")
        """
        try:
            # Navigate to the method location
            current_obj = target_class
            path_parts = method_path.split(".")

            # Navigate to the parent object containing the method
            for part in path_parts[:-1]:
                current_obj = getattr(current_obj, part)

            method_name = path_parts[-1]
            original_method = getattr(current_obj, method_name)

            # Store original method
            original_key = f"_traigent_original_{method_path.replace('.', '_')}"
            if not hasattr(current_obj, original_key):
                setattr(current_obj, original_key, original_method)

            # Create and apply override
            override_method = self._create_override_method(
                original_method, class_name, method_path
            )
            setattr(current_obj, method_name, override_method)

            # Track for cleanup
            method_key = f"{class_name}.{method_path}"
            self._original_methods[method_key] = (
                current_obj,
                method_name,
                original_method,
            )

        except (AttributeError, TypeError):
            # Method not found or not accessible, skip silently
            logger.debug(
                "Skipping override for %s because %s.%s is unavailable",
                method_path,
                class_name,
                method_path,
            )

    def activate_overrides(self, framework_targets: list[str]) -> None:
        """Activate framework overrides for specified targets.

        Args:
            framework_targets: List of framework class names to override
        """
        self._override_active.enabled = True

        for target in framework_targets:
            if self.is_override_registered(target):
                continue  # Already overridden

            # Try to find and override the target class
            override_applied = False

            # Handle built-in mock classes
            if target in ["MockOpenAI", "MockLangChainOpenAI"]:
                # These are handled at the module level when demo is run
                override_applied = True

            # Try to import and override real framework classes
            else:
                try:
                    module_path, class_name = target.rsplit(".", 1)
                    module = __import__(module_path, fromlist=[class_name])
                    target_class = getattr(module, class_name)

                    # Store original constructor
                    if not hasattr(target_class, "_traigent_original_init"):
                        target_class._traigent_original_init = target_class.__init__

                    # Apply constructor override
                    override_constructor = self._create_override_constructor(
                        target_class._traigent_original_init, target
                    )
                    target_class.__init__ = override_constructor

                    # Apply method overrides
                    method_mappings = self._method_mappings.get(target, {})
                    for method_path in method_mappings.keys():
                        self._apply_method_override(target_class, target, method_path)

                    self.register_active_override(target, target_class)
                    override_applied = True

                except (ImportError, AttributeError) as e:
                    # Handle specific compatibility errors
                    error_msg = str(e)
                    if (
                        "PydanticUserError" in error_msg
                        or "__modify_schema__" in error_msg
                    ):
                        logger.warning(
                            f"Pydantic compatibility issue with {target}: {error_msg}"
                        )
                        logger.info(
                            f"Consider upgrading {target.split('.')[0]} package for Pydantic v2 compatibility"
                        )
                    else:
                        logger.debug(f"Framework {target} not available: {error_msg}")
                    # Framework not available, skip silently
                except Exception as e:
                    # Catch any other errors (including PydanticUserError)
                    error_msg = str(e)
                    error_type = type(e).__name__
                    if (
                        "PydanticUserError" in error_type
                        or "pydantic" in error_msg.lower()
                    ):
                        logger.warning(
                            f"Pydantic compatibility issue with {target}: {error_type}: {error_msg}"
                        )
                        logger.info(
                            f"Skipping {target} due to Pydantic v2 compatibility issues. Consider upgrading the package."
                        )
                    else:
                        logger.debug(
                            f"Unexpected error loading {target}: {error_type}: {error_msg}"
                        )
                    # Skip silently to avoid breaking optimization

            if not override_applied:
                logger.warning(f"Could not override framework target: {target}")

    def deactivate_overrides(self) -> None:
        """Deactivate all framework overrides and restore original constructors and methods."""
        self._override_active.enabled = False

        # Restore original constructors
        for _target, target_class in self.get_active_overrides_copy().items():
            if hasattr(target_class, "_traigent_original_init"):
                target_class.__init__ = target_class._traigent_original_init

        # Restore original methods
        for _method_key, (
            obj,
            method_name,
            original_method,
        ) in self._original_methods.items():
            try:
                setattr(obj, method_name, original_method)
            except (AttributeError, TypeError):
                # Ignore if object no longer exists or method can't be restored
                logger.debug(
                    "Could not restore method %s on %s during deactivation",
                    method_name,
                    obj,
                )

        self.clear_active_overrides()
        self._original_methods.clear()

    @contextmanager
    def override_context(
        self,
        framework_key: str | list[str] | None = None,
        config: TraigentConfig | None = None,
    ) -> Generator[None, None, None]:
        """Context manager for temporary framework overrides.

        Args:
            framework_key: List of framework class names to override (or single name)
            config: Optional config (ignored in this implementation but kept for compatibility)

        Usage:
            with manager.override_context(['openai.OpenAI']):
                # Framework calls will be automatically overridden
                llm = OpenAI(model="gpt-4o-mini")  # Parameters auto-injected
        """
        targets: list[str] = []
        if isinstance(framework_key, list):
            targets = framework_key
        elif isinstance(framework_key, str):
            targets = [framework_key]

        if targets:
            self.activate_overrides(targets)

        try:
            yield
        finally:
            if targets:
                self.deactivate_overrides()

    def override_mock_classes(self, mock_classes: dict[str, type]) -> None:
        """Apply overrides to mock classes directly (for demos/testing).

        Args:
            mock_classes: Dictionary mapping class names to mock class objects
        """
        for class_name, mock_class in mock_classes.items():
            if class_name in self._parameter_mappings:
                # Store original if not already stored
                # Cast to Any to allow setting arbitrary attributes and accessing __init__
                mock_cls_any = cast(Any, mock_class)
                if not hasattr(mock_cls_any, "_traigent_original_init"):
                    mock_cls_any._traigent_original_init = mock_cls_any.__init__

                # Apply override
                override_constructor = self._create_override_constructor(
                    mock_cls_any._traigent_original_init, class_name
                )
                mock_cls_any.__init__ = override_constructor
                self.register_active_override(class_name, mock_class)

    def create_intelligent_override(
        self, target_class: type, method_name: str | None = None
    ) -> Callable[..., Any]:
        """Create override with automatic parameter discovery and validation.

        This method provides enhanced override capabilities when the enhanced features
        are available, otherwise falls back to standard override creation.

        Args:
            target_class: Target class to override
            method_name: Optional method name (None for constructor)

        Returns:
            Override wrapper function
        """
        if not self._use_enhanced_features:
            # Fallback to standard override
            class_name = f"{target_class.__module__}.{target_class.__name__}"
            if method_name:
                return self._create_override_method(
                    getattr(target_class, method_name), class_name, method_name
                )
            else:
                return self._create_override_constructor(
                    cast(Callable[..., Any], target_class.__init__),  # type: ignore[misc]
                    class_name,
                )

        # Enhanced override with discovery and validation
        class_name = f"{target_class.__module__}.{target_class.__name__}"

        # Check for existing or discover new parameter mapping
        if class_name not in self._discovered_mappings:
            package = target_class.__module__.split(".")[0]
            version = (
                self.version_manager.get_package_version(package)
                if self.version_manager is not None
                else None
            )
            self._discovered_mappings[class_name] = self._discover_mapping(
                target_class, version
            )

        parameter_mapping = self._discovered_mappings[class_name]

        # Get the method to override
        if method_name:
            original_method = self._get_method(target_class, method_name)
        else:
            original_method = cast(
                Callable[..., Any], target_class.__init__  # type: ignore[misc]
            )

        # Create the intelligent wrapper
        return self._create_resilient_override(
            original_method, class_name, parameter_mapping
        )

    def _discover_mapping(
        self, target_class: type, version: str | None
    ) -> dict[str, str]:
        """Discover parameter mapping for a class using enhanced discovery.

        Args:
            target_class: Target class
            version: Package version

        Returns:
            Parameter mapping dictionary
        """
        if not self._use_enhanced_features:
            # Return empty mapping if enhanced features not available
            return {}

        package = target_class.__module__.split(".")[0]

        # Try version-specific mapping first
        if version and self.version_manager is not None:
            version_mapping = self.version_manager.get_compatible_mapping(
                package, version
            )
            if version_mapping:
                return version_mapping

        # Discover parameters dynamically
        if self.discovery is not None:
            params = self.discovery.discover_init_parameters(target_class)
            param_names = list(params.keys())

            # Create mapping using universal patterns
            universal_mapping = self.discovery.create_universal_mapping()
            discovered_mapping = {}

            for traigent_param, variations in universal_mapping.items():
                for variation in variations:
                    if variation in param_names:
                        discovered_mapping[traigent_param] = variation
                        break

            return discovered_mapping

        return {}

    def _get_method(self, target_class: type, method_path: str) -> Callable[..., Any]:
        """Get method from class using dot-separated path.

        Args:
            target_class: The class containing the method
            method_path: Dot-separated path to the method

        Returns:
            The method callable
        """
        current_obj = target_class
        parts = method_path.split(".")

        for part in parts[:-1]:
            current_obj = getattr(current_obj, part)

        return cast(Callable[..., Any], getattr(current_obj, parts[-1]))

    def _create_resilient_override(
        self,
        original_method: Callable[..., Any],
        class_name: str,
        parameter_mapping: dict[str, str],
    ) -> Callable[..., Any]:
        """Create a resilient override with multiple fallback strategies.

        Args:
            original_method: The original method to wrap
            class_name: Name of the class being overridden
            parameter_mapping: Parameter mapping dictionary

        Returns:
            Resilient override wrapper
        """
        override_active = self._override_active  # Capture in closure

        @functools.wraps(original_method)
        def resilient_wrapper(*args, **kwargs):
            # Check if we're in an optimization context
            if not getattr(override_active, "enabled", False):
                return original_method(*args, **kwargs)

            # Get current Traigent configuration
            config = get_config()
            if not config:
                return original_method(*args, **kwargs)

            # Extract configuration values
            config_dict = self._extract_config_dict(config)
            if not config_dict:
                return original_method(*args, **kwargs)

            # Apply overrides with fallback strategies
            overridden_kwargs = kwargs.copy()

            # Strategy 1: Direct mapping
            for traigent_param, framework_param in parameter_mapping.items():
                if traigent_param in config_dict:
                    overridden_kwargs[framework_param] = config_dict[traigent_param]

            # Strategy 2: Try exact parameter names if not already mapped
            for param_name, param_value in config_dict.items():
                if param_name not in overridden_kwargs:
                    overridden_kwargs[param_name] = param_value

            return original_method(*args, **overridden_kwargs)

        @functools.wraps(original_method)
        async def async_resilient_wrapper(*args, **kwargs):
            # Async version of the resilient wrapper
            if not getattr(override_active, "enabled", False):
                return await original_method(*args, **kwargs)

            config = get_config()
            if not config:
                return await original_method(*args, **kwargs)

            config_dict = self._extract_config_dict(config)
            if not config_dict:
                return await original_method(*args, **kwargs)

            overridden_kwargs = kwargs.copy()

            for traigent_param, framework_param in parameter_mapping.items():
                if traigent_param in config_dict:
                    overridden_kwargs[framework_param] = config_dict[traigent_param]

            for param_name, param_value in config_dict.items():
                if param_name not in overridden_kwargs:
                    overridden_kwargs[param_name] = param_value

            return await original_method(*args, **overridden_kwargs)

        # Return async version if original is async
        if inspect.iscoroutinefunction(original_method):
            return async_resilient_wrapper
        else:
            return resilient_wrapper

    def _extract_config_dict(
        self, config: TraigentConfig | dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract configuration dictionary from TraigentConfig or dict.

        Args:
            config: Configuration object

        Returns:
            Configuration dictionary or None
        """
        if isinstance(config, TraigentConfig):
            config_dict = cast(dict[Any, Any], config.to_dict())
            config_dict.update(config.custom_params)
            return config_dict
        elif isinstance(config, dict):
            return config
        return None


# Global instance
_framework_override_manager = FrameworkOverrideManager()


def enable_framework_overrides(
    framework_targets: list[str],
) -> FrameworkOverrideManager:
    """Enable automatic framework parameter overrides.

    Args:
        framework_targets: List of framework class names to override

    Returns:
        The framework override manager
    """
    _framework_override_manager.activate_overrides(framework_targets)
    return _framework_override_manager


def disable_framework_overrides() -> None:
    """Disable automatic framework parameter overrides."""
    _framework_override_manager.deactivate_overrides()


def override_context(framework_targets: list[str]) -> AbstractContextManager[None]:
    """Context manager for temporary framework overrides.

    Args:
        framework_targets: List of framework class names to override

    Returns:
        Context manager that activates overrides temporarily
    """
    return _framework_override_manager.override_context(framework_targets)


def register_framework_mapping(
    target_class: str | type, parameter_mapping: dict[str, str]
) -> None:
    """Register a custom parameter mapping for a framework class.

    Args:
        target_class: Framework class name or class object
        parameter_mapping: Mapping from Traigent parameters to framework parameters
    """
    _framework_override_manager.register_framework_target(
        target_class, parameter_mapping
    )


def apply_mock_overrides(mock_classes: dict[str, type]) -> None:
    """Apply framework overrides to mock classes (for demos/testing).

    Args:
        mock_classes: Dictionary mapping class names to mock class objects
    """
    _framework_override_manager.override_mock_classes(mock_classes)


# Convenience functions for common frameworks
def override_openai_sdk() -> None:
    """Enable overrides for OpenAI SDK classes."""
    enable_framework_overrides(["openai.OpenAI", "openai.AsyncOpenAI"])


def override_langchain() -> None:
    """Enable overrides for LangChain classes."""
    enable_framework_overrides(
        [
            "langchain.llms.OpenAI",
            "langchain_openai.OpenAI",
            "langchain_openai.ChatOpenAI",
        ]
    )


def override_anthropic() -> None:
    """Enable overrides for Anthropic classes."""
    enable_framework_overrides(
        [
            "anthropic.Anthropic",
            "anthropic.AsyncAnthropic",
            "langchain_anthropic.ChatAnthropic",
        ]
    )


def override_cohere() -> None:
    """Enable overrides for Cohere classes."""
    enable_framework_overrides(["cohere.Client", "cohere.AsyncClient"])


def override_huggingface() -> None:
    """Enable overrides for HuggingFace classes."""
    enable_framework_overrides(
        ["transformers.pipeline", "transformers.AutoModelForCausalLM"]
    )


def override_all_platforms() -> None:
    """Enable overrides for all supported platforms."""
    enable_framework_overrides(
        [
            # OpenAI
            "openai.OpenAI",
            "openai.AsyncOpenAI",
            # LangChain OpenAI
            "langchain.llms.OpenAI",
            "langchain_openai.OpenAI",
            "langchain_openai.ChatOpenAI",
            # Anthropic
            "anthropic.Anthropic",
            "anthropic.AsyncAnthropic",
            "langchain_anthropic.ChatAnthropic",
            # Cohere
            "cohere.Client",
            "cohere.AsyncClient",
            # HuggingFace
            "transformers.pipeline",
            "transformers.AutoModelForCausalLM",
        ]
    )


# Enhanced features convenience functions
def enable_intelligent_overrides(targets: str | list[str]) -> None:
    """Enable intelligent framework overrides with auto-discovery.

    This function provides the same interface as the v2 version but uses
    the consolidated implementation.

    Args:
        targets: Framework name(s) or class name(s) to override
    """
    if isinstance(targets, str):
        targets = [targets]

    # Check if enhanced features are available
    if ENHANCED_FEATURES_AVAILABLE:
        # Use enhanced auto-discovery for package patterns
        package_patterns = []
        class_names = []

        for target in targets:
            if "." in target and target.count(".") >= 2:
                # Likely a specific class name
                class_names.append(target)
            else:
                # Package pattern
                package_patterns.append(target)

        # Auto-discover packages if supported
        if package_patterns and hasattr(_framework_override_manager, "discovery"):
            for pattern in package_patterns:
                # Use discovery to find classes
                logger.info(f"Auto-discovering classes in {pattern}")
                # Note: Full auto-discovery requires additional implementation
                logger.debug("Discovery is not yet implemented for pattern %s", pattern)

        # Override specific classes
        enable_framework_overrides(class_names)
    else:
        # Fallback to standard override
        enable_framework_overrides(targets)


# Alias for backward compatibility
def enable_enhanced_overrides(targets: str | list[str]):
    """Alias for enable_intelligent_overrides for backward compatibility."""
    return enable_intelligent_overrides(targets)
