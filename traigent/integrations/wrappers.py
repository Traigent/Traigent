"""Wrapper factory for framework override injection.

This module provides the Protocol-based interface and helper functions
for creating constructor and method wrappers used in framework overrides.

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from ..config.types import TraigentConfig


class OverrideContext(Protocol):
    """Protocol defining what wrapper factories need from the override manager.

    Using Protocol avoids circular import with FrameworkOverrideManager.
    Any class implementing these methods can be used as a context.
    """

    def is_override_active(self) -> bool:
        """Check if overrides are currently active.

        Returns:
            True if overrides should be applied
        """
        ...

    def get_parameter_mapping(self, class_name: str) -> dict[str, str]:
        """Get parameter mapping for a class.

        Args:
            class_name: Fully-qualified class name

        Returns:
            Mapping from traigent parameter names to framework parameter names
        """
        ...

    def get_method_params(self, class_name: str, method_name: str) -> list[str]:
        """Get supported parameters for a method.

        Args:
            class_name: Fully-qualified class name
            method_name: Method name

        Returns:
            List of parameter names the method accepts
        """
        ...

    def extract_config_dict(
        self, config: TraigentConfig | dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Extract configuration dictionary.

        Args:
            config: Configuration object

        Returns:
            Configuration dictionary or None
        """
        ...


def apply_parameter_overrides(
    kwargs: dict[str, Any],
    config_dict: dict[str, Any],
    parameter_mapping: dict[str, str],
    supported_params: list[str] | None = None,
    config_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply parameter overrides to kwargs based on mapping.

    Args:
        kwargs: Original keyword arguments
        config_dict: Configuration values
        parameter_mapping: Mapping from traigent params to framework params
        supported_params: If provided, only override params in this list
        config_space: If provided, only override params in this space

    Returns:
        Modified kwargs with overrides applied
    """
    overridden_kwargs = kwargs.copy()

    for traigent_param, framework_param in parameter_mapping.items():
        if traigent_param not in config_dict:
            continue

        # Check if param is supported (for method-level filtering)
        if supported_params is not None and traigent_param not in supported_params:
            continue

        # Check if param is in config space (for optimization filtering)
        if config_space is not None and traigent_param not in config_space:
            continue

        overridden_kwargs[framework_param] = config_dict[traigent_param]

    return overridden_kwargs


def create_wrapper(
    original: Callable[..., Any],
    is_active_check: Callable[[], bool],
    get_config: Callable[[], Any],
    extract_config: Callable[[Any], dict[str, Any] | None],
    apply_overrides: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[..., Any]:
    """Create a wrapper function that applies overrides.

    This is a generic wrapper factory that delegates the actual override
    logic to the provided functions.

    Args:
        original: Original function to wrap
        is_active_check: Function to check if overrides are active
        get_config: Function to get current config
        extract_config: Function to extract config dict
        apply_overrides: Function to apply overrides to kwargs

    Returns:
        Wrapped function
    """

    @functools.wraps(original)
    def sync_wrapper(*args, **kwargs):
        if not is_active_check():
            return original(*args, **kwargs)

        config = get_config()
        config_dict = extract_config(config)
        if not config_dict:
            return original(*args, **kwargs)

        overridden_kwargs = apply_overrides(kwargs)
        return original(*args, **overridden_kwargs)

    @functools.wraps(original)
    async def async_wrapper(*args, **kwargs):
        if not is_active_check():
            return await original(*args, **kwargs)

        config = get_config()
        config_dict = extract_config(config)
        if not config_dict:
            return await original(*args, **kwargs)

        overridden_kwargs = apply_overrides(kwargs)
        return await original(*args, **overridden_kwargs)

    if inspect.iscoroutinefunction(original):
        return async_wrapper
    return sync_wrapper


def create_method_wrapper(
    original_method: Callable[..., Any],
    is_active_check: Callable[[], bool],
    get_config: Callable[[], Any],
    extract_config: Callable[[Any], dict[str, Any] | None],
    apply_overrides: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[..., Any]:
    """Create a wrapper for instance methods that applies overrides.

    Similar to create_wrapper but expects 'self'/'instance' as first arg.

    Args:
        original_method: Original method to wrap
        is_active_check: Function to check if overrides are active
        get_config: Function to get current config
        extract_config: Function to extract config dict
        apply_overrides: Function to apply overrides to kwargs

    Returns:
        Wrapped method
    """

    @functools.wraps(original_method)
    def sync_wrapper(instance, *args, **kwargs):
        if not is_active_check():
            return original_method(instance, *args, **kwargs)

        config = get_config()
        config_dict = extract_config(config)
        if not config_dict:
            return original_method(instance, *args, **kwargs)

        overridden_kwargs = apply_overrides(kwargs)
        return original_method(instance, *args, **overridden_kwargs)

    @functools.wraps(original_method)
    async def async_wrapper(instance, *args, **kwargs):
        if not is_active_check():
            return await original_method(instance, *args, **kwargs)

        config = get_config()
        config_dict = extract_config(config)
        if not config_dict:
            return await original_method(instance, *args, **kwargs)

        overridden_kwargs = apply_overrides(kwargs)
        return await original_method(instance, *args, **overridden_kwargs)

    if inspect.iscoroutinefunction(original_method):
        return async_wrapper
    return sync_wrapper


def create_resilient_wrapper(
    original: Callable[..., Any],
    wrapper: Callable[..., Any],
    fallback_on_error: bool = True,
) -> Callable[..., Any]:
    """Create a resilient wrapper that falls back to original on error.

    Args:
        original: Original function
        wrapper: Wrapper function that might fail
        fallback_on_error: Whether to fall back to original on wrapper error

    Returns:
        Resilient wrapper function
    """
    if not fallback_on_error:
        return wrapper

    @functools.wraps(original)
    def resilient_sync_wrapper(*args, **kwargs):
        try:
            return wrapper(*args, **kwargs)
        except Exception:
            # Fallback to original on any wrapper error
            return original(*args, **kwargs)

    @functools.wraps(original)
    async def resilient_async_wrapper(*args, **kwargs):
        try:
            return await wrapper(*args, **kwargs)
        except Exception:
            # Fallback to original on any wrapper error
            return await original(*args, **kwargs)

    if inspect.iscoroutinefunction(original):
        return resilient_async_wrapper
    return resilient_sync_wrapper
