"""Base Framework Override System.

This module provides the core functionality for automatic framework parameter override,
refactored from framework_override.py with enhanced capabilities.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Maintainability FUNC-INTEGRATIONS FUNC-INVOKERS REQ-INT-008 REQ-INJ-002 SYNC-IntegrationHook

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from typing import Any, Callable, cast

from ..config.context import get_config
from ..config.types import TraigentConfig
from ..utils.logging import get_logger
from .activation import ActivationState

logger = get_logger(__name__)


class BaseOverrideManager:
    """Enhanced base manager for framework parameter overrides with dynamic discovery.

    Uses ActivationState for thread-safe state management of original methods,
    constructors, and active overrides.
    """

    def __init__(self) -> None:
        """Initialize the base override manager."""
        # Use ActivationState for thread-safe state management
        self._state = ActivationState()

        # Keep backward-compatible references (delegate to _state)
        self._override_active = self._state._override_active

    # Backward-compatible property accessors (delegate to _state)
    @property
    def _original_constructors(self) -> dict[str, Any]:
        """Backward-compatible access to original constructors."""
        return self._state.original_constructors

    @property
    def _original_methods(self) -> dict[str, Any]:
        """Backward-compatible access to original methods."""
        return self._state.original_methods

    @property
    def _active_overrides(self) -> dict[str, Any]:
        """Backward-compatible access to active overrides."""
        return self._state.get_active_overrides()

    def is_override_active(self) -> bool:
        """Check if override is currently active."""
        return self._state.is_active()

    def set_override_active(self, active: bool) -> None:
        """Set override active state."""
        self._state.set_active(active)

    def extract_config_dict(
        self, config: TraigentConfig | dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Extract configuration dictionary from TraigentConfig or dict."""
        if config is None:
            return None

        if isinstance(config, TraigentConfig):
            config_dict: dict[str, Any] = config.to_dict()
            config_dict.update(config.custom_params)
            return config_dict
        elif isinstance(config, dict):
            return config
        else:
            return None

    def get_method_wrapper(
        self, original_method: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Determine the appropriate wrapper based on method type."""
        if inspect.iscoroutinefunction(original_method):
            return self.create_async_wrapper
        else:
            return self.create_sync_wrapper

    def create_sync_wrapper(
        self, original_method: Callable[..., Any], apply_overrides: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create a synchronous method wrapper."""

        @functools.wraps(original_method)
        def sync_wrapper(*args, **kwargs):
            if not self.is_override_active():
                return original_method(*args, **kwargs)

            config = get_config()
            config_dict = self.extract_config_dict(config)
            if not config_dict:
                return original_method(*args, **kwargs)

            # Apply overrides
            overridden_kwargs = apply_overrides(config_dict, kwargs)
            return original_method(*args, **overridden_kwargs)

        return sync_wrapper

    def create_async_wrapper(
        self, original_method: Callable[..., Any], apply_overrides: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create an asynchronous method wrapper."""

        @functools.wraps(original_method)
        async def async_wrapper(*args, **kwargs):
            if not self.is_override_active():
                return await original_method(*args, **kwargs)

            config = get_config()
            config_dict = self.extract_config_dict(config)
            if not config_dict:
                return await original_method(*args, **kwargs)

            # Apply overrides
            overridden_kwargs = apply_overrides(config_dict, kwargs)
            return await original_method(*args, **overridden_kwargs)

        return async_wrapper

    def store_original(
        self, target_class: type, attr_name: str, original: Any, storage_key: str
    ) -> None:
        """Store original method/constructor for later restoration.

        Thread-safe: Delegates to ActivationState.
        """
        full_key = f"{target_class.__module__}.{target_class.__name__}.{attr_name}"
        if storage_key == "constructor":
            self._state.store_original_constructor(
                full_key, (target_class, attr_name, original)
            )
        else:
            self._state.store_original_method(
                full_key, original, target=target_class, attribute=attr_name
            )

    def restore_originals(self) -> None:
        """Restore all original methods and constructors.

        Thread-safe: Delegates to ActivationState with lock protection.
        """
        # Get copies and restore constructors
        for stored in self._state.original_constructors.values():
            try:
                if isinstance(stored, tuple) and len(stored) == 3:
                    target_class, attr_name, original = stored
                    setattr(target_class, attr_name, original)
            except Exception as e:
                logger.debug(f"Could not restore {stored}: {e}")

        # Get copies and restore methods
        for stored in self._state.original_methods.values():
            target_obj: Any | None = None
            method_name: str | None = None
            original = stored

            if isinstance(stored, tuple) and len(stored) == 3:
                target_obj, method_name, original = stored

            if target_obj is not None and method_name:
                try:
                    setattr(target_obj, method_name, original)
                except Exception as e:
                    logger.debug(f"Could not restore {method_name}: {e}")

        # Clear state through the state manager
        self._state.clear_all()

    def merge_parameters(
        self, user_params: dict[str, Any] | None, config_params: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merge user parameters with config parameters, user takes precedence."""
        if user_params is None:
            user_params = {}
        if config_params is None:
            config_params = {}

        # Start with config parameters
        merged = config_params.copy()

        # Override with user parameters (user takes precedence)
        merged.update(user_params)

        return merged

    def is_constructor_overridden(self, framework_key: str) -> bool:
        """Check if constructor is overridden for given framework."""
        return self._state.has_original_constructor(framework_key)

    def store_original_constructor(
        self, framework_key: str, original_constructor: Callable[..., Any]
    ) -> None:
        """Store original constructor for restoration."""
        self._state.store_original_constructor(framework_key, original_constructor)

    def restore_original_constructor(
        self, framework_key: str
    ) -> Callable[..., Any] | None:
        """Restore and return original constructor."""
        return self._state.remove_original_constructor(framework_key)

    def create_overridden_constructor(
        self, framework_key: str, original_constructor: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create overridden constructor wrapper."""

        @functools.wraps(original_constructor)
        def wrapper(*args, **kwargs):
            if not self.is_override_active():
                return original_constructor(*args, **kwargs)

            config = get_config()
            config_dict = self.extract_config_dict(config)
            if not config_dict:
                return original_constructor(*args, **kwargs)

            # Merge config into kwargs (user params take precedence)
            merged_kwargs = self.merge_parameters(kwargs, config_dict)
            return original_constructor(*args, **merged_kwargs)

        return wrapper

    def is_method_overridden(self, method_key: str) -> bool:
        """Check if method is overridden."""
        return self._state.has_original_method(method_key)

    def store_original_method(
        self,
        method_key: str,
        original_method: Callable[..., Any],
        *,
        target: Any | None = None,
        attribute: str | None = None,
    ) -> None:
        """Store original method for restoration."""
        self._state.store_original_method(
            method_key, original_method, target=target, attribute=attribute
        )

    def restore_original_method(self, method_key: str) -> Callable[..., Any] | None:
        """Restore and return original method."""
        stored = self._state.remove_original_method(method_key)
        if stored is None:
            return None

        target_obj: Any | None = None
        method_name: str | None = None
        original: Any = stored

        if isinstance(stored, tuple) and len(stored) == 3:
            target_obj, method_name, original = stored

        if target_obj is not None and method_name:
            try:
                setattr(target_obj, method_name, original)
            except Exception as e:
                logger.debug(f"Could not restore {method_name}: {e}")

        return cast(Callable[..., Any] | None, original)

    def create_overridden_method(
        self, method_key: str, original_method: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Create overridden method wrapper."""

        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            if not self.is_override_active():
                return original_method(*args, **kwargs)

            config = get_config()
            config_dict = self.extract_config_dict(config)
            if not config_dict:
                return original_method(*args, **kwargs)

            # Merge config into kwargs (user params take precedence)
            merged_kwargs = self.merge_parameters(kwargs, config_dict)
            return original_method(*args, **merged_kwargs)

        return wrapper

    def start_override_context(self, framework_key: str) -> None:
        """Start override context for framework."""
        self.set_override_active(True)
        self._state.register_active_override(framework_key, True)

    def end_override_context(self, framework_key: str) -> None:
        """End override context for framework."""
        # Unregister the framework override using the public method
        self.unregister_active_override(framework_key)

        # If no active overrides, deactivate
        if not self._state.get_active_overrides():
            self.set_override_active(False)

    def cleanup_override(self, framework_key: str) -> None:
        """Cleanup specific framework override."""
        # Remove constructor if exists
        self._state.remove_original_constructor(framework_key)

        # Remove related methods (get keys first, then remove)
        method_keys = list(self._state.original_methods.keys())
        for method_key in method_keys:
            if method_key.startswith(framework_key + "."):
                self._state.remove_original_method(method_key)

        # Remove from active overrides using the public method
        self.unregister_active_override(framework_key)

    def cleanup_all_overrides(self) -> None:
        """Cleanup all overrides."""
        self._state.clear_all()

    # Public methods for active override management (use instead of _active_overrides property)
    def register_active_override(self, target_key: str, target_class: Any) -> None:
        """Register an active override.

        Thread-safe: Delegates to ActivationState.

        Args:
            target_key: Key identifying the override target
            target_class: The class that was overridden
        """
        self._state.register_active_override(target_key, target_class)

    def is_override_registered(self, target_key: str) -> bool:
        """Check if a target is already overridden.

        Thread-safe: Delegates to ActivationState.

        Args:
            target_key: Key to check

        Returns:
            True if the target has an active override
        """
        return self._state.is_override_registered(target_key)

    def unregister_active_override(self, target_key: str) -> None:
        """Remove an active override registration.

        Thread-safe: Uses ActivationState lock.

        Args:
            target_key: Key of override to remove
        """
        with self._state._lock:
            if target_key in self._state._active_overrides:
                del self._state._active_overrides[target_key]

    def get_active_overrides_copy(self) -> dict[str, Any]:
        """Get a thread-safe copy of all active overrides.

        Returns:
            Copy of the active overrides dictionary
        """
        return self._state.get_active_overrides()

    def clear_active_overrides(self) -> dict[str, Any]:
        """Clear and return all active overrides.

        Thread-safe: Delegates to ActivationState.

        Returns:
            The cleared active overrides dictionary
        """
        return self._state.clear_active_overrides()

    @contextmanager
    def override_context(
        self, framework_key: str | None = None, config: TraigentConfig | None = None
    ):
        """Context manager for temporary override activation."""
        if framework_key:
            self.start_override_context(framework_key)
        else:
            self.set_override_active(True)

        try:
            yield
        finally:
            if framework_key:
                self.end_override_context(framework_key)
            else:
                self.set_override_active(False)
            self.restore_originals()
