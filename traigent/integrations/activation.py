"""Framework override activation and deactivation management.

This module provides thread-safe activation state management for framework overrides.
The ActivationState class manages the override lifecycle including storing/restoring
original methods and constructors.

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class _ContextLocalFlag:
    """Bool flag with task-local and thread-local isolation."""

    def __init__(self) -> None:
        self._enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
            "traigent_override_active",
            default=False,
        )

    @property
    def enabled(self) -> bool:
        """Return the active state for the current execution context."""
        return self._enabled.get()

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set the active state for the current execution context."""
        self._enabled.set(bool(value))


class ActivationState:
    """Thread-safe state container for framework override activation.

    Manages enabling/disabling of framework overrides with proper synchronization.
    Uses a context-local flag for the active state and an RLock for protecting
    shared state (original methods, active overrides).

    Thread Safety:
        - _override_active: task-local and thread-local active state
        - _original_methods, _active_overrides: Protected by RLock
        - All access to shared state goes through lock-protected methods
    """

    def __init__(self) -> None:
        """Initialize the activation state manager."""
        # Context-local active flag - isolated per thread and asyncio task.
        self._override_active = _ContextLocalFlag()

        # Shared state protected by lock
        self._original_methods: dict[str, tuple[Any, str | None, Any]] = {}
        self._original_constructors: dict[str, Any] = {}
        self._active_overrides: dict[str, Any] = {}

        # RLock allows re-entrant locking (same thread can acquire multiple times)
        self._lock = threading.RLock()

    def is_active(self) -> bool:
        """Check if override is currently active for this execution context.

        Thread-safe: Uses context-local storage.

        Returns:
            True if overrides are active in the current thread/task.
        """
        return self._override_active.enabled

    def set_active(self, active: bool) -> None:
        """Set override active state for this execution context.

        Thread-safe: Uses context-local storage.

        Args:
            active: Whether overrides should be active.
        """
        self._override_active.enabled = active

    def store_original_method(
        self,
        method_key: str,
        original_method: Callable[..., Any],
        *,
        target: Any | None = None,
        attribute: str | None = None,
    ) -> None:
        """Store original method for later restoration.

        Thread-safe: Protected by RLock.

        Args:
            method_key: Unique key identifying the method (e.g., "openai.OpenAI.chat.completions.create")
            original_method: The original method callable
            target: Optional object the method belongs to (for restoration)
            attribute: Optional attribute name on the target (for restoration)
        """
        with self._lock:
            self._original_methods[method_key] = (target, attribute, original_method)

    def get_original_method(
        self, method_key: str
    ) -> tuple[Any, str | None, Any] | None:
        """Get stored original method.

        Thread-safe: Protected by RLock.

        Args:
            method_key: Key of the method to retrieve

        Returns:
            Tuple of (target, attribute, original_method) or None if not found
        """
        with self._lock:
            return self._original_methods.get(method_key)

    def has_original_method(self, method_key: str) -> bool:
        """Check if an original method is stored.

        Thread-safe: Protected by RLock.

        Args:
            method_key: Key to check

        Returns:
            True if the method is stored
        """
        with self._lock:
            return method_key in self._original_methods

    def remove_original_method(
        self, method_key: str
    ) -> tuple[Any, str | None, Any] | None:
        """Remove and return stored original method.

        Thread-safe: Protected by RLock.

        Args:
            method_key: Key of the method to remove

        Returns:
            Tuple of (target, attribute, original_method) or None if not found
        """
        with self._lock:
            return self._original_methods.pop(method_key, None)

    def store_original_constructor(
        self, class_key: str, original_constructor: Any
    ) -> None:
        """Store original constructor for later restoration.

        Thread-safe: Protected by RLock.

        Args:
            class_key: Unique key identifying the class (e.g., "openai.OpenAI")
            original_constructor: The original __init__ method or tuple with metadata
        """
        with self._lock:
            self._original_constructors[class_key] = original_constructor

    def get_original_constructor(self, class_key: str) -> Callable[..., Any] | None:
        """Get stored original constructor.

        Thread-safe: Protected by RLock.

        Args:
            class_key: Key of the class to retrieve

        Returns:
            Original constructor or None if not found
        """
        with self._lock:
            return self._original_constructors.get(class_key)

    def has_original_constructor(self, class_key: str) -> bool:
        """Check if an original constructor is stored.

        Thread-safe: Protected by RLock.

        Args:
            class_key: Key to check

        Returns:
            True if the constructor is stored
        """
        with self._lock:
            return class_key in self._original_constructors

    def remove_original_constructor(self, class_key: str) -> Callable[..., Any] | None:
        """Remove and return stored original constructor.

        Thread-safe: Protected by RLock.

        Args:
            class_key: Key of the class to remove

        Returns:
            Original constructor or None if not found
        """
        with self._lock:
            result: Callable[..., Any] | None = self._original_constructors.pop(
                class_key, None
            )
            return result

    def register_active_override(self, target_key: str, target_class: Any) -> None:
        """Register an active override.

        Thread-safe: Protected by RLock.

        Args:
            target_key: Key identifying the override target (e.g., "openai.OpenAI")
            target_class: The class that was overridden
        """
        with self._lock:
            self._active_overrides[target_key] = target_class

    def is_override_registered(self, target_key: str) -> bool:
        """Check if a target is already overridden.

        Thread-safe: Protected by RLock.

        Args:
            target_key: Key to check

        Returns:
            True if the target has an active override
        """
        with self._lock:
            return target_key in self._active_overrides

    def get_active_overrides(self) -> dict[str, Any]:
        """Get a copy of all active overrides.

        Thread-safe: Protected by RLock.

        Returns:
            Copy of the active overrides dictionary
        """
        with self._lock:
            return dict(self._active_overrides)

    def clear_active_overrides(self) -> dict[str, Any]:
        """Clear and return all active overrides.

        Thread-safe: Protected by RLock.

        Returns:
            The cleared active overrides dictionary
        """
        with self._lock:
            overrides = dict(self._active_overrides)
            self._active_overrides.clear()
            return overrides

    def clear_all(self) -> None:
        """Clear all stored state.

        Thread-safe: Protected by RLock.
        """
        with self._lock:
            self._original_methods.clear()
            self._original_constructors.clear()
            self._active_overrides.clear()
        self.set_active(False)

    @property
    def original_methods(self) -> dict[str, tuple[Any, str | None, Any]]:
        """Get a thread-safe copy of original methods.

        Returns:
            Copy of the original methods dictionary
        """
        with self._lock:
            return dict(self._original_methods)

    @property
    def original_constructors(self) -> dict[str, Any]:
        """Get a thread-safe copy of original constructors.

        Returns:
            Copy of the original constructors dictionary
        """
        with self._lock:
            return dict(self._original_constructors)


def create_activation_state() -> ActivationState:
    """Factory function to create an ActivationState instance.

    Returns:
        New ActivationState instance
    """
    return ActivationState()
