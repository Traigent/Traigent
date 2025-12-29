"""Test isolation utilities for Traigent test suite.

This module provides utilities to ensure proper test isolation and prevent
global state pollution between tests.
"""

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any


class GlobalStateManager:
    """Manages global state for test isolation.

    This class provides utilities to capture, reset, and restore global state
    to ensure tests don't interfere with each other.
    """

    @staticmethod
    def capture_state() -> dict[str, Any]:
        """Capture current global state from all known global objects.

        Returns:
            Dictionary containing snapshots of all global state
        """
        from traigent.api.functions import _GLOBAL_CONFIG
        from traigent.config.api_keys import _API_KEY_MANAGER

        return {
            "api_keys": _API_KEY_MANAGER._keys.copy(),
            "api_warned": _API_KEY_MANAGER._warned,
            "global_config": _GLOBAL_CONFIG.copy(),
            "warning_filters": (
                warnings.filters.copy() if hasattr(warnings, "filters") else []
            ),
        }

    @staticmethod
    def restore_state(state: dict[str, Any]) -> None:
        """Restore global state from a captured snapshot.

        Args:
            state: Dictionary containing state snapshots to restore
        """
        from traigent.api.functions import _GLOBAL_CONFIG
        from traigent.config.api_keys import _API_KEY_MANAGER

        # Restore API key manager state
        _API_KEY_MANAGER._keys.clear()
        _API_KEY_MANAGER._keys.update(state.get("api_keys", {}))
        _API_KEY_MANAGER._warned = state.get("api_warned", False)

        # Restore global config
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(
            state.get(
                "global_config",
                {
                    "default_storage_backend": "edge_analytics",
                    "parallel_workers": 1,
                    "cache_policy": "memory",
                    "logging_level": "INFO",
                    "api_keys": {},
                },
            )
        )

        # Restore warning filters
        warnings.resetwarnings()
        if "warning_filters" in state:
            warnings.filters = state["warning_filters"].copy()

    @staticmethod
    def reset_all() -> None:
        """Reset all known global state to clean defaults."""
        from traigent.api.functions import _GLOBAL_CONFIG
        from traigent.config.api_keys import _API_KEY_MANAGER

        # Reset API key manager
        _API_KEY_MANAGER._keys.clear()
        _API_KEY_MANAGER._warned = False

        # Reset global config to defaults
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(
            {
                "default_storage_backend": "edge_analytics",
                "parallel_workers": 1,
                "cache_policy": "memory",
                "logging_level": "INFO",
                "api_keys": {},
            }
        )

        # Reset warnings
        warnings.resetwarnings()

    @staticmethod
    def verify_clean_state() -> bool:
        """Verify that global state is in a clean default state.

        Returns:
            True if state is clean, False otherwise
        """
        from traigent.api.functions import _GLOBAL_CONFIG
        from traigent.config.api_keys import _API_KEY_MANAGER

        # Check API key manager is clean
        if _API_KEY_MANAGER._keys or _API_KEY_MANAGER._warned:
            return False

        # Check global config has default values
        expected_config = {
            "default_storage_backend": "edge_analytics",
            "parallel_workers": 1,
            "cache_policy": "memory",
            "logging_level": "INFO",
            "api_keys": {},
        }

        if _GLOBAL_CONFIG != expected_config:
            return False

        return True

    @staticmethod
    def get_state_diff(
        state1: dict[str, Any], state2: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare two state snapshots and return differences.

        Args:
            state1: First state snapshot
            state2: Second state snapshot

        Returns:
            Dictionary describing the differences
        """
        diff = {}

        # Compare API keys
        if state1.get("api_keys") != state2.get("api_keys"):
            diff["api_keys"] = {
                "before": state1.get("api_keys"),
                "after": state2.get("api_keys"),
            }

        # Compare warned flag
        if state1.get("api_warned") != state2.get("api_warned"):
            diff["api_warned"] = {
                "before": state1.get("api_warned"),
                "after": state2.get("api_warned"),
            }

        # Compare global config
        if state1.get("global_config") != state2.get("global_config"):
            diff["global_config"] = {
                "before": state1.get("global_config"),
                "after": state2.get("global_config"),
            }

        return diff


@contextmanager
def isolated_test_context():
    """Context manager that ensures test runs with clean global state.

    Example:
        with isolated_test_context():
            # Test code here runs with clean state
            run_test()
    """
    manager = GlobalStateManager()

    # Capture current state
    original_state = manager.capture_state()

    # Reset to clean state
    manager.reset_all()

    try:
        yield
    finally:
        # Always restore original state
        manager.restore_state(original_state)


def with_clean_state(func: Callable) -> Callable:
    """Decorator that ensures a test function runs with clean global state.

    Args:
        func: Test function to wrap

    Returns:
        Wrapped function that runs with clean state

    Example:
        @with_clean_state
        def test_something():
            # This test runs with guaranteed clean state
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with isolated_test_context():
            return func(*args, **kwargs)

    return wrapper


def detect_state_leaks(func: Callable) -> Callable:
    """Decorator that detects if a test leaks global state.

    Args:
        func: Test function to monitor

    Returns:
        Wrapped function that checks for state leaks

    Example:
        @detect_state_leaks
        def test_something():
            # This test will fail if it modifies global state
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        manager = GlobalStateManager()

        # Capture state before test
        state_before = manager.capture_state()

        # Run the test
        result = func(*args, **kwargs)

        # Capture state after test
        state_after = manager.capture_state()

        # Check for differences
        diff = manager.get_state_diff(state_before, state_after)

        if diff:
            import pytest

            pytest.fail(f"Test leaked global state: {diff}")

        return result

    return wrapper


class TestIsolationMixin:
    """Mixin class that provides automatic test isolation for test classes.

    Example:
        class TestMyFeature(TestIsolationMixin):
            def test_something(self):
                # Automatically runs with clean state
                pass
    """

    def setup_method(self, method):
        """Set up clean state before each test method."""
        self._state_manager = GlobalStateManager()
        self._original_state = self._state_manager.capture_state()
        self._state_manager.reset_all()

        # Call parent setup if it exists
        if hasattr(super(), "setup_method"):
            super().setup_method(method)

    def teardown_method(self, method):
        """Clean up and restore state after each test method."""
        # Call parent teardown if it exists
        if hasattr(super(), "teardown_method"):
            super().teardown_method(method)

        # Restore original state
        if hasattr(self, "_state_manager") and hasattr(self, "_original_state"):
            self._state_manager.restore_state(self._original_state)
