"""Tests for activation state management.

Tests thread safety and state management in ActivationState class.
# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from traigent.integrations.activation import ActivationState, create_activation_state


class TestActivationStateBasic:
    """Basic ActivationState functionality tests."""

    def test_initial_state(self):
        """Test initial state is inactive with empty collections."""
        state = ActivationState()

        assert state.is_active() is False
        assert state.original_methods == {}
        assert state.original_constructors == {}
        assert state.get_active_overrides() == {}

    def test_set_active_state(self):
        """Test setting active state."""
        state = ActivationState()

        state.set_active(True)
        assert state.is_active() is True

        state.set_active(False)
        assert state.is_active() is False

    def test_store_and_retrieve_method(self):
        """Test storing and retrieving original methods."""
        state = ActivationState()

        def original_method():
            return "original"

        state.store_original_method("test.Class.method", original_method)

        assert state.has_original_method("test.Class.method")
        stored = state.get_original_method("test.Class.method")
        assert stored is not None
        assert stored[2] == original_method

    def test_store_and_retrieve_constructor(self):
        """Test storing and retrieving original constructors."""
        state = ActivationState()

        def original_init():
            pass

        state.store_original_constructor("test.Class", original_init)

        assert state.has_original_constructor("test.Class")
        stored = state.get_original_constructor("test.Class")
        assert stored == original_init

    def test_remove_method(self):
        """Test removing stored method."""
        state = ActivationState()

        def original_method():
            return "original"

        state.store_original_method("test.Class.method", original_method)
        assert state.has_original_method("test.Class.method")

        removed = state.remove_original_method("test.Class.method")
        assert removed is not None
        assert not state.has_original_method("test.Class.method")

    def test_remove_constructor(self):
        """Test removing stored constructor."""
        state = ActivationState()

        def original_init():
            pass

        state.store_original_constructor("test.Class", original_init)
        assert state.has_original_constructor("test.Class")

        removed = state.remove_original_constructor("test.Class")
        assert removed == original_init
        assert not state.has_original_constructor("test.Class")

    def test_register_active_override(self):
        """Test registering active overrides."""
        state = ActivationState()

        class MockClass:
            pass

        state.register_active_override("test.Class", MockClass)

        assert state.is_override_registered("test.Class")
        overrides = state.get_active_overrides()
        assert "test.Class" in overrides
        assert overrides["test.Class"] == MockClass

    def test_clear_all(self):
        """Test clearing all state."""
        state = ActivationState()

        state.set_active(True)
        state.store_original_method("test.method", lambda: None)
        state.store_original_constructor("test.Class", lambda: None)
        state.register_active_override("test.Class", object)

        state.clear_all()

        assert state.is_active() is False
        assert state.original_methods == {}
        assert state.original_constructors == {}
        assert state.get_active_overrides() == {}


class TestActivationStateThreadSafety:
    """Thread safety tests for ActivationState."""

    def test_thread_local_active_state(self):
        """Test that active state is thread-local."""
        state = ActivationState()
        results = {}

        def thread_func(thread_id, should_activate):
            if should_activate:
                state.set_active(True)
            time.sleep(0.01)  # Give other threads time to interfere
            results[thread_id] = state.is_active()

        threads = []
        # Thread 0 and 2 activate, threads 1 and 3 don't
        for i in range(4):
            t = threading.Thread(target=thread_func, args=(i, i % 2 == 0))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should see its own state
        assert results[0] is True
        assert results[1] is False
        assert results[2] is True
        assert results[3] is False

    def test_concurrent_method_storage(self):
        """Test concurrent method storage is thread-safe."""
        state = ActivationState()
        num_threads = 10
        methods_per_thread = 100

        def store_methods(thread_id):
            for i in range(methods_per_thread):
                key = f"thread_{thread_id}.method_{i}"
                state.store_original_method(key, lambda: None)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(store_methods, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # All methods should be stored
        assert len(state.original_methods) == num_threads * methods_per_thread

    def test_concurrent_constructor_storage(self):
        """Test concurrent constructor storage is thread-safe."""
        state = ActivationState()
        num_threads = 10
        constructors_per_thread = 50

        def store_constructors(thread_id):
            for i in range(constructors_per_thread):
                key = f"thread_{thread_id}.Class_{i}"
                state.store_original_constructor(key, lambda: None)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(store_constructors, i) for i in range(num_threads)
            ]
            for future in as_completed(futures):
                future.result()

        assert len(state.original_constructors) == num_threads * constructors_per_thread

    def test_concurrent_store_and_remove(self):
        """Test concurrent store and remove operations are thread-safe."""
        state = ActivationState()
        num_operations = 100

        def store_and_remove(operation_id):
            key = f"test.method_{operation_id}"
            state.store_original_method(key, lambda: None)
            time.sleep(0.001)
            state.remove_original_method(key)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(store_and_remove, i) for i in range(num_operations)
            ]
            for future in as_completed(futures):
                future.result()

        # All methods should be removed
        assert len(state.original_methods) == 0

    def test_concurrent_active_override_registration(self):
        """Test concurrent active override registration is thread-safe."""
        state = ActivationState()
        num_threads = 10
        overrides_per_thread = 50

        def register_overrides(thread_id):
            for i in range(overrides_per_thread):
                key = f"thread_{thread_id}.Class_{i}"
                state.register_active_override(key, object)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(register_overrides, i) for i in range(num_threads)
            ]
            for future in as_completed(futures):
                future.result()

        assert len(state.get_active_overrides()) == num_threads * overrides_per_thread

    @pytest.mark.asyncio
    async def test_active_state_isolated_between_coroutines(self):
        """Test active state is isolated across concurrent async tasks."""
        state = ActivationState()
        results: dict[str, tuple[bool, bool]] = {}

        async def worker(name: str, active: bool, delay: float) -> None:
            state._override_active.enabled = active
            before = state.is_active()
            await asyncio.sleep(delay)
            after = state.is_active()
            results[name] = (before, after)
            state._override_active.enabled = False

        await asyncio.gather(
            worker("a", True, 0.01),
            worker("b", False, 0.0),
        )

        assert results["a"] == (True, True)
        assert results["b"] == (False, False)
        assert state.is_active() is False


class TestActivationStatePropertyCopies:
    """Test that property accessors return copies for safety."""

    def test_original_methods_returns_copy(self):
        """Test that original_methods property returns a copy."""
        state = ActivationState()
        state.store_original_method("test.method", lambda: None)

        copy1 = state.original_methods
        copy2 = state.original_methods

        # Should be equal but not the same object
        assert copy1 == copy2
        assert copy1 is not copy2

        # Modifying copy should not affect state
        copy1["new_key"] = "value"
        assert "new_key" not in state.original_methods

    def test_original_constructors_returns_copy(self):
        """Test that original_constructors property returns a copy."""
        state = ActivationState()
        state.store_original_constructor("test.Class", lambda: None)

        copy1 = state.original_constructors
        copy2 = state.original_constructors

        assert copy1 == copy2
        assert copy1 is not copy2

        copy1["new_key"] = "value"
        assert "new_key" not in state.original_constructors

    def test_get_active_overrides_returns_copy(self):
        """Test that get_active_overrides returns a copy."""
        state = ActivationState()
        state.register_active_override("test.Class", object)

        copy1 = state.get_active_overrides()
        copy2 = state.get_active_overrides()

        assert copy1 == copy2
        assert copy1 is not copy2

        copy1["new_key"] = "value"
        assert "new_key" not in state.get_active_overrides()


class TestActivationStateFactory:
    """Test factory function."""

    def test_create_activation_state(self):
        """Test factory function creates new instance."""
        state1 = create_activation_state()
        state2 = create_activation_state()

        assert isinstance(state1, ActivationState)
        assert isinstance(state2, ActivationState)
        assert state1 is not state2
