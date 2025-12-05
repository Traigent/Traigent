"""Comprehensive tests for BaseOverrideManager (base.py).

This test suite covers:
- Base override manager functionality
- Thread-local override state management
- Configuration extraction and handling
- Context management and lifecycle
- Error handling and edge cases
"""

import threading
from unittest.mock import patch

import pytest

from traigent.config.types import TraigentConfig
from traigent.integrations.base import BaseOverrideManager

# Mock classes for testing


class MockFrameworkClass:
    """Mock framework class for testing overrides."""

    def __init__(self, model: str = "default", temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    def generate(self, prompt: str, **kwargs):
        return f"Generated: {prompt} with {self.model}"


# Test fixtures


@pytest.fixture
def base_manager():
    """Fresh BaseOverrideManager for each test."""
    return BaseOverrideManager()


@pytest.fixture
def sample_traigent_config():
    """Sample TraigentConfig for testing."""
    return TraigentConfig(
        model="gpt-4",
        temperature=0.8,
        max_tokens=2000,
        custom_params={"extra_param": "extra_value"},
    )


@pytest.fixture
def sample_dict_config():
    """Sample dictionary config for testing."""
    return {
        "model": "claude-3-sonnet",
        "temperature": 0.6,
        "max_tokens": 1500,
        "stream": False,
    }


# Test Classes


class TestBaseOverrideManagerInitialization:
    """Test BaseOverrideManager initialization."""

    def test_basic_initialization(self, base_manager):
        """Test basic initialization."""
        assert base_manager is not None
        assert hasattr(base_manager, "_active_overrides")
        assert hasattr(base_manager, "_original_constructors")
        assert hasattr(base_manager, "_original_methods")
        assert hasattr(base_manager, "_override_active")

        # Initial state should be empty
        assert len(base_manager._active_overrides) == 0
        assert len(base_manager._original_constructors) == 0
        assert len(base_manager._original_methods) == 0

    def test_thread_local_initialization(self, base_manager):
        """Test thread-local override state initialization."""
        # Initially should not be active
        assert base_manager.is_override_active() is False

        # Set active
        base_manager.set_override_active(True)
        assert base_manager.is_override_active() is True

        # Set inactive
        base_manager.set_override_active(False)
        assert base_manager.is_override_active() is False


class TestOverrideStateManagement:
    """Test override state management."""

    def test_override_active_state(self, base_manager):
        """Test getting and setting override active state."""
        # Initial state
        assert base_manager.is_override_active() is False

        # Activate
        base_manager.set_override_active(True)
        assert base_manager.is_override_active() is True

        # Deactivate
        base_manager.set_override_active(False)
        assert base_manager.is_override_active() is False

    def test_thread_isolation_of_override_state(self, base_manager):
        """Test that override state is isolated between threads."""
        results = {}

        def thread_worker(thread_id, should_activate):
            """Worker function to test thread isolation."""
            if should_activate:
                base_manager.set_override_active(True)
            results[thread_id] = base_manager.is_override_active()

        # Start multiple threads with different states
        threads = []
        for i in range(4):
            should_activate = i % 2 == 0  # Activate for even thread IDs
            thread = threading.Thread(target=thread_worker, args=(i, should_activate))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == 4
        assert results[0] is True  # Even - activated
        assert results[1] is False  # Odd - not activated
        assert results[2] is True  # Even - activated
        assert results[3] is False  # Odd - not activated

    def test_override_state_persistence_within_thread(self, base_manager):
        """Test that override state persists within a thread."""
        # Set state
        base_manager.set_override_active(True)

        # State should persist across multiple calls
        assert base_manager.is_override_active() is True
        assert base_manager.is_override_active() is True

        # Change state
        base_manager.set_override_active(False)
        assert base_manager.is_override_active() is False
        assert base_manager.is_override_active() is False


class TestConfigurationExtraction:
    """Test configuration extraction functionality."""

    def test_extract_traigent_config_dict(self, base_manager, sample_traigent_config):
        """Test extracting dict from TraigentConfig."""
        config_dict = base_manager.extract_config_dict(sample_traigent_config)

        assert config_dict is not None
        assert isinstance(config_dict, dict)
        assert config_dict["model"] == "gpt-4"
        assert config_dict["temperature"] == 0.8
        assert config_dict["max_tokens"] == 2000

        # Should include custom params
        assert config_dict["extra_param"] == "extra_value"

    def test_extract_dict_config(self, base_manager, sample_dict_config):
        """Test extracting dict from dictionary config."""
        config_dict = base_manager.extract_config_dict(sample_dict_config)

        assert config_dict is not None
        assert config_dict == sample_dict_config
        assert config_dict["model"] == "claude-3-sonnet"
        assert config_dict["temperature"] == 0.6

    def test_extract_config_from_none(self, base_manager):
        """Test extracting config from None."""
        config_dict = base_manager.extract_config_dict(None)

        assert config_dict is None

    def test_extract_config_from_empty_dict(self, base_manager):
        """Test extracting config from empty dict."""
        config_dict = base_manager.extract_config_dict({})

        assert config_dict is not None  # Empty dict is still valid
        assert config_dict == {}

    def test_extract_config_from_invalid_type(self, base_manager):
        """Test extracting config from invalid type."""
        config_dict = base_manager.extract_config_dict("invalid_config")

        assert config_dict is None


class TestMethodWrappers:
    """Test method wrapper functionality."""

    def test_get_method_wrapper_sync(self, base_manager):
        """Test getting wrapper for synchronous method."""

        def sync_method():
            return "sync"

        wrapper_func = base_manager.get_method_wrapper(sync_method)
        assert wrapper_func == base_manager.create_sync_wrapper

    def test_get_method_wrapper_async(self, base_manager):
        """Test getting wrapper for asynchronous method."""

        async def async_method():
            return "async"

        wrapper_func = base_manager.get_method_wrapper(async_method)
        assert wrapper_func == base_manager.create_async_wrapper

    def test_create_sync_wrapper(self, base_manager):
        """Test creating synchronous wrapper."""

        def original_method(**kwargs):
            return kwargs

        def apply_overrides(config_dict, kwargs):
            kwargs["overridden"] = True
            return kwargs

        wrapper = base_manager.create_sync_wrapper(original_method, apply_overrides)
        assert callable(wrapper)

        # Test without override active
        result = wrapper(test="value")
        assert result == {"test": "value"}

        # Test with override active but no config
        with patch("traigent.integrations.base.get_config", return_value=None):
            base_manager.set_override_active(True)
            result = wrapper(test="value")
            assert result == {"test": "value"}

    @pytest.mark.asyncio
    async def test_create_async_wrapper(self, base_manager):
        """Test creating asynchronous wrapper."""

        async def original_method(**kwargs):
            return kwargs

        def apply_overrides(config_dict, kwargs):
            kwargs["overridden"] = True
            return kwargs

        wrapper = base_manager.create_async_wrapper(original_method, apply_overrides)
        assert callable(wrapper)

        # Test without override active
        result = await wrapper(test="value")
        assert result == {"test": "value"}

        # Test with override active but no config
        with patch("traigent.integrations.base.get_config", return_value=None):
            base_manager.set_override_active(True)
            result = await wrapper(test="value")
            assert result == {"test": "value"}


class TestStoreAndRestore:
    """Test storing and restoring original methods."""

    def test_store_original_constructor(self, base_manager):
        """Test storing original constructor."""
        base_manager.store_original(
            MockFrameworkClass, "__init__", MockFrameworkClass.__init__, "constructor"
        )

        key = f"{MockFrameworkClass.__module__}.{MockFrameworkClass.__name__}.__init__"
        assert key in base_manager._original_constructors
        assert base_manager._original_constructors[key][0] == MockFrameworkClass
        assert base_manager._original_constructors[key][1] == "__init__"

    def test_store_original_method(self, base_manager):
        """Test storing original method."""
        original_method = MockFrameworkClass.generate
        base_manager.store_original(
            MockFrameworkClass, "generate", original_method, "method"
        )

        key = f"{MockFrameworkClass.__module__}.{MockFrameworkClass.__name__}.generate"
        assert key in base_manager._original_methods
        assert base_manager._original_methods[key][0] == MockFrameworkClass
        assert base_manager._original_methods[key][1] == "generate"

    def test_restore_originals(self, base_manager):
        """Test restoring all originals."""
        # Store some originals
        original_init = MockFrameworkClass.__init__
        original_generate = MockFrameworkClass.generate

        base_manager.store_original(
            MockFrameworkClass, "__init__", original_init, "constructor"
        )
        base_manager.store_original(
            MockFrameworkClass, "generate", original_generate, "method"
        )

        # Verify they're stored
        assert len(base_manager._original_constructors) == 1
        assert len(base_manager._original_methods) == 1

        # Restore
        base_manager.restore_originals()

        # Verify they're cleared
        assert len(base_manager._original_constructors) == 0
        assert len(base_manager._original_methods) == 0


class TestContextManagement:
    """Test context management functionality."""

    def test_override_context_manager(self, base_manager):
        """Test override context manager."""
        # Should not be active initially
        assert base_manager.is_override_active() is False

        # Use context manager
        with base_manager.override_context():
            # Should be active inside context
            assert base_manager.is_override_active() is True

        # Should be inactive after context
        assert base_manager.is_override_active() is False

    def test_override_context_with_exception(self, base_manager):
        """Test override context manager with exception."""
        # Should not be active initially
        assert base_manager.is_override_active() is False

        # Use context manager with exception
        try:
            with base_manager.override_context():
                # Should be active inside context
                assert base_manager.is_override_active() is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should be inactive after exception
        assert base_manager.is_override_active() is False

    def test_override_context_restores_originals(self, base_manager):
        """Test that override context restores originals."""
        # Store some originals
        base_manager.store_original(
            MockFrameworkClass, "__init__", MockFrameworkClass.__init__, "constructor"
        )

        assert len(base_manager._original_constructors) == 1

        # Use context manager
        with base_manager.override_context():
            assert len(base_manager._original_constructors) == 1

        # Should be restored after context
        assert len(base_manager._original_constructors) == 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_extract_invalid_config_type(self, base_manager):
        """Test extracting invalid config type."""
        invalid_configs = [123, [], "string", object()]

        for invalid_config in invalid_configs:
            result = base_manager.extract_config_dict(invalid_config)
            # Should return None for invalid types
            assert result is None

    def test_restore_with_error(self, base_manager):
        """Test restore originals with error."""

        # Create a mock class that raises error on setattr
        class ErrorClass:
            def __setattr__(self, name, value):
                raise RuntimeError("Cannot set attribute")

        error_obj = ErrorClass()

        # Store with error object
        base_manager._original_methods["test.error"] = (
            error_obj,
            "method",
            lambda: None,
        )

        # Should handle error gracefully
        base_manager.restore_originals()

        # Should have cleared the storage
        assert len(base_manager._original_methods) == 0

    def test_concurrent_access_safety(self, base_manager):
        """Test thread safety of concurrent access."""
        results = {}
        errors = []

        def worker_thread(thread_id):
            """Worker thread for concurrent access test."""
            try:
                # Set override state
                base_manager.set_override_active(thread_id % 2 == 0)

                # Store something
                base_manager.store_original(
                    MockFrameworkClass, f"method_{thread_id}", lambda: None, "method"
                )

                # Check state
                results[thread_id] = base_manager.is_override_active()

            except Exception as e:
                errors.append(e)

        # Run concurrent threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 10

        # Verify thread-local isolation
        for thread_id, is_active in results.items():
            expected = thread_id % 2 == 0
            assert is_active == expected
