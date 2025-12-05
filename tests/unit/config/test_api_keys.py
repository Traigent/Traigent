"""Tests for API key management functionality."""

import warnings
from unittest.mock import patch

from traigent.api.functions import configure, get_api_key
from traigent.config.api_keys import APIKeyManager

try:
    from tests.utils.isolation import TestIsolationMixin
except ImportError:
    # Fallback if running from different directory
    class TestIsolationMixin:
        def setup_method(self, method):
            pass


class TestAPIKeyManager(TestIsolationMixin):
    """Test API key management functionality."""

    def setup_method(self, method):
        """Reset API key manager state before each test."""
        # Call parent setup for isolation
        super().setup_method(method)
        # Create fresh manager instance for testing
        self.manager = APIKeyManager()
        with self.manager._lock:
            self.manager._keys.clear()
            self.manager._warned = False

    def test_security_warning_on_code_keys(self):
        """Test that API key manager shows security warning when keys set from code."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.manager.set_api_key("openai", "test-key", source="code")

            assert len(w) == 1
            assert "API keys detected in code" in str(w[0].message)
            assert "Environment variables" in str(w[0].message)

    def test_environment_priority(self):
        """Test that environment variables take priority over set keys."""
        # Set a key directly (suppress warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.manager.set_api_key("openai", "direct-key", source="code")

        # Mock environment variable
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            key = self.manager.get_api_key("openai")
            assert key == "env-key", "Environment variable should take priority"

        # Without environment variable, should return direct key
        with patch.dict("os.environ", {}, clear=True):
            key = self.manager.get_api_key("openai")
            assert key == "direct-key"

    def test_repr_security(self):
        """Test that API key manager doesn't expose keys in repr."""
        self.manager.set_api_key("openai", "secret-key", source="code")

        repr_str = repr(self.manager)
        str_str = str(self.manager)

        assert "secret-key" not in repr_str
        assert "secret-key" not in str_str
        assert "APIKeyManager" in repr_str
        assert "1 keys" in repr_str

    def test_configure_function_with_api_keys(self):
        """Test that configure function properly handles API keys."""
        # Clear environment to ensure test isolation
        with patch.dict("os.environ", {}, clear=True):
            # Configure with API keys should trigger warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                configure(api_keys={"openai": "test-key-123"})

                # Should have warned about API keys in code
                assert any(
                    "API keys detected in code" in str(warning.message) for warning in w
                )

            # Verify key was set
            key = get_api_key("openai")
            assert key == "test-key-123"

    def test_no_warning_on_environment_keys(self):
        """Test that no warning is shown when keys come from environment."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.manager.set_api_key("openai", "env-key", source="environment")

            # Should not warn for environment variables
            assert len(w) == 0

    def test_warning_only_once(self):
        """Test that warning is only shown once per session."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # First warning
            self.manager.set_api_key("openai", "key1", source="code")
            assert len(w) == 1

            # Second time should not warn
            self.manager.set_api_key("anthropic", "key2", source="code")
            assert len(w) == 1  # Still only one warning
