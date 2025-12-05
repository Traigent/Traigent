"""Base test class and utilities for decorator tests."""

import pytest

from traigent.config.context import get_config, set_config
from traigent.integrations.framework_override import (
    disable_framework_overrides,
)

from .mock_infrastructure import ConfigurationLogger


class DecoratorTestBase:
    """Base class for decorator tests with common setup and utilities."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Store original config
        self.original_config = get_config()

        # Create fresh logger for each test
        self.config_logger = ConfigurationLogger()

        # Clear any existing framework overrides
        disable_framework_overrides()

        yield

        # Restore original config
        set_config(self.original_config)

        # Clear framework overrides
        disable_framework_overrides()

    def create_test_function(self, framework: str, async_mode: bool = False):
        """Create a test function that uses the specified framework."""
        if async_mode:

            async def test_func(text: str) -> str:
                # Simulate framework usage
                return f"{framework} async response for: {text}"

            return test_func
        else:

            def test_func(text: str) -> str:
                # Simulate framework usage
                return f"{framework} response for: {text}"

            return test_func

    def assert_config_applied(self, expected_config: dict):
        """Assert that expected configuration was applied."""
        # Verify the logger recorded configuration application
        assert len(self.config_logger.logs) > 0, "Expected config logs but none found"

        # Check that each expected key-value pair was logged
        for key, value in expected_config.items():
            assert self.config_logger.has_config(key, value), (
                f"Expected config {key}={value} not found in logs. "
                f"Logged configs: {[log.config for log in self.config_logger.logs]}"
            )

    def assert_no_config_applied(self):
        """Assert that no configuration was applied."""
        assert len(self.config_logger.logs) == 0
