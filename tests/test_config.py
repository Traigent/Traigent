"""Test configuration for TraiGent tests.

This module provides test-specific configuration and mock credentials.
These are NOT real credentials and should only be used in testing.
"""

import os


class TestConfig:
    """Test configuration with mock credentials."""

    # Mock credentials for testing only
    # These are NOT real and should never be used in production
    TEST_API_KEY = os.getenv("TEST_API_KEY", "test-api-key-mock-only")
    TEST_SECRET_KEY = os.getenv("TEST_SECRET_KEY", "test-secret-key-mock-only")
    TEST_JWT_SECRET = os.getenv("TEST_JWT_SECRET", "test-jwt-secret-mock-only")
    TEST_PASSWORD = os.getenv("TEST_PASSWORD", "test-password-mock-only")
    TEST_ADMIN_PASSWORD = os.getenv("TEST_ADMIN_PASSWORD", "test-admin-mock-only")

    # Test database configuration
    TEST_DATABASE_URL = os.getenv(
        "TEST_DATABASE_URL", "sqlite:///:memory:"  # In-memory database for tests
    )

    # Test Redis URL
    TEST_REDIS_URL = os.getenv(
        "TEST_REDIS_URL", "redis://localhost:6379/15"  # Use database 15 for tests
    )

    # Mock mode for tests
    MOCK_MODE = True

    @classmethod
    def get_test_credential(cls, key: str) -> str:
        """Get test credential with clear indication it's for testing."""
        value = getattr(cls, key, None)
        if not value:
            return f"mock-{key.lower()}-for-tests-only"
        return value
