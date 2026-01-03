"""Shared fixtures and configuration for security tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_credentials_path():
    """Create temporary path for credential storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_credentials.enc"


@pytest.fixture
def mock_jwks_server():
    """Mock JWKS server for JWT testing."""
    mock_server = MagicMock()
    mock_server.get_signing_key_from_jwt.return_value = MagicMock(
        key="test_key",
        algorithm_name="RS256",
        key_id="test_key_id",
        key_size=2048,
    )
    return mock_server


@pytest.fixture
def test_environments():
    """Provide test environment configurations."""
    return {
        "production": {
            "TRAIGENT_ENVIRONMENT": "production",
            "TRAIGENT_JWKS_URL": "https://test.example.com/jwks",
            "TRAIGENT_JWT_ISSUER": "test_issuer",
            "TRAIGENT_JWT_AUDIENCE": "test_audience",
            "TRAIGENT_MASTER_PASSWORD": "test_secure_passphrase_123",
        },
        "staging": {
            "TRAIGENT_ENVIRONMENT": "staging",
            "TRAIGENT_JWKS_URL": "https://staging.example.com/jwks",
            "TRAIGENT_JWT_ISSUER": "staging_issuer",
            "TRAIGENT_JWT_AUDIENCE": "staging_audience",
            "TRAIGENT_MASTER_PASSWORD": "staging_password_123",
        },
        "development": {
            "TRAIGENT_ENVIRONMENT": "development",
            "TRAIGENT_MASTER_PASSWORD": "dev_password_123",
        },
    }


@pytest.fixture
def clean_environment():
    """Clean environment variables before and after tests."""
    # Save original environment
    original_env = os.environ.copy()

    # Clean security-related variables
    security_vars = [
        "TRAIGENT_ENVIRONMENT",
        "TRAIGENT_JWKS_URL",
        "TRAIGENT_JWT_ISSUER",
        "TRAIGENT_JWT_AUDIENCE",
        "TRAIGENT_JWT_VALIDATION_MODE",
        "TRAIGENT_JWT_BYPASS",
        "TRAIGENT_JWT_DISABLE_SECURITY",
        "TRAIGENT_MASTER_PASSWORD",
        "TRAIGENT_MASTER_KEY",
        "TRAIGENT_ENABLE_HSM",
    ]

    for var in security_vars:
        os.environ.pop(var, None)

    # Remove any other project-specific variables that start with the prefix
    dynamic_security_vars = [key for key in os.environ if key.startswith("TRAIGENT_")]
    for var in dynamic_security_vars:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_audit_callback():
    """Mock audit callback for testing."""
    callback = MagicMock()
    return callback


@pytest.fixture
def sample_jwt_payload():
    """Sample JWT payload for testing."""
    import time

    return {
        "sub": "test_user_123",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "nbf": int(time.time()),
        "jti": "unique_token_id_123",
        "iss": "test_issuer",
        "aud": "test_audience",
        "scope": "read write",
        "roles": ["user", "admin"],
    }


@pytest.fixture
def sample_credentials():
    """Sample credentials for testing."""
    return {
        "api_key": {
            "name": "OPENAI_API_KEY",
            "value": "example-key-1234567890",
            "type": "api_key",
        },
        "database": {
            "name": "DATABASE_URL",
            "value": "postgresql://user:pass@localhost/db",
            "type": "database_url",
        },
        "secret": {
            "name": "APP_SECRET",
            "value": "super_secret_value_123",
            "type": "secret",
        },
    }
