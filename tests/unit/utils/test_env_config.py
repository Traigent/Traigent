"""Tests for environment configuration utilities."""

import sys
from types import SimpleNamespace

import pytest

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None)

from traigent.utils import env_config


def _reset_env(monkeypatch):
    """Clear critical environment variables to isolate test cases."""
    for key in (
        "JWT_SECRET_KEY",
        "TRAIGENT_MOCK_LLM",
        "TRAIGENT_OFFLINE_MODE",
        "ENVIRONMENT",
        "TRAIGENT_DEV_JWT_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env_config, "_GENERATED_DEV_JWT_SECRET", None, raising=False)


@pytest.mark.parametrize(
    ("env", "expect_error"),
    [("production", True), ("development", False), ("Dev", False)],
)
def test_get_jwt_secret_missing_secret(monkeypatch, env, expect_error):
    """Missing JWT secret should be disallowed in production but generate a secure dev secret otherwise."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", env)

    if expect_error:
        with pytest.raises(ValueError):
            env_config.get_jwt_secret()
    else:
        with pytest.warns(UserWarning):
            secret = env_config.get_jwt_secret()
        assert secret
        assert len(secret) >= env_config._MIN_JWT_SECRET_LENGTH
        # Subsequent calls should return the same generated secret for stability.
        assert env_config.get_jwt_secret() == secret


def test_get_jwt_secret_accepts_mock_mode(monkeypatch):
    """Mock LLM mode should allow fallback secret even if environment not set."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    with pytest.warns(UserWarning):
        secret = env_config.get_jwt_secret()
    assert secret
    assert len(secret) >= env_config._MIN_JWT_SECRET_LENGTH
    assert env_config.get_jwt_secret() == secret


def test_get_jwt_secret_returns_existing_value(monkeypatch):
    """Explicit JWT secret should be returned unchanged."""
    _reset_env(monkeypatch)
    value = "a" * 40
    monkeypatch.setenv("JWT_SECRET_KEY", value)
    monkeypatch.setenv("ENVIRONMENT", "production")

    secret = env_config.get_jwt_secret()
    assert secret == value


def test_get_jwt_secret_warns_on_short_secret(monkeypatch):
    """Short JWT secrets trigger a warning to encourage rotation."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("JWT_SECRET_KEY", "short-secret")

    with pytest.warns(UserWarning):
        assert env_config.get_jwt_secret() == "short-secret"


def test_get_jwt_secret_uses_override(monkeypatch):
    """Custom development override should take precedence."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("TRAIGENT_DEV_JWT_SECRET", "custom-dev-key")

    jwt_value = env_config.get_jwt_secret()  # pragma: allowlist secret
    assert jwt_value == "custom-dev-key"


def test_get_env_var_masks_value_in_logs(monkeypatch, caplog):
    """Sensitive environment variables should be masked in logs."""
    key = "SECRET_TEST_VAR"
    value = "supersecretvalue"
    monkeypatch.setenv(key, value)

    with caplog.at_level("INFO"):
        retrieved = env_config.get_env_var(key, mask_in_logs=True)

    assert retrieved == value
    assert key in caplog.text
    assert value not in caplog.text


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("false", False),
        ("", False),
    ],
)
def test_is_strict_cost_accounting(monkeypatch, raw_value, expected):
    """Strict cost accounting flag should parse bool-like env values."""
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", raw_value)
    assert env_config.is_strict_cost_accounting() is expected
