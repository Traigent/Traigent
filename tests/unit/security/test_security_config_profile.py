"""Tests for SDK security profile environment resolution."""

from traigent.security.config import SecurityProfile, get_security_profile


def _clear_security_env(monkeypatch):
    for key in (
        "TRAIGENT_SECURITY_PROFILE",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
        "TRAIGENT_ENVIRONMENT",
        "PYTEST_CURRENT_TEST",
    ):
        monkeypatch.delenv(key, raising=False)


def test_security_profile_honors_environment_primary_key(monkeypatch):
    """ENVIRONMENT=development should not silently fall through to production."""
    _clear_security_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")

    assert get_security_profile() is SecurityProfile.DEVELOPMENT


def test_security_profile_honors_legacy_environment_alias(monkeypatch):
    """Legacy TRAIGENT_ENV aliases should go through the central resolver."""
    _clear_security_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_ENV", "staging")

    assert get_security_profile() is SecurityProfile.STAGING


def test_security_profile_unknown_environment_fails_closed(monkeypatch):
    """Unknown deployment names keep production security defaults."""
    _clear_security_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "qa")

    assert get_security_profile() is SecurityProfile.PRODUCTION
