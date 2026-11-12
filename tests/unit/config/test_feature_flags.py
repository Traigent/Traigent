"""Tests for the feature flag registry."""

from __future__ import annotations

import pytest

from traigent.api.functions import configure
from traigent.config.feature_flags import Flag, flag_registry

# Register a test-only flag once at module level (FlagRegistry has no
# deregister API, so registration must be permanent for the test session).
# The flag name and env var are namespaced under "test.*" to avoid any
# future conflict with production flags.
_TEST_FLAG_NAME = "test.infrastructure.flag"
_TEST_ENV_VAR = "TRAIGENT_TEST_INFRA_FLAG_ENABLED"

flag_registry.register(
    Flag(
        name=_TEST_FLAG_NAME,
        default=True,
        env_var=_TEST_ENV_VAR,
        description="Test-only flag for registry infrastructure tests.",
        config_path="test.infrastructure.flag",
    )
)


@pytest.fixture(autouse=True)
def reset_flags(monkeypatch):
    """Reset registry overrides/config and env before each test."""
    flag_registry.reset()
    monkeypatch.delenv(_TEST_ENV_VAR, raising=False)
    yield
    flag_registry.reset()


def test_flag_defaults_to_registered_default():
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True


def test_environment_override(monkeypatch):
    monkeypatch.setenv(_TEST_ENV_VAR, "1")
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True

    monkeypatch.setenv(_TEST_ENV_VAR, "off")
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is False


def test_config_override(monkeypatch):
    monkeypatch.delenv(_TEST_ENV_VAR, raising=False)
    flag_registry.apply_config({"test": {"infrastructure": {"flag": True}}})
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True

    flag_registry.apply_config({"test": {"infrastructure": {"flag": "no"}}})
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is False


def test_manual_override_context():
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True

    with flag_registry.override(_TEST_FLAG_NAME, False):
        assert flag_registry.is_enabled(_TEST_FLAG_NAME) is False

    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True


def test_registering_duplicate_flag_raises():
    duplicate = Flag(name=_TEST_FLAG_NAME)
    with pytest.raises(ValueError):
        flag_registry.register(duplicate)


def test_snapshot_includes_registered_flags():
    snapshot = flag_registry.snapshot()
    assert _TEST_FLAG_NAME in snapshot
    assert snapshot[_TEST_FLAG_NAME] is True


def test_configure_applies_feature_flags(monkeypatch):
    monkeypatch.delenv(_TEST_ENV_VAR, raising=False)
    flag_registry.reset()

    configure(feature_flags={"test": {"infrastructure": {"flag": True}}})
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True


def test_override_restores_preexisting_override():
    flag_registry.set_override(_TEST_FLAG_NAME, False)
    with flag_registry.override(_TEST_FLAG_NAME, True):
        assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is False


def test_nested_overrides_restore_outer_value():
    with flag_registry.override(_TEST_FLAG_NAME, False):
        with flag_registry.override(_TEST_FLAG_NAME, True):
            assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True
        assert flag_registry.is_enabled(_TEST_FLAG_NAME) is False
    # No override left: back to the registered default.
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is True
    assert _TEST_FLAG_NAME not in flag_registry._overrides


def test_override_restores_previous_on_exception():
    flag_registry.set_override(_TEST_FLAG_NAME, False)
    with pytest.raises(RuntimeError):
        with flag_registry.override(_TEST_FLAG_NAME, True):
            raise RuntimeError("boom")
    assert flag_registry.is_enabled(_TEST_FLAG_NAME) is False


def test_override_clears_when_no_previous_on_exception():
    with pytest.raises(RuntimeError):
        with flag_registry.override(_TEST_FLAG_NAME, False):
            raise RuntimeError("boom")
    assert _TEST_FLAG_NAME not in flag_registry._overrides
