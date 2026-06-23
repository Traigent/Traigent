"""Tests for the feature flag registry."""

from __future__ import annotations

import pytest

from traigent.api.functions import configure
from traigent.config.feature_flags import (
    Flag,
    FlagNames,
    flag_registry,
    is_local_advanced_optimizers_enabled,
)


@pytest.fixture(autouse=True)
def reset_flags(monkeypatch):
    """Reset registry overrides/config before each test."""
    flag_registry.reset()
    monkeypatch.delenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", raising=False)
    yield
    flag_registry.reset()


def test_local_advanced_optimizer_flag_defaults_to_true():
    assert is_local_advanced_optimizers_enabled() is True


def test_environment_override(monkeypatch):
    monkeypatch.setenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", "1")
    assert is_local_advanced_optimizers_enabled() is True

    monkeypatch.setenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", "off")
    assert is_local_advanced_optimizers_enabled() is False


def test_config_override(monkeypatch):
    monkeypatch.delenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", raising=False)
    flag_registry.apply_config({"optimizers": {"local_advanced": {"enabled": True}}})
    assert is_local_advanced_optimizers_enabled() is True

    flag_registry.apply_config({"optimizers": {"local_advanced": {"enabled": "no"}}})
    assert is_local_advanced_optimizers_enabled() is False


def test_manual_override_context():
    assert is_local_advanced_optimizers_enabled() is True

    with flag_registry.override(FlagNames.LOCAL_ADVANCED_OPTIMIZERS, False):
        assert is_local_advanced_optimizers_enabled() is False

    assert is_local_advanced_optimizers_enabled() is True


def test_registering_duplicate_flag_raises():
    duplicate = Flag(name=FlagNames.LOCAL_ADVANCED_OPTIMIZERS)
    with pytest.raises(ValueError):
        flag_registry.register(duplicate)


def test_snapshot_includes_registered_flags():
    snapshot = flag_registry.snapshot()
    assert FlagNames.LOCAL_ADVANCED_OPTIMIZERS in snapshot
    assert snapshot[FlagNames.LOCAL_ADVANCED_OPTIMIZERS] is True


def test_configure_applies_feature_flags(monkeypatch):
    monkeypatch.delenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", raising=False)
    flag_registry.reset()

    configure(feature_flags={"optimizers": {"local_advanced": {"enabled": True}}})
    assert is_local_advanced_optimizers_enabled() is True
