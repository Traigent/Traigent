"""Tests for the feature flag registry."""

from __future__ import annotations

import pytest

from traigent.api.functions import configure
from traigent.config.feature_flags import (
    Flag,
    FlagNames,
    flag_registry,
    is_backend_smart_optimizers_enabled,
    is_local_advanced_optimizers_enabled,
)


@pytest.fixture(autouse=True)
def reset_flags(monkeypatch):
    """Reset registry overrides/config before each test."""
    flag_registry.reset()
    monkeypatch.delenv("TRAIGENT_BACKEND_SMART_OPTIMIZERS_ENABLED", raising=False)
    monkeypatch.delenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", raising=False)
    yield
    flag_registry.reset()


def test_smart_optimizer_flags_default_to_false():
    assert is_backend_smart_optimizers_enabled() is False
    assert is_local_advanced_optimizers_enabled() is False


def test_manual_override_context():
    assert is_backend_smart_optimizers_enabled() is False

    with flag_registry.override(FlagNames.BACKEND_SMART_OPTIMIZERS, True):
        assert is_backend_smart_optimizers_enabled() is True

    assert is_backend_smart_optimizers_enabled() is False


def test_registering_duplicate_flag_raises():
    duplicate = Flag(name=FlagNames.BACKEND_SMART_OPTIMIZERS)
    with pytest.raises(ValueError):
        flag_registry.register(duplicate)


def test_snapshot_includes_registered_flags():
    snapshot = flag_registry.snapshot()
    assert FlagNames.BACKEND_SMART_OPTIMIZERS in snapshot
    assert snapshot[FlagNames.BACKEND_SMART_OPTIMIZERS] is False
    assert FlagNames.LOCAL_ADVANCED_OPTIMIZERS in snapshot
    assert snapshot[FlagNames.LOCAL_ADVANCED_OPTIMIZERS] is False


def test_backend_smart_optimizer_flag_overrides(monkeypatch):
    monkeypatch.setenv("TRAIGENT_BACKEND_SMART_OPTIMIZERS_ENABLED", "enabled")
    assert is_backend_smart_optimizers_enabled() is True

    monkeypatch.setenv("TRAIGENT_BACKEND_SMART_OPTIMIZERS_ENABLED", "0")
    assert is_backend_smart_optimizers_enabled() is False

    monkeypatch.delenv("TRAIGENT_BACKEND_SMART_OPTIMIZERS_ENABLED", raising=False)
    flag_registry.apply_config({"optimizers": {"backend_smart": {"enabled": True}}})
    assert is_backend_smart_optimizers_enabled() is True


def test_configure_applies_backend_smart_feature_flag(monkeypatch):
    flag_registry.reset()

    configure(feature_flags={"optimizers": {"backend_smart": {"enabled": True}}})
    assert is_backend_smart_optimizers_enabled() is True


def test_configure_does_not_register_smart_optimizers_when_enabled(monkeypatch):
    from traigent.optimizers.registry import (
        _register_builtin_optimizers,
        clear_registry,
        list_optimizers,
    )

    flag_registry.reset()
    clear_registry()
    _register_builtin_optimizers()
    assert "tpe" not in list_optimizers()

    try:
        configure(feature_flags={"optimizers": {"backend_smart": {"enabled": True}}})

        assert sorted(list_optimizers()) == ["grid", "random"]
    finally:
        clear_registry()
        _register_builtin_optimizers()


def test_deprecated_local_advanced_optimizer_flag_overrides_are_ignored(monkeypatch):
    monkeypatch.setenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", "enabled")
    assert is_local_advanced_optimizers_enabled() is False

    monkeypatch.setenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", "0")
    assert is_local_advanced_optimizers_enabled() is False

    monkeypatch.delenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", raising=False)
    flag_registry.apply_config({"optimizers": {"advanced_local": {"enabled": True}}})
    assert is_local_advanced_optimizers_enabled() is False
