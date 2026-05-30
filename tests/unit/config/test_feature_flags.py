"""Tests for the feature flag registry."""

from __future__ import annotations

import pytest

from traigent.api.functions import configure
from traigent.config.feature_flags import (
    Flag,
    FlagNames,
    flag_registry,
    is_local_advanced_optimizers_enabled,
    is_optuna_enabled,
)


@pytest.fixture(autouse=True)
def reset_flags(monkeypatch):
    """Reset registry overrides/config before each test."""
    flag_registry.reset()
    monkeypatch.delenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", raising=False)
    monkeypatch.delenv("TRAIGENT_OPTUNA_ENABLED", raising=False)
    yield
    flag_registry.reset()


def test_smart_optimizer_flags_default_to_false():
    assert is_optuna_enabled() is False
    assert is_local_advanced_optimizers_enabled() is False


def test_environment_override(monkeypatch):
    monkeypatch.setenv("TRAIGENT_OPTUNA_ENABLED", "1")
    assert is_optuna_enabled() is True

    monkeypatch.setenv("TRAIGENT_OPTUNA_ENABLED", "off")
    assert is_optuna_enabled() is False


def test_config_override(monkeypatch):
    monkeypatch.delenv("TRAIGENT_OPTUNA_ENABLED", raising=False)
    flag_registry.apply_config({"optimizers": {"optuna": {"enabled": True}}})
    assert is_optuna_enabled() is True

    flag_registry.apply_config({"optimizers": {"optuna": {"enabled": "no"}}})
    assert is_optuna_enabled() is False


def test_manual_override_context():
    assert is_optuna_enabled() is False

    with flag_registry.override(FlagNames.OPTUNA_ROLLOUT, True):
        assert is_optuna_enabled() is True

    assert is_optuna_enabled() is False


def test_registering_duplicate_flag_raises():
    duplicate = Flag(name=FlagNames.OPTUNA_ROLLOUT)
    with pytest.raises(ValueError):
        flag_registry.register(duplicate)


def test_snapshot_includes_registered_flags():
    snapshot = flag_registry.snapshot()
    assert FlagNames.OPTUNA_ROLLOUT in snapshot
    assert snapshot[FlagNames.OPTUNA_ROLLOUT] is False
    assert FlagNames.LOCAL_ADVANCED_OPTIMIZERS in snapshot
    assert snapshot[FlagNames.LOCAL_ADVANCED_OPTIMIZERS] is False


def test_configure_applies_feature_flags(monkeypatch):
    monkeypatch.delenv("TRAIGENT_OPTUNA_ENABLED", raising=False)
    flag_registry.reset()

    configure(feature_flags={"optimizers": {"optuna": {"enabled": True}}})
    assert is_optuna_enabled() is True


def test_configure_registers_optuna_optimizers_when_enabled(monkeypatch):
    pytest.importorskip("optuna")
    from traigent.optimizers.registry import (
        _register_builtin_optimizers,
        clear_registry,
        get_optimizer,
        list_optimizers,
    )

    monkeypatch.delenv("TRAIGENT_OPTUNA_ENABLED", raising=False)
    flag_registry.reset()
    clear_registry()
    _register_builtin_optimizers()
    assert "tpe" not in list_optimizers()

    try:
        configure(feature_flags={"optimizers": {"optuna": {"enabled": True}}})

        assert "tpe" in list_optimizers()
        assert (
            get_optimizer("tpe", {"x": [1, 2]}, ["accuracy"]).__class__.__name__
            == "OptunaTPEOptimizer"
        )
    finally:
        clear_registry()
        _register_builtin_optimizers()


def test_local_advanced_optimizer_flag_overrides(monkeypatch):
    monkeypatch.setenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", "enabled")
    assert is_local_advanced_optimizers_enabled() is True

    monkeypatch.setenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", "0")
    assert is_local_advanced_optimizers_enabled() is False

    monkeypatch.delenv("TRAIGENT_LOCAL_ADVANCED_OPTIMIZERS_ENABLED", raising=False)
    flag_registry.apply_config({"optimizers": {"advanced_local": {"enabled": True}}})
    assert is_local_advanced_optimizers_enabled() is True
