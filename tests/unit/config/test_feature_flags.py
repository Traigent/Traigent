"""Tests for the feature flag registry."""

from __future__ import annotations

import pytest

from traigent.api.functions import configure
from traigent.config.feature_flags import (
    Flag,
    FlagNames,
    flag_registry,
    is_optuna_enabled,
)


@pytest.fixture(autouse=True)
def reset_flags(monkeypatch):
    """Reset registry overrides/config before each test."""
    flag_registry.reset()
    monkeypatch.delenv("TRAIGENT_OPTUNA_ENABLED", raising=False)
    yield
    flag_registry.reset()


def test_optuna_flag_defaults_to_true():
    assert is_optuna_enabled() is True


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
    assert is_optuna_enabled() is True

    with flag_registry.override(FlagNames.OPTUNA_ROLLOUT, False):
        assert is_optuna_enabled() is False

    assert is_optuna_enabled() is True


def test_registering_duplicate_flag_raises():
    duplicate = Flag(name=FlagNames.OPTUNA_ROLLOUT)
    with pytest.raises(ValueError):
        flag_registry.register(duplicate)


def test_snapshot_includes_registered_flags():
    snapshot = flag_registry.snapshot()
    assert FlagNames.OPTUNA_ROLLOUT in snapshot
    assert snapshot[FlagNames.OPTUNA_ROLLOUT] is True


def test_configure_applies_feature_flags(monkeypatch):
    monkeypatch.delenv("TRAIGENT_OPTUNA_ENABLED", raising=False)
    flag_registry.reset()

    configure(feature_flags={"optimizers": {"optuna": {"enabled": True}}})
    assert is_optuna_enabled() is True
