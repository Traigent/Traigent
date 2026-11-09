"""#3267: one environment resolver; the stricter value wins on disagreement."""

from __future__ import annotations

import logging

import pytest

from traigent.identity.workload import resolve_environment
from traigent.utils import env_config
from traigent.utils.env_config import (
    is_production_strict_env,
    resolve_environment_label,
    resolve_environment_name,
    treat_as_production_policy,
)

_KEYS = ("TRAIGENT_ENVIRONMENT", "ENVIRONMENT", "TRAIGENT_ENV")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in (*_KEYS, "OTEL_RESOURCE_ATTRIBUTES"):
        monkeypatch.delenv(key, raising=False)
    getattr(env_config, "_warned_env_disagreements", set()).clear()


def _set(monkeypatch, **values):
    for key, value in values.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize(
    "values",
    [
        {"TRAIGENT_ENVIRONMENT": "production", "ENVIRONMENT": "development"},
        {"TRAIGENT_ENVIRONMENT": "development", "ENVIRONMENT": "production"},
        {"TRAIGENT_ENVIRONMENT": "production", "TRAIGENT_ENV": "dev"},
        {"ENVIRONMENT": "dev", "TRAIGENT_ENV": "production"},
    ],
)
def test_disagreement_is_production_everywhere(monkeypatch, values):
    _set(monkeypatch, **values)
    assert resolve_environment_name() == "production"
    assert resolve_environment_label() == "production"
    assert treat_as_production_policy() is True
    assert is_production_strict_env() is True
    if values.get("TRAIGENT_ENVIRONMENT") == "production":
        # the product variable wins the telemetry ladder, so it agrees too
        assert resolve_environment().environment == "production"


def test_staging_beats_development_on_disagreement(monkeypatch):
    _set(monkeypatch, TRAIGENT_ENVIRONMENT="development", ENVIRONMENT="staging")
    assert resolve_environment_name() == "staging"


def test_unknown_value_disagreement_fails_closed(monkeypatch):
    _set(monkeypatch, TRAIGENT_ENVIRONMENT="development", ENVIRONMENT="banana")
    assert treat_as_production_policy() is True


def test_production_outranks_unknown_value(monkeypatch):
    _set(monkeypatch, TRAIGENT_ENVIRONMENT="banana", ENVIRONMENT="production")
    assert resolve_environment_name() == "production"
    assert is_production_strict_env() is True


@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize(
    ("value", "policy"), [("production", True), ("development", False)]
)
def test_single_variable_unchanged(monkeypatch, key, value, policy):
    _set(monkeypatch, **{key: value})
    assert resolve_environment_name() == value
    assert treat_as_production_policy() is policy
    assert resolve_environment().environment == value


def test_single_traigent_environment_production_is_strict(monkeypatch):
    _set(monkeypatch, TRAIGENT_ENVIRONMENT="production")
    assert is_production_strict_env() is True


@pytest.mark.parametrize(
    "values",
    [
        {"TRAIGENT_ENVIRONMENT": "production", "ENVIRONMENT": "development"},
        {"ENVIRONMENT": "development", "TRAIGENT_ENV": "production"},
    ],
)
def test_mock_mode_is_blocked_when_any_variable_says_production(monkeypatch, values):
    from traigent.testing import enable_mock_mode_for_quickstart

    _set(monkeypatch, **values)
    with pytest.raises(RuntimeError, match="hard-blocked"):
        enable_mock_mode_for_quickstart()
    assert env_config.any_environment_declared_production() is True


def test_legacy_helpers_warn_on_disagreement(monkeypatch, caplog):
    _set(monkeypatch, TRAIGENT_ENVIRONMENT="development", ENVIRONMENT="staging")
    with caplog.at_level(logging.WARNING):
        assert env_config._get_environment_name() == "staging"
    assert any("disagree" in r.getMessage() for r in caplog.records)


def test_agreeing_pair_unchanged_and_silent(monkeypatch, caplog):
    _set(monkeypatch, TRAIGENT_ENVIRONMENT="development", ENVIRONMENT="development")
    with caplog.at_level(logging.WARNING):
        assert resolve_environment_name() == "development"
    assert not caplog.records or "disagree" not in caplog.text


def test_disagreement_warns_once_naming_variables_not_values(monkeypatch, caplog):
    _set(monkeypatch, TRAIGENT_ENVIRONMENT="production", ENVIRONMENT="development")
    with caplog.at_level(logging.WARNING):
        resolve_environment_name()
        resolve_environment_name()
    records = [r for r in caplog.records if "disagree" in r.getMessage()]
    assert len(records) == 1
    text = records[0].getMessage()
    assert "TRAIGENT_ENVIRONMENT" in text and "ENVIRONMENT" in text
    assert "development" not in text.replace("Deployment", "")
