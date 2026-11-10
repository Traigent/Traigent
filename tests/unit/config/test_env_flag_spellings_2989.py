"""Security-relevant TRAIGENT_* flags share one boolean grammar (#2989)."""

from __future__ import annotations

import pytest

from traigent.config.types import TraigentConfig, is_traigent_disabled
from traigent.security import credentials
from traigent.security.credentials import SecurityLevel, get_secure_credential_store

TRUE_SPELLINGS = ["true", " true", "True", "1", "yes", "on", "yes\n"]
FALSE_SPELLINGS = ["", "0", "False", "off"]


@pytest.mark.parametrize("value", TRUE_SPELLINGS)
def test_disabled_flag_accepts_all_true_spellings(monkeypatch, value) -> None:
    monkeypatch.setenv("TRAIGENT_DISABLED", value)
    assert is_traigent_disabled() is True


@pytest.mark.parametrize("value", FALSE_SPELLINGS)
def test_disabled_flag_false_spellings(monkeypatch, value) -> None:
    monkeypatch.setenv("TRAIGENT_DISABLED", value)
    assert is_traigent_disabled() is False


@pytest.mark.parametrize("value", TRUE_SPELLINGS)
def test_edge_analytics_mode_forces_offline_for_all_true_spellings(
    monkeypatch, value
) -> None:
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_EDGE_ANALYTICS_MODE", value)
    assert TraigentConfig.from_environment().offline is True


@pytest.mark.parametrize("value", ["production", "production\n", " Production", "prod"])
def test_credential_store_production_spellings_get_maximum(monkeypatch, value) -> None:
    for key in ("ENVIRONMENT", "TRAIGENT_ENV"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", value)
    captured: dict[str, SecurityLevel] = {}

    def fake_store(*, security_level, **_kwargs):
        captured["level"] = security_level

    monkeypatch.setattr(credentials, "EnhancedCredentialStore", fake_store)
    get_secure_credential_store()
    assert captured["level"] == SecurityLevel.MAXIMUM
