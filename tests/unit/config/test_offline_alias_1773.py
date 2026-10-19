"""#1773: TRAIGENT_OFFLINE and TRAIGENT_OFFLINE_MODE both request offline."""

from __future__ import annotations

import pytest

from traigent.config.types import TraigentConfig, is_traigent_offline_requested


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("TRAIGENT_OFFLINE", raising=False)
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)


@pytest.mark.parametrize("name", ["TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"])
def test_either_alias_requests_offline(monkeypatch, name):
    monkeypatch.setenv(name, "true")
    assert is_traigent_offline_requested() is True
    assert TraigentConfig.from_environment().offline is True


def test_no_alias_is_not_offline():
    assert is_traigent_offline_requested() is False


def test_falsy_values_do_not_request_offline(monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "0")
    assert is_traigent_offline_requested() is False
