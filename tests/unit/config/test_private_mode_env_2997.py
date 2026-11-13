"""#2997: TRAIGENT_PRIVATE_DATA / TRAIGENT_COMPLIANCE_MODE force local-only execution."""

from __future__ import annotations

import pytest

from traigent.config.types import (
    ExecutionIntent,
    is_traigent_private_mode_requested,
    resolve_execution_policy,
)
from traigent.utils.exceptions import ConfigurationError

_NAMES = ("TRAIGENT_PRIVATE_DATA", "TRAIGENT_COMPLIANCE_MODE")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (*_NAMES, "TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TRAIGENT_REQUIRE_CLOUD", raising=False)


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "banana", "2", "enabled"])
def test_set_value_forces_local_only(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert is_traigent_private_mode_requested() is True
    policy = resolve_execution_policy(algorithm="auto")
    assert policy.intent is ExecutionIntent.LOCAL_ONLY
    assert policy.offline is True


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  "])
def test_off_values_do_not_force_local(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert is_traigent_private_mode_requested() is False
    assert resolve_execution_policy(algorithm="auto").intent is (
        ExecutionIntent.CLOUD_BRAIN
    )


def test_unset_does_not_force_local():
    assert is_traigent_private_mode_requested() is False


@pytest.mark.parametrize("name", _NAMES)
def test_explicit_cloud_requests_raise(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(ConfigurationError, match="local-only"):
        resolve_execution_policy(algorithm="bayesian")
    with pytest.raises(ConfigurationError, match="local-only"):
        resolve_execution_policy(algorithm="auto", require_cloud=True)
    with pytest.raises(ConfigurationError, match="local-only"):
        resolve_execution_policy(algorithm="auto", execution_mode="hybrid")


def test_unparseable_value_fails_closed_for_cloud_request(monkeypatch):
    monkeypatch.setenv("TRAIGENT_COMPLIANCE_MODE", "maybe")
    with pytest.raises(ConfigurationError):
        resolve_execution_policy(algorithm="optuna_tpe")


def test_local_algorithms_still_work(monkeypatch):
    monkeypatch.setenv("TRAIGENT_PRIVATE_DATA", "1")
    assert resolve_execution_policy(algorithm="grid").offline is True


def test_cwd_private_file_is_not_a_trigger(monkeypatch, tmp_path):
    (tmp_path / ".traigent-private").write_text("")
    monkeypatch.chdir(tmp_path)
    assert is_traigent_private_mode_requested() is False
    assert resolve_execution_policy(algorithm="auto").intent is (
        ExecutionIntent.CLOUD_BRAIN
    )
