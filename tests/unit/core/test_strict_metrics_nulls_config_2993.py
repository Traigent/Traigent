"""TraigentConfig.strict_metrics_nulls must take effect without the env var (#2993)."""

from __future__ import annotations

import pytest

from traigent.api.functions import _apply_config_settings
from traigent.config.types import TraigentConfig
from traigent.core.utils import (
    safe_float_convert,
    set_configured_strict_metrics_nulls,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("TRAIGENT_STRICT_METRICS_NULLS", raising=False)
    set_configured_strict_metrics_nulls(False)
    yield
    set_configured_strict_metrics_nulls(False)


def test_config_flag_alone_turns_on_strict_nulls() -> None:
    assert safe_float_convert(None) == 0.0
    _apply_config_settings(TraigentConfig(strict_metrics_nulls=True))
    assert safe_float_convert(None) is None


def test_reapplying_a_config_without_the_flag_turns_it_off() -> None:
    _apply_config_settings(TraigentConfig(strict_metrics_nulls=True))
    _apply_config_settings(TraigentConfig(strict_metrics_nulls=False))
    assert safe_float_convert(None) == 0.0


def test_env_var_still_works(monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_METRICS_NULLS", "true")
    assert safe_float_convert(None) is None


def test_status_line_does_not_claim_automatic_sync(monkeypatch) -> None:
    for name in (
        "TRAIGENT_OFFLINE",
        "TRAIGENT_OFFLINE_MODE",
        "TRAIGENT_EDGE_ANALYTICS_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    from traigent.cli.local_commands import _portal_sync_status

    status = _portal_sync_status(TraigentConfig(auto_sync=True))
    assert "not implemented" in status
    assert "automatic for eligible" not in status


@pytest.mark.parametrize("raw", [" true", "on", "ON", "yes", "1", "True "])
def test_env_var_uses_shared_boolean_grammar(monkeypatch, raw: str) -> None:
    from traigent.core.utils import strict_metrics_nulls_enabled

    monkeypatch.setenv("TRAIGENT_STRICT_METRICS_NULLS", raw)
    assert strict_metrics_nulls_enabled() is True


@pytest.mark.parametrize("raw", ["", "off", "0", "false", "maybe"])
def test_env_var_falsy_values_stay_off(monkeypatch, raw: str) -> None:
    from traigent.core.utils import strict_metrics_nulls_enabled

    monkeypatch.setenv("TRAIGENT_STRICT_METRICS_NULLS", raw)
    assert strict_metrics_nulls_enabled() is False
