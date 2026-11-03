"""Characterization tests pinning IntegrationPlugin.apply_overrides."""

from __future__ import annotations

import logging

import pytest

from traigent.config.types import TraigentConfig
from traigent.integrations import base_plugin
from traigent.integrations.base_plugin import IntegrationPlugin, PluginMetadata

MODULE_LOGGER = "traigent.integrations.base_plugin"


class _Plugin(IntegrationPlugin):
    def _get_metadata(self):
        return PluginMetadata(name="demo", version="1", supported_packages=[])

    def _get_default_mappings(self):
        return {
            "model": "model_name",
            "temperature": "temp",
            "system": "system",
            "system_prompt": "sys_prompt",
        }

    def _get_validation_rules(self):
        return {}

    def get_target_classes(self):
        return []

    def get_target_methods(self):
        return {}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(base_plugin, "_system_prompt_warning_emitted", False)
    monkeypatch.delenv("TRAIGENT_LOCK_SYSTEM_PROMPT", raising=False)


def test_disabled_plugin_returns_same_kwargs_object():
    plugin = _Plugin()
    plugin._enabled = False
    kwargs = {"a": 1}
    assert plugin.apply_overrides(kwargs, TraigentConfig(model="m")) is kwargs


def test_mapped_params_applied_and_input_not_mutated():
    kwargs = {"x": 1}
    out = _Plugin().apply_overrides(kwargs, TraigentConfig(model="m", temperature=0.2))
    assert out == {"x": 1, "model_name": "m", "temp": 0.2}
    assert kwargs == {"x": 1}


def test_explicit_framework_param_wins_over_mapping():
    out = _Plugin().apply_overrides({"model_name": "mine"}, TraigentConfig(model="m"))
    assert out == {"model_name": "mine"}


def test_none_attrs_are_skipped():
    out = _Plugin().apply_overrides({}, TraigentConfig(model="m"))
    assert out == {"model_name": "m"}


def test_custom_params_mapped_and_passed_through():
    cfg = TraigentConfig(
        custom_params={"temperature": 0.7, "native": 5, "model_name": 1}
    )
    out = _Plugin().apply_overrides({"native": "kw"}, cfg)
    # mapped via mapping; native does not clobber existing kwarg; model_name is a
    # framework-native passthrough
    assert out == {"native": "kw", "temp": 0.7, "model_name": 1}


def test_passthrough_does_not_duplicate_mapped_param():
    cfg = TraigentConfig(custom_params={"temperature": 0.7})
    out = _Plugin().apply_overrides({"temp": "kw"}, cfg)
    assert out == {"temp": "kw"}


def test_system_prompt_param_is_audited_and_applied(caplog):
    cfg = TraigentConfig(custom_params={"system": "short"})
    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        out = _Plugin().apply_overrides({}, cfg)
    assert out["system"] == "short"
    messages = [r.getMessage() for r in caplog.records]
    assert any("included in optimization config space" in m for m in messages)
    assert "Optimizer is modifying system prompt param 'system': short" in messages
    assert base_plugin._system_prompt_warning_emitted is True


def test_info_logged_once_but_audit_each_time(caplog):
    cfg = TraigentConfig(custom_params={"system": "s"})
    plugin = _Plugin()
    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        plugin.apply_overrides({}, cfg)
        plugin.apply_overrides({}, cfg)
    messages = [r.getMessage() for r in caplog.records]
    assert sum("included in optimization" in m for m in messages) == 1
    assert sum("Optimizer is modifying" in m for m in messages) == 2


def test_long_system_prompt_truncated_in_audit(caplog):
    value = "x" * 130
    cfg = TraigentConfig(custom_params={"system": value})
    with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
        _Plugin().apply_overrides({}, cfg)
    msg = next(
        r.getMessage()
        for r in caplog.records
        if "Optimizer is modifying" in r.getMessage()
    )
    assert (
        msg == "Optimizer is modifying system prompt param 'system': " + "x" * 120 + "…"
    )


def test_exactly_120_chars_not_truncated(caplog):
    value = "y" * 120
    cfg = TraigentConfig(custom_params={"system": value})
    with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
        _Plugin().apply_overrides({}, cfg)
    msg = next(
        r.getMessage()
        for r in caplog.records
        if "Optimizer is modifying" in r.getMessage()
    )
    assert msg.endswith(value) and "…" not in msg


@pytest.mark.parametrize("flag", ["true", "1", "YES", "True"])
def test_lock_drops_mapped_prompt_param_but_logs(monkeypatch, caplog, flag):
    monkeypatch.setenv("TRAIGENT_LOCK_SYSTEM_PROMPT", flag)
    cfg = TraigentConfig(custom_params={"system_prompt": "evil"})
    with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
        out = _Plugin().apply_overrides({}, cfg)
    # dropped from the mapping step; the custom_params passthrough still applies
    assert "sys_prompt" not in out
    assert out == {"system_prompt": "evil"}
    assert any(
        "TRAIGENT_LOCK_SYSTEM_PROMPT is set" in r.getMessage()
        and "'system_prompt'" in r.getMessage()
        for r in caplog.records
    )
    assert not any("Optimizer is modifying" in r.getMessage() for r in caplog.records)


def test_lock_flag_off_values_keep_audit_mode(monkeypatch, caplog):
    monkeypatch.setenv("TRAIGENT_LOCK_SYSTEM_PROMPT", "no")
    cfg = TraigentConfig(custom_params={"system_prompt": "ok"})
    with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
        out = _Plugin().apply_overrides({}, cfg)
    assert out["sys_prompt"] == "ok"
    assert any("Optimizer is modifying" in r.getMessage() for r in caplog.records)


def test_no_prompt_params_no_logging(caplog):
    with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER):
        _Plugin().apply_overrides({}, TraigentConfig(model="m"))
    assert base_plugin._system_prompt_warning_emitted is False
    assert not caplog.records


def test_dict_config_is_normalised():
    out = _Plugin().apply_overrides({}, {"model": "m", "custom_params": {"k": 1}})
    assert out == {"model_name": "m", "k": 1}
