"""Characterization tests pinning IntegrationPlugin._load_config_overrides."""

from __future__ import annotations

import json
from unittest.mock import patch

from traigent.integrations import base_plugin as bp
from traigent.integrations.base_plugin import (
    IntegrationPlugin,
    PluginMetadata,
    ValidationRule,
)


class _Plugin(IntegrationPlugin):
    def _get_metadata(self):
        return PluginMetadata(name="demo", version="1", supported_packages=[])

    def _get_default_mappings(self):
        return {"temperature": "temp"}

    def _get_validation_rules(self):
        return {"temp": ValidationRule(required=True)}

    def get_target_classes(self):
        return []

    def get_target_methods(self):
        return {}


def _load(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    plugin = _Plugin()
    with patch.object(bp, "logger") as lg:
        plugin._load_config_overrides(path)
    return plugin, lg, path


def test_missing_file_warns_and_returns(tmp_path):
    plugin = _Plugin()
    path = tmp_path / "nope.yaml"
    with patch.object(bp, "logger") as lg:
        plugin._load_config_overrides(path)
    lg.warning.assert_called_once_with(f"Configuration file not found: {path}")
    lg.info.assert_not_called()
    lg.error.assert_not_called()


def test_yaml_full_override(tmp_path):
    text = (
        "mappings:\n  top_p: nucleus\n"
        "validation:\n  temp:\n    required: false\n    max_value: 2.0\n    bogus: 1\n"
        "  fresh:\n    min_value: 1\n"
        "metadata:\n  version: '9'\n  nonexistent: 1\n"
    )
    plugin, lg, path = _load(tmp_path, "c.yaml", text)
    assert plugin._parameter_mappings["top_p"] == "nucleus"
    assert plugin._parameter_mappings["temperature"] == "temp"
    rule = plugin._validation_rules["temp"]
    assert (rule.required, rule.max_value) == (False, 2.0)
    assert not hasattr(rule, "bogus")
    assert plugin._validation_rules["fresh"].min_value == 1
    assert plugin.metadata.version == "9"
    assert not hasattr(plugin.metadata, "nonexistent")
    lg.info.assert_called_once_with(f"Loaded configuration overrides from {path}")
    lg.error.assert_not_called()


def test_yml_and_uppercase_suffix(tmp_path):
    plugin, lg, _ = _load(tmp_path, "c.YML", "mappings:\n  a: b\n")
    assert plugin._parameter_mappings["a"] == "b"
    lg.info.assert_called_once()


def test_json_override(tmp_path):
    plugin, lg, _ = _load(
        tmp_path,
        "c.json",
        json.dumps({"mappings": {"x": "y"}, "metadata": {"version": "3"}}),
    )
    assert plugin._parameter_mappings["x"] == "y"
    assert plugin.metadata.version == "3"
    lg.info.assert_called_once()


def test_unsupported_suffix_warns_without_info(tmp_path):
    plugin, lg, path = _load(tmp_path, "c.txt", "mappings: {a: b}")
    lg.warning.assert_called_once_with(f"Unsupported config file format: {path}")
    lg.info.assert_not_called()
    lg.error.assert_not_called()
    assert "a" not in plugin._parameter_mappings


def test_empty_yaml_logs_error_not_raises(tmp_path):
    # yaml.safe_load returns None -> `"mappings" in None` raises TypeError.
    plugin, lg, path = _load(tmp_path, "c.yaml", "")
    assert lg.error.call_count == 1
    assert lg.error.call_args.args[0].startswith(
        f"Failed to load configuration from {path}: "
    )
    lg.info.assert_not_called()


def test_malformed_json_logs_error(tmp_path):
    _, lg, path = _load(tmp_path, "c.json", "{not json")
    assert lg.error.call_args.args[0].startswith(
        f"Failed to load configuration from {path}: "
    )


def test_error_midway_keeps_earlier_overrides(tmp_path):
    text = "mappings:\n  a: b\nvalidation: 5\n"
    plugin, lg, _ = _load(tmp_path, "c.yaml", text)
    assert plugin._parameter_mappings["a"] == "b"
    assert lg.error.call_count == 1
    lg.info.assert_not_called()


def test_sections_absent_still_logs_loaded(tmp_path):
    _, lg, path = _load(tmp_path, "c.yaml", "other: 1\n")
    lg.info.assert_called_once_with(f"Loaded configuration overrides from {path}")
