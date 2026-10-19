"""Hooks init template only advertises read keys; check warns on unknown keys (#2640)."""

# Traceability: CONC-Quality-Reliability REQ-CONFIG-001

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from traigent.cli.hooks_commands import hooks
from traigent.hooks.config import create_default_config, find_unknown_config_keys

UNREAD = ["min_accuracy", "max_latency_ms", "max_monthly_budget", "environments"]


def test_init_template_has_no_unread_keys(tmp_path: Path) -> None:
    path = create_default_config(tmp_path / "traigent.yml")
    text = path.read_text(encoding="utf-8")
    for key in UNREAD:
        assert key not in text
    assert "traigent-performance" not in text
    assert "traigent-config-check" not in text
    assert find_unknown_config_keys(yaml.safe_load(text)) == []


def test_unknown_keys_with_did_you_mean() -> None:
    found = dict(
        find_unknown_config_keys(
            {"constrains": {"blocked_model": ["gpt-4o"]}, "validation": {"enabled": 1}}
        )
    )
    assert found == {"constrains": "constraints"}
    nested = dict(
        find_unknown_config_keys({"constraints": {"blocked_model": ["gpt-4o"]}})
    )
    assert nested == {"constraints.blocked_model": "constraints.blocked_models"}
    assert dict(find_unknown_config_keys({"zzz": 1})) == {"zzz": None}


def test_non_mapping_is_not_scanned() -> None:
    assert find_unknown_config_keys(None) == []
    assert find_unknown_config_keys({"constraints": ["x"]}) == []


def test_check_warns_without_value_and_exits_zero() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("traigent.yml").write_text(
            "constrains:\n  blocked_model: [SECRET-VALUE-123]\n", encoding="utf-8"
        )
        result = runner.invoke(hooks, ["check", "--quick"])
    assert result.exit_code == 0
    assert "Configuration valid" in result.output
    assert "constrains" in result.output
    assert "constraints" in result.output
    assert "SECRET-VALUE-123" not in result.output
