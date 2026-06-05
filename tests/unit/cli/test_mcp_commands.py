"""Tests for the Traigent MCP CLI command group."""

from __future__ import annotations

import builtins

from click.testing import CliRunner

from traigent.cli.main import cli


def test_mcp_help_does_not_require_optional_dependency() -> None:
    result = CliRunner().invoke(cli, ["mcp", "--help"])

    assert result.exit_code == 0
    assert "serve" in result.output


def test_mcp_serve_without_extra_prints_install_hint(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mcp.server.fastmcp":
            raise ImportError("mcp missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = CliRunner().invoke(cli, ["mcp", "serve"])

    assert result.exit_code != 0
    assert "pip install 'traigent[mcp]'" in result.output
