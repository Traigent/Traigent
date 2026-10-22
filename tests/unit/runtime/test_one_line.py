"""Terminal/log hygiene: control characters never reach echo or logger sinks."""

from __future__ import annotations

import logging

import pytest
from click.testing import CliRunner

from traigent.cli import requirements_commands as cmds
from traigent.cli.main import cli
from traigent.runtime import manifest
from traigent.runtime.model import RuntimeRequirementError, one_line
from traigent.runtime.watch import LOGGER_NAME

from .conftest import AGENT, PROJECT, FakeBackend, breach
from .test_watch import LIST, _declare, _poll

ESC = "\x1b"


@pytest.mark.parametrize(
    "raw",
    [
        "\x1b[31mred\x1b[0m",
        "a\x00b",
        "a\x7fb",
        "a\u0085b",
        "a b",
        "a b",
        "a\rb",
        "a\nb",
        "a\x9bb",
    ],
)
def test_one_line_neutralises_control_characters(raw: str) -> None:
    out = one_line(raw)
    assert len(out) == len(raw)
    assert all(ord(c) >= 0x20 and not 0x7F <= ord(c) <= 0x9F for c in out)
    assert " " not in out and " " not in out


def test_one_line_exact_and_unicode_preserved() -> None:
    assert one_line("\x1b[31mx") == " [31mx"
    assert one_line("a\r\nb") == "a  b"
    assert one_line("café 日本") == "café 日本"
    assert one_line(RuntimeError("bad\x1b[0m")) == "bad [0m"


def test_cli_error_text_is_cleaned(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def boom(*_a, **_k):
        raise RuntimeRequirementError(f"bad {ESC}[31m env\nforged: line")

    monkeypatch.setattr(cmds, "_find_function", boom)
    path = tmp_path / "m.py"
    path.write_text("")
    result = CliRunner().invoke(
        cli, ["requirements", "validate", str(path), "--function", "f"]
    )
    assert result.exit_code == 1
    assert ESC not in result.output and "\n\n" not in result.output
    assert "error: bad  [31m env forged: line" in result.output


def test_cli_explain_cleans_local_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        manifest, "local_for", lambda *_a, **_k: f"prod{ESC}]0;pwned\x07"
    )
    result = CliRunner().invoke(cli, ["requirements", "explain", "ref"])
    assert result.exit_code == 0
    assert ESC not in result.output and "\x07" not in result.output


def test_cli_respond_prompt_facts_are_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cmds.click, "confirm", lambda *_a, **_k: False)

    async def fake_respond(*_a, confirm, **_k):
        confirm({"environment": f"prod{ESC}[2J", f"k{ESC}": "v w"})
        raise RuntimeRequirementError("declined")

    class Fn:
        optimize = staticmethod(lambda: None)

    monkeypatch.setattr(cmds, "run_respond", fake_respond)
    monkeypatch.setattr(cmds, "_find_function", lambda *_a: Fn())
    monkeypatch.setattr(cmds, "_declared", lambda _fn: object())
    monkeypatch.setattr(cmds, "_ids", lambda *_a: (PROJECT, AGENT))
    path = tmp_path / "m.py"
    path.write_text("")
    result = CliRunner().invoke(
        cli, ["requirements", "respond", str(path), "--function", "f"]
    )
    assert "environment: prod [2J" in result.output
    assert ESC not in result.output and " " not in result.output


@pytest.mark.backend_online
def test_watch_logger_gets_cleaned_breach_id(
    backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    raw_id = f"b{ESC}[31m\r\nINFO forged"
    backend.on(*LIST, 200, {"items": [breach(raw_id)], "has_more": False})
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _poll(_declare("alert"))
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "episode_opened" in text and "forged" in text
    assert ESC not in text and "\r" not in text
    assert all("\n" not in r.getMessage() for r in caplog.records)
