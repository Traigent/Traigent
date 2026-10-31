"""Terminal/log hygiene: control characters never reach echo or logger sinks."""

from __future__ import annotations

import logging

import pytest
from click.testing import CliRunner

from traigent.cli import requirements_commands as cmds
from traigent.cli.main import cli
from traigent.runtime import manifest
from traigent.runtime.model import RuntimeRequirementError, one_line, safe_lines
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
        "a\u061cb",
        "a\u200bb",
        "a\u200eb",
        "a\u200fb",
        "a\u202ab",
        "a\u202eb",
        "a\u2066b",
        "a\u2069b",
        "a\ufeffb",
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


def test_one_line_keeps_joiners() -> None:
    assert one_line("a\u200db\u200cc") == "a\u200db\u200cc"


def test_safe_lines_splits_cleans_and_prefixes_every_line() -> None:
    text = "a\r\nb\nc\rd\u0085e\u2028f\u2029g\x1b[0m\u202eh"
    assert safe_lines(text) == "a\nb\nc\nd\ne\nf\ng [0m h"
    out = safe_lines("one\ntwo\u2028three\r\nfour", "| ")
    assert out == "| one\n| two\n| three\n| four"
    assert safe_lines("") == "" and safe_lines("x", "p:") == "p:x"
    assert safe_lines("a\n\nb") == "a\n\nb"  # empty lines keep structure


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


# --- poisoned fields reach each CLI branch cleaned (P5) -------------------------

BAD = f"x{ESC}[31m\r\nforged{chr(0x202E)}"


def _assert_clean(output: str) -> None:
    assert ESC not in output and "\r" not in output
    assert chr(0x202E) not in output and "\u2028" not in output


def _fake_fn(monkeypatch: pytest.MonkeyPatch, tmp_path):
    class Fn:
        class requirements:  # noqa: N801 - attribute holder
            owner = None

        optimize = staticmethod(lambda: None)

    path = tmp_path / "m.py"
    path.write_text("")
    monkeypatch.setattr(cmds, "_find_function", lambda *_a: Fn())
    monkeypatch.setattr(
        cmds, "_declared", lambda _fn: type("D", (), {"agent_id": "a"})()
    )
    monkeypatch.setattr(cmds, "_ids", lambda *_a: (PROJECT, AGENT))
    monkeypatch.setattr(cmds, "_state_dir", lambda: tmp_path)
    monkeypatch.setattr("traigent.runtime.tvl_fragment.module_name_for", lambda _o: "m")
    return path


def _single_record_lines(output: str, n: int) -> list[str]:
    lines = output.splitlines()
    assert len(lines) == n, lines  # a poisoned field never adds a record
    return lines


def test_apply_success_receipt_fields_are_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    receipt = type(
        "R",
        (),
        {
            "head_id": BAD,
            "revision_id": BAD,
            "revision_number": BAD,
            "provenance": BAD,
            "status": BAD,
        },
    )()
    result = type(
        "Res",
        (),
        {"outcome": BAD, "route": BAD, "policy_digest": BAD * 3, "receipt": receipt},
    )()
    path = _fake_fn(monkeypatch, tmp_path)
    monkeypatch.setattr(cmds, "apply_requirement", lambda *_a, **_k: result)
    out = CliRunner().invoke(
        cli, ["requirements", "apply", str(path), "--function", "f", "--dry-run"]
    )
    assert out.exit_code == 0, out.output
    lines = _single_record_lines(out.output, 2)
    assert all(ln.startswith(("x ", "head=")) for ln in lines)
    _assert_clean(out.output)


def test_apply_conflict_fields_are_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = _fake_fn(monkeypatch, tmp_path)

    def conflict(*_a, **_k):
        raise cmds.ApplyConflict([(BAD, BAD, BAD)])

    monkeypatch.setattr(cmds, "apply_requirement", conflict)
    out = CliRunner().invoke(
        cli, ["requirements", "apply", str(path), "--function", "f", "--dry-run"]
    )
    assert out.exit_code == cmds.EXIT_CONFLICT
    lines = _single_record_lines(out.output, 2)
    assert lines[1].startswith("  x")
    _assert_clean(out.output)


def test_status_fields_are_cleaned(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = _fake_fn(monkeypatch, tmp_path)

    class Api:
        def __init__(self, _p):
            pass

        def list_requirements(self):
            return []

        def list_breaches(self, _a):
            return [], False

    head_row = {
        "head_id": BAD,
        "revision_id": BAD,
        "status": BAD,
        "provenance": BAD,
    }
    monkeypatch.setattr(cmds, "RuntimeClient", Api)
    monkeypatch.setattr(cmds, "matching_heads", lambda *_a: [head_row])
    out = CliRunner().invoke(
        cli, ["requirements", "status", str(path), "--function", "f"]
    )
    assert out.exit_code == 0, out.output
    lines = _single_record_lines(out.output, 2)
    assert lines[0].startswith("head=x ")
    _assert_clean(out.output)


def test_respond_success_and_occurrence_used_are_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = _fake_fn(monkeypatch, tmp_path)
    result = type(
        "Res",
        (),
        {"breach_id": BAD, "occurrence": BAD, "run_id": BAD, "cost_ceiling_usd": BAD},
    )()

    async def ok(*_a, **_k):
        return result

    monkeypatch.setattr(cmds, "run_respond", ok)
    out = CliRunner().invoke(
        cli, ["requirements", "respond", str(path), "--function", "f"]
    )
    assert out.exit_code == 0, out.output
    _single_record_lines(out.output, 1)
    _assert_clean(out.output)

    async def used(*_a, **_k):
        used_exc = cmds.OccurrenceUsed("occurrence_used")
        used_exc.run_id = BAD
        raise used_exc

    monkeypatch.setattr(cmds, "run_respond", used)
    out = CliRunner().invoke(
        cli, ["requirements", "respond", str(path), "--function", "f"]
    )
    assert out.exit_code == 1
    _assert_clean(out.output)
    assert out.output.startswith("occurrence already used: run id x ")


def test_validate_keeps_yaml_structure_and_cleans_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = _fake_fn(monkeypatch, tmp_path)
    tvl = f"a: 1\nb: 2{ESC}\n  c: x\u2028d: y"
    declared = type("D", (), {"to_tvl": lambda self: tvl, "agent_id": "a"})()
    monkeypatch.setattr(cmds, "_declared", lambda _fn: declared)

    class Fn:
        class requirements:  # noqa: N801
            @staticmethod
            def validate():
                return [
                    type("I", (), {"severity": "warning", "code": BAD, "path": BAD})()
                ]

    monkeypatch.setattr(cmds, "_find_function", lambda *_a: Fn())
    out = CliRunner().invoke(
        cli, ["requirements", "validate", str(path), "--function", "f"]
    )
    assert out.exit_code == 0, out.output
    assert out.stdout.splitlines() == ["a: 1", "b: 2 ", "  c: x", "d: y"]
    assert len(out.stderr.splitlines()) == 1 and ESC not in out.stderr
