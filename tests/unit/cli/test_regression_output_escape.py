"""Buffered output and failing formatters cannot get past the output boundary.

Each test runs the real CLI in a subprocess with stdout and stderr as pipes:
an in-process runner replaces ``sys.stdout`` and cannot show what reaches the
real descriptors.
"""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import pytest

from tests.unit.cli.regression_run_helpers import MODULE_SOURCE
from tests.unit.cli.regression_subprocess_helpers import make_workspace, run_cli

MARK_PY = "MARKER-NO-NEWLINE-PY"
MARK_C = "MARKER-NO-NEWLINE-C"

_LIBC_OK = sys.platform != "win32"
if _LIBC_OK:
    try:
        _LIBC_OK = callable(ctypes.CDLL(None).fflush)
    except (OSError, AttributeError):
        _LIBC_OK = False


def _source(body: str) -> str:
    return MODULE_SOURCE.replace(
        '    return "ok" if q.startswith("pass") else "wrong"',
        body + '\n    return "ok" if q.startswith("pass") else "wrong"',
    )


BUFFERED_PY = _source(f'    sys.__stdout__.write("{MARK_PY}")')
BUFFERED_C = _source(
    "    import ctypes\n"
    f'    ctypes.CDLL(None).printf(b"{MARK_C}")'  # no newline, C stdio buffer
)


def _arm(root: Path) -> None:
    done = run_cli(root, "confirm", "--repeats", "2")
    assert done.returncode == 0, done.stderr[-500:]


def test_python_buffered_write_to_the_real_stdout_does_not_escape(
    tmp_path: Path,
) -> None:
    make_workspace(tmp_path, BUFFERED_PY)
    _arm(tmp_path)
    plain = run_cli(tmp_path, "check")
    assert plain.returncode == 0, plain.stderr[-500:]
    assert MARK_PY not in plain.stdout + plain.stderr
    as_json = run_cli(tmp_path, "check", "--json")
    assert as_json.returncode == 0, as_json.stderr[-500:]
    assert MARK_PY not in as_json.stdout + as_json.stderr
    assert json.loads(as_json.stdout)["verdict"] == "pass"  # exactly one object


@pytest.mark.skipif(not _LIBC_OK, reason="needs libc stdio through ctypes")
def test_c_stdio_buffered_write_does_not_escape(tmp_path: Path) -> None:
    make_workspace(tmp_path, BUFFERED_C)
    _arm(tmp_path)
    plain = run_cli(tmp_path, "check")
    assert plain.returncode == 0, plain.stderr[-500:]
    assert MARK_C not in plain.stdout + plain.stderr
    as_json = run_cli(tmp_path, "check", "--json")
    assert as_json.returncode == 0, as_json.stderr[-500:]
    assert MARK_C not in as_json.stdout + as_json.stderr
    assert json.loads(as_json.stdout)["verdict"] == "pass"


def test_show_output_still_reaches_stderr_only(tmp_path: Path) -> None:
    make_workspace(tmp_path, BUFFERED_PY)
    _arm(tmp_path)
    shown = run_cli(tmp_path, "check", "--json", "--show-output")
    assert shown.returncode == 0
    assert MARK_PY not in shown.stdout
    assert json.loads(shown.stdout)["verdict"] == "pass"
    assert MARK_PY in shown.stderr  # opt-in, stderr only


# ---------------------------------------------- cleanup cannot be skipped

RAISING_FORMATWARNING = _source(
    "    import warnings\n"
    "    def _bad(*args, **kwargs):\n"
    '        raise RuntimeError("formatter-secret")\n'
    "    warnings.formatwarning = _bad\n"
    '    warnings.warn("a warning")'
)
RAISING_LOG_FORMAT = _source(
    "    import logging\n"
    "    class _Handler(logging.StreamHandler):\n"
    "        def format(self, record):\n"
    '            raise RuntimeError("log-format-secret")\n'
    "    logging.getLogger().addHandler(_Handler())\n"
    '    logging.getLogger("traigent").addHandler(_Handler())\n'
    '    logging.getLogger("user.module").warning("a log line")\n'
    '    logging.getLogger("traigent").warning("an sdk log line")'
)
RAISING_FLUSH = _source(
    "    class _Out:\n"
    "        def write(self, text):\n"
    "            return len(text)\n"
    "        def flush(self):\n"
    '            raise RuntimeError("flush-secret")\n'
    "    sys.stdout = _Out()\n"
    "    sys.stderr = _Out()\n"
    '    print("some text")'
)
RAISING_DUNDER_FLUSH = RAISING_FLUSH.replace(
    "    sys.stdout = _Out()\n    sys.stderr = _Out()\n",
    "    sys.__stdout__ = _Out()\n    sys.__stderr__ = _Out()\n",
)
HAZARDS = {
    "dunder_flush": RAISING_DUNDER_FLUSH,
    "formatwarning": RAISING_FORMATWARNING,
    "log_format": RAISING_LOG_FORMAT,
    "flush": RAISING_FLUSH,
}


@pytest.mark.parametrize("hazard", sorted(HAZARDS))
def test_a_failing_formatter_or_flush_cannot_skip_the_restore(
    tmp_path: Path, hazard: str
) -> None:
    make_workspace(tmp_path, HAZARDS[hazard])
    confirm = run_cli(tmp_path, "confirm", "--repeats", "2")
    assert confirm.returncode == 0, confirm.stderr[-800:]
    assert "armed" in confirm.stdout  # the command's own result reached stdout
    plain = run_cli(tmp_path, "check")
    assert plain.returncode == 0, plain.stderr[-800:]
    assert "1 armed row(s): 1 passed" in plain.stdout
    as_json = run_cli(tmp_path, "check", "--json")
    assert as_json.returncode == 0, as_json.stderr[-800:]
    assert json.loads(as_json.stdout)["verdict"] == "pass"
    for secret in ("formatter-secret", "log-format-secret", "flush-secret"):
        assert secret not in plain.stdout + plain.stderr + as_json.stdout
        assert secret not in as_json.stderr


@pytest.mark.parametrize("hazard", sorted(HAZARDS))
def test_a_failing_check_exit_code_survives_the_hazard(
    tmp_path: Path, hazard: str
) -> None:
    """A regression is still exit 1 when cleanup hits a hazard."""
    make_workspace(tmp_path, HAZARDS[hazard])
    assert run_cli(tmp_path, "confirm", "--repeats", "2").returncode == 0
    (tmp_path / "agent.py").write_text(
        HAZARDS[hazard].replace('q.startswith("pass")', "False"), encoding="utf-8"
    )
    result = run_cli(tmp_path, "check", "--json")
    assert result.returncode == 1, result.stderr[-800:]
    assert json.loads(result.stdout)["verdict"] == "regression"


def test_in_process_a_raising_formatwarning_still_restores_descriptors(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import warnings

    from traigent.utils.output_boundary import capture_output

    def bad(*args: object, **kwargs: object) -> str:
        raise RuntimeError("formatter-secret")

    monkeypatch.setattr(warnings, "formatwarning", bad)
    with capture_output() as captured:
        warnings.warn("careful", stacklevel=2)
        os.write(1, b"inside\n")
    os.write(1, b"after\n")
    out = capfd.readouterr().out
    assert "after" in out and "inside" not in out
    assert captured.warning_lines == ["(a warning could not be formatted)"]
    assert "formatter-secret" not in captured.text()
    assert captured.line_count >= 2  # the note is in the count


def test_in_process_descriptors_come_back_even_if_flushing_blows_up(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from traigent.utils import output_boundary

    calls: list[int] = []

    def broken() -> None:
        calls.append(1)
        if len(calls) > 1:  # the closing flush, not the opening one
            raise RuntimeError("flush went wrong")

    monkeypatch.setattr(output_boundary, "_flush_everything", broken)
    with pytest.raises(RuntimeError):
        with output_boundary.capture_output():
            os.write(1, b"inside\n")
    os.write(1, b"after\n")
    out = capfd.readouterr().out
    assert len(calls) == 2
    assert "after" in out and "inside" not in out


# ------------------------------------------- check writes nothing, state is back


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_check_writes_no_file_anywhere_in_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    make_workspace(workspace)
    _arm(workspace)
    for pyc in workspace.rglob("*.pyc"):  # what a fresh CI checkout lacks
        pyc.unlink()
    before = _files(workspace)
    for argv in (("check",), ("check", "--json")):
        result = run_cli(workspace, *argv)
        assert result.returncode == 0, result.stderr[-500:]
    assert _files(workspace) == before  # no pyc, no lock, no temp file
    assert not list(workspace.rglob("*.pyc"))


def test_check_puts_the_bytecode_setting_and_open_descriptors_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from click.testing import CliRunner

    from traigent.cli.main import cli

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    make_workspace(tmp_path)
    args = [
        "regression",
        "check",
        "--module",
        "agent.py:answer",
        "--config",
        "config.json",
        "--dir",
        "reg",
    ]
    confirm = ["regression", "confirm", *args[2:]]
    assert CliRunner().invoke(cli, [*confirm, "--repeats", "2"]).exit_code == 0
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    fds_before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self") else 0
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert sys.dont_write_bytecode is False
    if fds_before:
        assert len(os.listdir("/proc/self/fd")) <= fds_before
