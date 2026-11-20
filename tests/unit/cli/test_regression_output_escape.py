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
