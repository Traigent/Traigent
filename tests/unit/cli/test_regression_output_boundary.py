"""Output from user code and the SDK is contained, counted and never on stdout."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from tests.unit.cli.regression_run_helpers import (
    MODULE_SOURCE,
    invoke,
    make_row,
    write_rows,
)
from traigent.utils.output_boundary import capture_output

PLAIN_SOURCE = MODULE_SOURCE.replace('"ok"', '"EXPECTED-SECRET-ANSWER"')

SECRETS = [
    "IMPORT-STDOUT-SECRET",
    "IMPORT-STDERR-SECRET",
    "IMPORT-LOG-SECRET",
    "IMPORT-SDK-LOG-SECRET",
    "IMPORT-WARNING-SECRET",
    "IMPORT-FD-SECRET",
    "FUNC-STDOUT-SECRET",
    "FUNC-LOG-SECRET",
    "FUNC-SUBPROCESS-SECRET",
]

NOISY_SOURCE = PLAIN_SOURCE.replace(
    "import sys\nimport traigent\n",
    "import logging, os, subprocess, sys, warnings\nimport traigent\n"
    'print("IMPORT-STDOUT-SECRET")\n'
    'print("IMPORT-STDERR-SECRET", file=sys.stderr)\n'
    'logging.getLogger("user.module").warning("IMPORT-LOG-SECRET")\n'
    'logging.getLogger("traigent").warning("IMPORT-SDK-LOG-SECRET")\n'
    'warnings.warn("IMPORT-WARNING-SECRET")\n'
    'os.write(1, b"IMPORT-FD-SECRET\\n")\n',
).replace(
    'return "EXPECTED-SECRET-ANSWER" if q.startswith("pass") else "wrong"',
    'print("FUNC-STDOUT-SECRET", q)\n'
    '    logging.getLogger("user.module").error("FUNC-LOG-SECRET %s", q)\n'
    '    subprocess.run(["echo", "FUNC-SUBPROCESS-SECRET"], check=False)\n'
    '    return "EXPECTED-SECRET-ANSWER" if q.startswith("pass") else "wrong"',
)

METRIC_SOURCE = PLAIN_SOURCE.replace(
    "    objectives=",
    "    metric_functions={'accuracy': _acc},\n    objectives=",
).replace(
    "@traigent.optimize(",
    "def _acc(output, expected_output=None, **kwargs):\n"
    "    raise ValueError(expected_output)\n\n\n@traigent.optimize(",
)


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text(PLAIN_SOURCE, encoding="utf-8")
    write_rows(tmp_path, [make_row("P", {"q": "pass"}, "EXPECTED-SECRET-ANSWER")])
    return tmp_path


def _armed(ws: Path) -> None:
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 0


def _noisy(ws: Path, source: str = NOISY_SOURCE) -> None:
    (ws / "agent.py").write_text(source, encoding="utf-8")


def _assert_no_secret(text: str) -> None:
    leaked = [secret for secret in SECRETS if secret in text]
    assert not leaked, leaked


def test_check_json_stays_one_object_and_nothing_leaks_by_default(
    ws: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    _armed(ws)
    _noisy(ws)
    capfd.readouterr()
    result = invoke(ws, "check", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)  # one JSON object and nothing else
    assert payload["verdict"] == "pass"
    _assert_no_secret(result.stdout)
    _assert_no_secret(result.stderr)
    fd_out, fd_err = capfd.readouterr()
    _assert_no_secret(fd_out + fd_err)
    assert (
        " lines of output from your code and the SDK were suppressed; rerun with --show-output to see them on stderr"
        in result.stderr
    )


def test_confirm_also_contains_output_and_prints_the_count_line(
    ws: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    _noisy(ws)
    capfd.readouterr()
    result = invoke(ws, "confirm", "--repeats", "2")
    assert result.exit_code == 0, result.output
    _assert_no_secret(result.stdout + result.stderr)
    _assert_no_secret("".join(capfd.readouterr()))
    assert "suppressed; rerun with --show-output" in result.stderr


def test_show_output_goes_to_stderr_only(
    ws: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    _armed(ws)
    _noisy(ws)
    capfd.readouterr()
    result = invoke(ws, "check", "--json", "--show-output")
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)
    _assert_no_secret(result.stdout)
    for secret in SECRETS:
        assert secret in result.stderr, secret
    assert "suppressed" not in result.stderr
    _assert_no_secret("".join(capfd.readouterr()))


def test_show_output_help_warns_about_example_content() -> None:
    from click.testing import CliRunner

    from traigent.cli.main import cli

    for verb in ("check", "confirm"):
        text = CliRunner().invoke(cli, ["regression", verb, "--help"]).output
        assert "--show-output" in text and "example content" in " ".join(text.split())


def test_a_metric_error_message_that_is_the_expected_output_does_not_leak(
    ws: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    _armed(ws)
    _noisy(ws, METRIC_SOURCE)
    capfd.readouterr()
    result = invoke(ws, "check", "--json")
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["verdict"] in {"error", "inconclusive"}
    everything = result.stdout + result.stderr + "".join(capfd.readouterr())
    assert "EXPECTED-SECRET-ANSWER" not in everything
    confirm = invoke(ws, "confirm", "--repeats", "2")
    assert "EXPECTED-SECRET-ANSWER" not in confirm.stdout + confirm.stderr
    shown = invoke(ws, "check", "--json", "--show-output")
    assert "EXPECTED-SECRET-ANSWER" in shown.stderr  # opt-in, stderr only
    assert "EXPECTED-SECRET-ANSWER" not in shown.stdout


def test_an_exiting_noisy_module_still_prints_one_json_object(ws: Path) -> None:
    _armed(ws)
    _noisy(
        ws,
        NOISY_SOURCE.replace("import traigent\n", "import traigent\nsys.exit(0)\n", 1),
    )
    result = invoke(ws, "check", "--json")
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"] == "module_load_failed"
    _assert_no_secret(result.stdout + result.stderr)


# ----------------------------------------------------------- the boundary itself


def _state() -> tuple[object, ...]:
    root = logging.getLogger()
    named = logging.getLogger("traigent")
    return (
        sys.stdout,
        sys.stderr,
        list(root.handlers),
        root.level,
        list(named.handlers),
        named.level,
        named.propagate,
        list(warnings.filters),
    )


def _noise() -> None:
    print("out")
    print("err", file=sys.stderr)
    logging.getLogger("traigent").warning("sdk log")
    warnings.warn("careful")
    os.write(1, b"fd out\n")
    subprocess.run(["echo", "child"], check=False)


def test_the_boundary_counts_and_restores_after_success() -> None:
    logging.getLogger("traigent").addHandler(logging.NullHandler())
    before = _state()
    with capture_output() as captured:
        _noise()
    assert _state() == before
    assert captured.line_count >= 6 and captured.fd_capture
    text = captured.text()
    for needle in ("out", "err", "sdk log", "careful", "fd out", "child"):
        assert needle in text


def test_the_boundary_restores_after_an_exception() -> None:
    before = _state()
    with pytest.raises(RuntimeError):
        with capture_output():
            _noise()
            raise RuntimeError("boom")
    assert _state() == before


def test_the_boundary_restores_after_system_exit() -> None:
    before = _state()
    with pytest.raises(SystemExit):
        with capture_output():
            _noise()
            sys.exit(0)
    assert _state() == before


def test_the_boundary_restores_streams_a_user_replaced_and_handlers_it_added() -> None:
    before = _state()
    with capture_output():
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        logging.getLogger().handlers = []
    assert _state() == before


def test_descriptor_output_is_restored_for_later_writes(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with capture_output():
        os.write(1, b"inside\n")
    os.write(1, b"after\n")
    out = capfd.readouterr().out
    assert "after" in out and "inside" not in out
