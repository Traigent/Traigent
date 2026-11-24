"""``traigent egress`` edge paths: refused starts, foreground relay, odd logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

from traigent.cli.egress_commands import RELAY_ENV_VAR, egress
from traigent.egress.log import RelayLog


def _relay(upstream, log_path: Path, *extra: str):
    return CliRunner().invoke(
        egress, ["relay", "--upstream", upstream.url, "--log", str(log_path), *extra]
    )


def test_a_blank_exec_command_is_a_usage_error(upstream, log_path):
    result = _relay(upstream, log_path, "--exec", "   ")
    assert result.exit_code == 2
    assert "--exec needs a command" in result.output
    RelayLog(log_path).close()  # the relay was stopped: its log lock is free


def test_a_command_that_cannot_start_exits_127_without_naming_it(upstream, log_path):
    result = _relay(upstream, log_path, "--exec", "/nonexistent/secret-tool-name")
    assert result.exit_code == 127
    assert "the --exec command could not be started" in result.output
    assert "secret-tool-name" not in result.output  # the error is closed
    RelayLog(log_path).close()


def test_a_child_killed_by_a_signal_exits_128_plus_the_signal(
    upstream, log_path, tmp_path
):
    script = tmp_path / "killed.py"
    script.write_text("import os,signal;os.kill(os.getpid(), signal.SIGTERM)")
    result = _relay(upstream, log_path, "--exec", f"{sys.executable} {script}")
    assert result.exit_code == 128 + 15


def test_a_log_that_cannot_be_opened_is_a_clean_error(upstream, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    result = CliRunner().invoke(
        egress,
        [
            "relay",
            "--upstream",
            upstream.url,
            "--log",
            str(blocker / "sub" / "relay.jsonl"),
            "--exec",
            "true",
        ],
    )
    assert result.exit_code == 1
    assert "the egress relay log cannot be opened" in result.output


def test_a_refused_upstream_releases_the_log(log_path):
    result = CliRunner().invoke(
        egress, ["relay", "--upstream", "http://example.com", "--log", str(log_path)]
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    RelayLog(log_path).close()  # opened then closed again: nothing holds the lock


def test_the_foreground_relay_prints_its_origin_and_stops_on_ctrl_c(
    upstream, log_path, monkeypatch
):
    from traigent.egress.relay import EgressRelay

    stopped: list[bool] = []

    def interrupt(self):  # noqa: ANN001
        raise KeyboardInterrupt

    real_stop = EgressRelay.stop

    def recording_stop(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        stopped.append(True)
        return real_stop(self, *args, **kwargs)

    monkeypatch.setattr(EgressRelay, "serve_forever", interrupt)
    monkeypatch.setattr(EgressRelay, "stop", recording_stop)
    result = _relay(upstream, log_path)
    assert result.exit_code == 0
    assert "egress relay listening on http://127.0.0.1:" in result.output
    assert "export TRAIGENT_BACKEND_URL=http://127.0.0.1:" in result.output
    assert f"export {RELAY_ENV_VAR}=1" in result.output
    assert "egress relay stopped" in result.output
    assert stopped == [True]
    RelayLog(log_path).close()


def _write_log(path: Path) -> None:
    log = RelayLog(path)
    first = log.log_request("GET", "/api/v1/things/42", 0, "a" * 64)
    second = log.log_request("POST", "/api/v1/things", 12, "b" * 64)
    log.log_response(first, 200, 5)
    log.log_response(99, 500, 1)  # a response whose request is not in the log
    assert second == 2
    log.close()


def test_list_shows_a_dash_for_a_request_without_a_response(log_path):
    _write_log(log_path)
    result = CliRunner().invoke(egress, ["list", "--log", str(log_path)])
    assert result.exit_code == 0
    rows = [line.split() for line in result.output.splitlines()[1:]]
    assert [(r[0], r[2], r[3]) for r in rows] == [
        ("1", "GET", "200"),
        ("2", "POST", "-"),
    ]


def test_list_skips_a_response_that_has_no_request(log_path):
    _write_log(log_path)
    result = CliRunner().invoke(egress, ["list", "--log", str(log_path)])
    assert "500" not in result.output  # the orphan response (seq 99) is not listed


def test_show_prints_only_the_lines_that_exist_for_a_seq(log_path):
    _write_log(log_path)
    runner = CliRunner()
    only_request = json.loads(
        runner.invoke(egress, ["show", "2", "--log", str(log_path)]).output
    )
    assert [e["kind"] for e in only_request] == ["request"]
    only_response = json.loads(
        runner.invoke(egress, ["show", "99", "--log", str(log_path)]).output
    )
    assert [e["kind"] for e in only_response] == ["response"]


def test_show_reports_a_missing_seq_on_stderr_and_exits_1(log_path):
    _write_log(log_path)
    result = CliRunner().invoke(egress, ["show", "7", "--log", str(log_path)])
    assert result.exit_code == 1
    assert "no logged request with seq 7" in result.output
