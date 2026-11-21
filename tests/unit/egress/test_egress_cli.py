"""``traigent egress`` CLI: --exec, list/show, and lazy import of the relay."""

from __future__ import annotations

import json
import subprocess
import sys

from click.testing import CliRunner

from traigent.cli.egress_commands import egress

_PRINT_ENV = (
    "import os,sys;"
    "sys.stdout.write(os.environ['TRAIGENT_BACKEND_URL']+' '+os.environ['TRAIGENT_API_URL'])"
)


def test_import_traigent_does_not_load_the_relay():
    code = (
        "import sys, traigent, traigent.cli.main;"
        "bad=[m for m in sys.modules if m.startswith('traigent.egress.relay')];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_exec_sets_both_env_vars_and_returns_child_exit_code(
    upstream, log_path, tmp_path, monkeypatch
):
    for name in ("TRAIGENT_ENVIRONMENT", "ENVIRONMENT", "TRAIGENT_ENV"):
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "env.txt"
    script = (
        "import os,sys;"
        f"open({str(out)!r},'w').write(os.environ['TRAIGENT_BACKEND_URL']+' '"
        "+os.environ['TRAIGENT_API_URL']+' '+repr([os.environ.get(k) for k in "
        "('TRAIGENT_ENVIRONMENT','ENVIRONMENT','TRAIGENT_ENV')]));sys.exit(7)"
    )
    script_file = tmp_path / "child.py"
    script_file.write_text(script)
    result = CliRunner().invoke(
        egress,
        [
            "relay",
            "--upstream",
            upstream.url,
            "--log",
            str(log_path),
            "--exec",
            f"{sys.executable} {script_file}",
        ],
    )
    assert result.exit_code == 7
    backend, api, envs = out.read_text().split(" ", 2)
    assert backend == api
    assert backend.startswith("http://127.0.0.1:")
    assert envs == "[None, None, None]"


def test_exec_refuses_a_plain_http_non_loopback_upstream(log_path):
    result = CliRunner().invoke(
        egress,
        [
            "relay",
            "--upstream",
            "http://example.com",
            "--log",
            str(log_path),
            "--exec",
            "true",
        ],
    )
    assert result.exit_code != 0


def test_list_and_show_read_the_log(upstream, log_path, tmp_path):
    script = tmp_path / "call.py"
    script.write_text(
        "import os,urllib.request;"
        "urllib.request.urlopen(os.environ['TRAIGENT_BACKEND_URL']+'/api/v1/things/42?k=v')"
        ".read()"
    )
    runner = CliRunner()
    runner.invoke(
        egress,
        [
            "relay",
            "--upstream",
            upstream.url,
            "--log",
            str(log_path),
            "--exec",
            f"{sys.executable} {script}",
        ],
    )
    listed = runner.invoke(egress, ["list", "--log", str(log_path)])
    assert listed.exit_code == 0
    assert "/api/v1/things/{}" in listed.output
    assert "200" in listed.output
    assert "k=v" not in listed.output
    shown = runner.invoke(egress, ["show", "1", "--log", str(log_path)])
    assert [e["kind"] for e in json.loads(shown.output)] == ["request", "response"]
    missing = runner.invoke(egress, ["show", "99", "--log", str(log_path)])
    assert missing.exit_code == 1


def test_list_on_missing_log(tmp_path):
    result = CliRunner().invoke(egress, ["list", "--log", str(tmp_path / "none.jsonl")])
    assert result.exit_code == 0
    assert "No logged requests" in result.output


def test_exec_is_not_run_through_a_shell(upstream, log_path):
    # A shell would run `false` after `;` and exit 1; shlex+exec passes ';' as an arg.
    result = CliRunner().invoke(
        egress,
        [
            "relay",
            "--upstream",
            upstream.url,
            "--log",
            str(log_path),
            "--exec",
            "true ; false",
        ],
    )
    assert result.exit_code == 0


def test_ctrl_c_during_exec_stops_the_relay_and_exits_130(
    upstream, log_path, monkeypatch
):
    def interrupted(*args, **kwargs):  # noqa: ANN002, ANN003
        raise KeyboardInterrupt

    monkeypatch.setattr("traigent.cli.egress_commands.subprocess.run", interrupted)
    result = CliRunner().invoke(
        egress,
        [
            "relay",
            "--upstream",
            upstream.url,
            "--log",
            str(log_path),
            "--exec",
            "sleep 5",
        ],
    )
    assert result.exit_code == 130
    from traigent.egress.log import RelayLog

    RelayLog(log_path).close()  # the lock is released: the relay stopped


def test_exec_child_sees_the_relay_switch(upstream, log_path, tmp_path, monkeypatch):
    monkeypatch.delenv("TRAIGENT_EGRESS_RELAY", raising=False)
    out = tmp_path / "switch.txt"
    script_file = tmp_path / "child_switch.py"
    script_file.write_text(
        "import os;"
        f"open({str(out)!r},'w').write(os.environ.get('TRAIGENT_EGRESS_RELAY','unset'))"
    )
    result = CliRunner().invoke(
        egress,
        [
            "relay",
            "--upstream",
            upstream.url,
            "--log",
            str(log_path),
            "--exec",
            f"{sys.executable} {script_file}",
        ],
    )
    assert result.exit_code == 0
    assert out.read_text() == "1"


def _child_no_proxy(tmp_path, monkeypatch, upstream, log_path, existing):
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    if existing is not None:
        monkeypatch.setenv("NO_PROXY", existing)
        monkeypatch.setenv("no_proxy", existing)
    out = tmp_path / "np.txt"
    script_file = tmp_path / "np.py"
    script_file.write_text(
        "import os;"
        f"open({str(out)!r},'w').write(repr((os.environ.get('NO_PROXY'),"
        "os.environ.get('no_proxy'))))"
    )
    result = CliRunner().invoke(
        egress,
        [
            "relay",
            "--upstream",
            upstream.url,
            "--log",
            str(log_path),
            "--exec",
            f"{sys.executable} {script_file}",
        ],
    )
    assert result.exit_code == 0
    return eval(out.read_text())  # noqa: S307 - our own repr of two strings


def test_exec_child_gets_the_relay_host_in_no_proxy(
    upstream, log_path, tmp_path, monkeypatch
):
    upper, lower = _child_no_proxy(tmp_path, monkeypatch, upstream, log_path, None)
    assert upper == "127.0.0.1" and lower == "127.0.0.1"


def test_exec_child_keeps_an_existing_no_proxy_value(
    upstream, log_path, tmp_path, monkeypatch
):
    upper, lower = _child_no_proxy(
        tmp_path, monkeypatch, upstream, log_path, "corp.example.com,10.0.0.0/8"
    )
    for value in (upper, lower):
        assert value.split(",") == ["corp.example.com", "10.0.0.0/8", "127.0.0.1"]


def test_no_exec_mode_prints_the_no_proxy_export(upstream, log_path, monkeypatch):

    from traigent.egress import relay as relay_module

    monkeypatch.setattr(relay_module.EgressRelay, "serve_forever", lambda self: None)
    result = CliRunner().invoke(
        egress, ["relay", "--upstream", upstream.url, "--log", str(log_path)]
    )
    assert result.exit_code == 0
    assert "export NO_PROXY=127.0.0.1" in result.output
