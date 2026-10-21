"""detect-tvars / generate-config failure modes (#2648)."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from traigent.cli.detect_tvars_command import detect_tvars
from traigent.cli.generate_config_command import generate_config
from traigent.hooks.validator import AgentValidator

AGENT = """\
def answer(q):
    temperature = 0.7
    model = "gpt-4o-mini"
    return model
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_missing_function_exits_nonzero(runner, tmp_path):
    p = tmp_path / "agent.py"
    p.write_text(AGENT)
    result = runner.invoke(detect_tvars, [str(p), "-f", "nosuch", "--json"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_missing_function_in_directory_exits_nonzero(runner, tmp_path):
    (tmp_path / "agent.py").write_text(AGENT)
    result = runner.invoke(detect_tvars, [str(tmp_path), "-f", "nosuch"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_existing_function_still_works(runner, tmp_path):
    p = tmp_path / "agent.py"
    p.write_text(AGENT)
    result = runner.invoke(detect_tvars, [str(p), "-f", "answer", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)


def test_directory_of_only_unparseable_files_exits_nonzero(runner, tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n")
    result = runner.invoke(detect_tvars, [str(tmp_path)])
    assert result.exit_code != 0


def test_directory_skips_unparseable_file_but_scans_others(runner, tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n")
    (tmp_path / "agent.py").write_text(AGENT)
    result = runner.invoke(detect_tvars, [str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)


def test_empty_result_on_readable_input_still_exit_zero(runner, tmp_path):
    (tmp_path / "boring.py").write_text("x = 1\n")
    result = runner.invoke(detect_tvars, [str(tmp_path)])
    assert result.exit_code == 0
    assert "No tunable variable candidates" in result.output


def test_dot_directory_in_user_supplied_path_is_not_skipped(runner, tmp_path):
    proj = tmp_path / ".hidden" / "proj"
    proj.mkdir(parents=True)
    (proj / "agent.py").write_text(AGENT)
    result = runner.invoke(detect_tvars, [str(proj), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)


def test_dotdot_target_is_not_skipped(runner, tmp_path):
    sub = tmp_path / "src" / "sub"
    sub.mkdir(parents=True)
    (tmp_path / "src" / "agent.py").write_text(AGENT)
    cwd = os.getcwd()
    os.chdir(sub)
    try:
        result = runner.invoke(detect_tvars, ["..", "--json"])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0
    assert json.loads(result.output)


def test_hidden_subdirectory_below_target_still_skipped(runner, tmp_path):
    hidden = tmp_path / ".venvish"
    hidden.mkdir()
    (hidden / "agent.py").write_text(AGENT)
    result = runner.invoke(detect_tvars, [str(tmp_path), "--json"])
    assert json.loads(result.output) == []


def test_generate_config_unparseable_file_exits_nonzero(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIGENT_CONFIG_BASE_DIR", str(tmp_path))
    p = tmp_path / "broken.py"
    p.write_text("def f(:\n")
    result = runner.invoke(generate_config, [str(p)])
    assert result.exit_code != 0
    assert "parse" in result.output.lower()
    assert "Agent type" not in result.output


def test_hook_validator_dot_parent_directory_not_skipped(tmp_path):
    proj = tmp_path / ".work" / "agents"
    proj.mkdir(parents=True)
    (proj / "a.py").write_text(
        "import traigent\n\n@traigent.optimize(configuration_space={'t': [0.1, 0.2]})\n"
        "def f(q):\n    return q\n"
    )
    results = AgentValidator().validate_directory(proj)
    assert results, "files under a dot-prefixed parent of the target must be scanned"
