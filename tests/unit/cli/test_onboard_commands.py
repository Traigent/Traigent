"""Tests for Traigent onboarding CLI commands."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli import onboard_commands
from traigent.cli.main import cli
from traigent.cli.onboard_commands import (
    PLAN_JSON_BEGIN,
    PLAN_JSON_END,
    AgentName,
    build_first_prompt,
)


class _RecordingSecureStore:
    def __init__(self) -> None:
        self.saved: list[tuple[object, ...]] = []

    def get(self, name: str, check_env: bool = True) -> str | None:
        return None

    def set(self, *args: object, **_kwargs: object) -> None:
        self.saved.append(args)


class _RecordingSession:
    def __init__(self) -> None:
        self.post_calls: list[dict[str, object]] = []

    async def __aenter__(self) -> _RecordingSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url: str, **kwargs: object) -> object:
        self.post_calls.append({"url": url, **kwargs})
        raise AssertionError(f"Unexpected POST: {url}")


def _extract_plan(output: str) -> dict[str, object]:
    start = output.index(PLAN_JSON_BEGIN) + len(PLAN_JSON_BEGIN)
    end = output.index(PLAN_JSON_END)
    return json.loads(output[start:end].strip())


def test_onboard_non_tty_emits_human_and_json_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        onboard_commands, "_detect_coding_agents", lambda _cwd: ["codex"]
    )
    monkeypatch.setattr(onboard_commands, "_mcp_help_succeeds", lambda: False)
    monkeypatch.setattr(
        onboard_commands,
        "_dependency_command",
        lambda _markers: ["uv", "add", "traigent[integrations]"],
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("pyproject.toml").write_text(
            "[project]\nname = 'demo'\n", encoding="utf-8"
        )
        result = runner.invoke(cli, ["onboard"])

    assert result.exit_code == 0
    assert "Traigent onboarding plan (non-interactive)" in result.output
    assert PLAN_JSON_BEGIN in result.output
    assert PLAN_JSON_END in result.output

    plan = _extract_plan(result.output)
    assert plan["auth_status"] == "not_checked"
    assert plan["auth_check_command"] == "traigent auth status"
    assert plan["login_command"] == "traigent onboard --login"
    assert plan["detected_agents"] == ["codex"]
    assert plan["python_project"] is True

    commands = plan["commands"]
    assert isinstance(commands, list)
    command_by_id = {
        command["id"]: command for command in commands if isinstance(command, dict)
    }
    assert command_by_id["add_dependency"]["command"] == (
        "uv add 'traigent[integrations]'"
    )
    assert command_by_id["device_login"]["command"] == "traigent onboard --login"
    assert (
        command_by_id["install_agent_skills"]["command"]
        == "npx skills add Traigent/traigent-skills"
    )
    assert command_by_id["verify_quickstart"]["command"] == "traigent quickstart"
    assert (
        command_by_id["first_prompt"]["command"]
        == "traigent first-prompt --agent codex"
    )

    steps = plan["steps"]
    assert isinstance(steps, list)
    assert {step["id"] for step in steps if isinstance(step, dict)} >= {
        "python_version",
        "project_dependency",
        "device_login",
        "agent_skills",
        "mcp_registration",
        "verification",
        "first_prompt",
    }


def test_onboard_non_tty_without_flags_does_not_call_network_or_write_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cli import auth_commands

    session = _RecordingSession()
    store = _RecordingSecureStore()
    assert auth_commands.aiohttp is not None
    monkeypatch.setattr(
        auth_commands.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: session,
    )
    monkeypatch.setattr(auth_commands, "get_secure_credential_store", lambda: store)
    monkeypatch.setattr(onboard_commands, "_detect_coding_agents", lambda _cwd: [])
    monkeypatch.setattr(onboard_commands, "_mcp_help_succeeds", lambda: False)
    monkeypatch.setenv("TRAIGENT_API_KEY", "sk_" + "a" * 43)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["onboard"])
        env_exists = Path(".env").exists()

    assert result.exit_code == 0
    assert session.post_calls == []
    assert store.saved == []
    assert env_exists is False


def test_onboard_non_tty_check_auth_opts_into_live_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    async def fake_auth_status(auth_cli) -> str:
        calls.append(auth_cli)
        return "authenticated"

    monkeypatch.setattr(onboard_commands, "_auth_status", fake_auth_status)
    monkeypatch.setattr(onboard_commands, "_detect_coding_agents", lambda _cwd: [])
    monkeypatch.setattr(onboard_commands, "_mcp_help_succeeds", lambda: False)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["onboard", "--check-auth"])

    assert result.exit_code == 0
    assert len(calls) == 1
    plan = _extract_plan(result.output)
    assert plan["auth_status"] == "authenticated"


def test_onboard_non_tty_login_flag_runs_device_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeAuthCLI:
        def __init__(self, backend_url_override: str | None = None) -> None:
            self.backend_url_override = backend_url_override

        async def device_login(self, **kwargs: object) -> bool:
            calls.append({"backend_url": self.backend_url_override, "kwargs": kwargs})
            return True

    monkeypatch.setattr(onboard_commands, "TraigentAuthCLI", FakeAuthCLI)
    monkeypatch.setattr(onboard_commands, "_detect_coding_agents", lambda _cwd: [])
    monkeypatch.setattr(onboard_commands, "_mcp_help_succeeds", lambda: False)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "onboard",
                "--login",
                "--write-env",
                "--backend-url",
                "https://backend.example.test",
            ],
        )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["backend_url"] == "https://backend.example.test"
    kwargs = calls[0]["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["save_env_file"] is True
    assert kwargs["write_env_explicit"] is True


@pytest.mark.parametrize(
    ("agent", "tool_line"),
    [
        (
            "claude",
            "Use Claude Code tools to inspect code and propose the smallest safe change.",
        ),
        (
            "cursor",
            "Use Cursor agent tools to inspect code and propose the smallest safe change.",
        ),
        (
            "codex",
            "Use Codex tools to inspect code and propose the smallest safe change.",
        ),
    ],
)
def test_first_prompt_golden_outputs(agent: AgentName, tool_line: str) -> None:
    expected = "\n".join(
        [
            "You are using Traigent in this project: /work/project",
            "Read the agent guide first: https://traigent.ai/agent.md",
            tool_line,
            "Always run Traigent in dry-run/mock mode first.",
            "Never spend real provider tokens or start paid optimization without my approval.",
            "Treat scaffolded evals as drafts until I approve their dataset and metrics.",
            "When approved for real execution, keep cost estimates and limits visible.",
            "Finish with evidence, changed files, test results, and a PR-ready summary.",
        ]
    )

    assert build_first_prompt(agent, Path("/work/project")) == expected


def test_first_prompt_command_outputs_selected_agent() -> None:
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["first-prompt", "--agent", "claude"])

    assert result.exit_code == 0
    assert "https://traigent.ai/agent.md" in result.output
    assert "Use Claude Code tools" in result.output
    assert "Never spend real provider tokens" in result.output


def test_quickstart_command_wraps_packaged_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []
    fake_module = types.ModuleType("traigent.examples.quickstart.__main__")
    fake_module.main = lambda: called.append(True)
    monkeypatch.setitem(
        sys.modules, "traigent.examples.quickstart.__main__", fake_module
    )

    result = CliRunner().invoke(cli, ["quickstart"])

    assert result.exit_code == 0
    assert called == [True]
