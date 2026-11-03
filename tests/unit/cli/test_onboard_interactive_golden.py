"""Golden characterization tests for ``_run_interactive_onboard``."""

from __future__ import annotations

from typing import Any

import click
import pytest

from traigent.cli import onboard_commands as oc


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    markers: list[str],
    agents: list[str],
    python_status: str = "passed",
    auth_status: str = "authenticated",
    login_ok: bool = True,
    mcp_ok: bool = True,
    quickstart_ok: bool = True,
    answers: list[bool] | None = None,
) -> list[Any]:
    log: list[Any] = []
    scripted = list(answers or [])

    monkeypatch.setattr(oc, "_detect_python_project", lambda cwd: markers)
    monkeypatch.setattr(oc, "_detect_coding_agents", lambda cwd: agents)
    monkeypatch.setattr(
        oc,
        "_python_version_step",
        lambda: {"status": python_status, "description": "py-desc"},
    )
    monkeypatch.setattr(
        oc, "_print_detection_table", lambda m, a: log.append(("table", m, a))
    )
    monkeypatch.setattr(oc, "_dependency_command", lambda m: ["dep", *m])
    monkeypatch.setattr(oc, "_skills_command_for_profile", lambda p: ["skills", p])
    monkeypatch.setattr(oc, "_mcp_registration_command", lambda a: ["mcp", a])
    monkeypatch.setattr(oc, "_run_command", lambda c: log.append(("run", c)) or True)
    monkeypatch.setattr(oc, "_mcp_help_succeeds", lambda: log.append("mcp?") or mcp_ok)
    monkeypatch.setattr(
        oc,
        "_run_quickstart_verification",
        lambda: log.append("quickstart") or quickstart_ok,
    )
    monkeypatch.setattr(oc, "_select_prompt_agent", lambda a: "codex")
    monkeypatch.setattr(oc, "build_first_prompt", lambda agent, cwd: f"PROMPT-{agent}")

    class _Auth:
        def __init__(self, backend_url_override: str | None = None) -> None:
            log.append(("auth_init", backend_url_override))

        async def device_login(self, **kwargs: Any) -> bool:
            log.append(("login", kwargs))
            return login_ok

    async def _status(_cli: Any) -> str:
        log.append("status")
        return auth_status

    monkeypatch.setattr(oc, "TraigentAuthCLI", _Auth)
    monkeypatch.setattr(oc, "_auth_status", _status)

    def _confirm(text: str, default: bool = False) -> bool:
        log.append(("confirm", text, default))
        return scripted.pop(0)

    monkeypatch.setattr(oc.click, "confirm", _confirm)
    monkeypatch.setattr(oc.click, "echo", lambda msg="", **k: log.append(("echo", msg)))
    monkeypatch.setattr(
        oc.console, "print", lambda *a, **k: log.append(("print", a[0] if a else ""))
    )
    return log


def test_python_too_old_raises_before_detection_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _install(monkeypatch, markers=[], agents=[], python_status="failed")
    with pytest.raises(click.ClickException, match="Python >=3.12 is required."):
        oc._run_interactive_onboard(False, None, False)
    assert log == [
        ("print", "\n[bold blue]Traigent Onboarding[/bold blue]"),
        ("print", "py-desc"),
    ]


def test_no_markers_authenticated_no_agents_unavailable_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _install(monkeypatch, markers=[], agents=[], mcp_ok=False)
    assert oc._run_interactive_onboard(False, "http://b", False) is True
    assert log == [
        ("print", "\n[bold blue]Traigent Onboarding[/bold blue]"),
        ("print", "py-desc"),
        ("table", [], []),
        ("print", "No Python project marker found; dependency install skipped."),
        ("auth_init", "http://b"),
        "status",
        "mcp?",
        (
            "print",
            "`traigent mcp` is not available in this SDK build yet; "
            "MCP registration skipped.",
        ),
        "quickstart",
        ("print", "\n[bold]First Prompt[/bold]"),
        ("echo", "PROMPT-codex"),
    ]


def test_markers_dependency_confirm_yes_and_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _install(monkeypatch, markers=["pyproject.toml"], agents=[], answers=[True])
    assert oc._run_interactive_onboard(False, None, False) is True
    assert ("run", ["dep", "pyproject.toml"]) in log
    confirm = [e for e in log if isinstance(e, tuple) and e[0] == "confirm"]
    assert confirm == [
        (
            "confirm",
            "Add the SDK to this project with `dep pyproject.toml`?",
            False,
        )
    ]

    log = _install(monkeypatch, markers=["pyproject.toml"], agents=[], answers=[False])
    oc._run_interactive_onboard(False, None, False)
    assert not any(isinstance(e, tuple) and e[0] == "run" for e in log)


def test_unauthenticated_no_login_skips_login(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _install(monkeypatch, markers=[], agents=[], auth_status="none")
    assert oc._run_interactive_onboard(True, None, False) is True
    assert ("print", "Device login skipped because --no-login was set.") in log
    assert not any(isinstance(e, tuple) and e[0] == "login" for e in log)


def test_unauthenticated_write_env_does_not_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _install(monkeypatch, markers=[], agents=[], auth_status="none")
    assert oc._run_interactive_onboard(False, None, True) is True
    assert ("login", {"save_env_file": True, "write_env_explicit": True}) in log
    assert not any(isinstance(e, tuple) and e[0] == "confirm" for e in log)


def test_unauthenticated_prompts_for_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _install(
        monkeypatch, markers=[], agents=[], auth_status="none", answers=[False]
    )
    assert oc._run_interactive_onboard(False, None, False) is True
    assert (
        "confirm",
        "Write project credentials to this directory's .env file?",
        False,
    ) in log
    assert ("login", {"save_env_file": False, "write_env_explicit": False}) in log


def test_login_failure_returns_false_before_skills_and_quickstart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _install(
        monkeypatch,
        markers=[],
        agents=["claude"],
        auth_status="none",
        login_ok=False,
        answers=[True],
    )
    assert oc._run_interactive_onboard(False, None, False) is False
    assert "quickstart" not in log
    assert "mcp?" not in log


@pytest.mark.parametrize(
    ("profile", "label"),
    [
        ("beginner", "the Traigent first-run beginner journey"),
        ("advanced", "Traigent agent skills"),
    ],
)
def test_skills_and_mcp_prompts_per_agent(
    monkeypatch: pytest.MonkeyPatch, profile: str, label: str
) -> None:
    log = _install(
        monkeypatch,
        markers=[],
        agents=["claude", "cursor"],
        answers=[True, False, False, True],
    )
    assert oc._run_interactive_onboard(False, None, False, profile) is True  # type: ignore[arg-type]
    seq = [e for e in log if isinstance(e, tuple) and e[0] in ("confirm", "run")]
    assert seq == [
        ("confirm", f"Install {label} for Claude Code?", False),
        ("run", ["skills", profile]),
        ("confirm", f"Install {label} for Cursor?", False),
        ("confirm", "Register Traigent MCP for Claude Code?", False),
        ("confirm", "Register Traigent MCP for Cursor?", False),
        ("run", ["mcp", "cursor"]),
    ]


def test_quickstart_failure_returns_false_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _install(monkeypatch, markers=[], agents=[], quickstart_ok=False)
    assert oc._run_interactive_onboard(False, None, False) is False
    assert not any(e == ("echo", "PROMPT-codex") for e in log)
