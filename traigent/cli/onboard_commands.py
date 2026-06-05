"""Guided onboarding commands for the Traigent CLI."""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

import click
from rich.console import Console
from rich.table import Table

from traigent.cli.auth_commands import TraigentAuthCLI

console = Console()

AgentName = Literal["claude", "cursor", "codex"]

AGENT_LABELS: dict[AgentName, str] = {
    "claude": "Claude Code",
    "cursor": "Cursor",
    "codex": "Codex",
}
FIRST_PROMPT_TOOL_LINE: dict[AgentName, str] = {
    "claude": "Use Claude Code tools to inspect code and propose the smallest safe change.",
    "cursor": "Use Cursor agent tools to inspect code and propose the smallest safe change.",
    "codex": "Use Codex tools to inspect code and propose the smallest safe change.",
}
PLAN_JSON_BEGIN = "BEGIN_TRAIGENT_ONBOARD_PLAN_JSON"
PLAN_JSON_END = "END_TRAIGENT_ONBOARD_PLAN_JSON"
SKILLS_COMMAND = ["npx", "skills", "add", "Traigent/agents-skills"]


def _stdin_is_tty() -> bool:
    return bool(sys.stdin.isatty())


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _detect_python_project(cwd: Path) -> list[str]:
    markers = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "Pipfile",
        "poetry.lock",
        "uv.lock",
    ]
    return [marker for marker in markers if (cwd / marker).exists()]


def _dependency_command(project_markers: list[str]) -> list[str]:
    if shutil.which("uv") and (
        "pyproject.toml" in project_markers or "uv.lock" in project_markers
    ):
        return ["uv", "add", "traigent[integrations]"]
    return [sys.executable, "-m", "pip", "install", "traigent[integrations]"]


def _detect_coding_agents(cwd: Path, home: Path | None = None) -> list[AgentName]:
    resolved_home = home or Path.home()
    agents: list[AgentName] = []
    if (resolved_home / ".claude").exists() or shutil.which("claude"):
        agents.append("claude")
    if (resolved_home / ".cursor").exists() or (cwd / ".cursor").exists():
        agents.append("cursor")
    if (resolved_home / ".codex").exists() or (cwd / "AGENTS.md").exists():
        agents.append("codex")
    return agents


def _mcp_help_succeeds() -> bool:
    executable = shutil.which("traigent")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _mcp_registration_command(agent: AgentName) -> list[str]:
    return ["traigent", "mcp", "register", "--agent", agent]


def _run_command(command: list[str]) -> bool:
    console.print(f"[dim]$ {_command_text(command)}[/dim]")
    try:
        result = subprocess.run(command, check=False)
    except OSError as exc:
        console.print(f"[red]Command failed to start: {exc}[/red]")
        return False
    if result.returncode != 0:
        console.print(f"[red]Command exited with status {result.returncode}[/red]")
        return False
    return True


async def _auth_status(auth_cli: TraigentAuthCLI) -> str:
    if await auth_cli._check_stored_api_key():
        return "authenticated"
    if await auth_cli._check_env_api_key():
        return "authenticated"
    return "not_authenticated"


def _python_version_step() -> dict[str, Any]:
    version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    status = "passed" if sys.version_info >= (3, 11) else "failed"
    return {
        "id": "python_version",
        "status": status,
        "description": f"Python {version}; Traigent requires Python >=3.11",
    }


def _select_prompt_agent(agents: list[AgentName]) -> AgentName:
    if "codex" in agents:
        return "codex"
    if agents:
        return agents[0]
    return "codex"


def build_first_prompt(agent: AgentName | None, project_path: Path) -> str:
    """Build the paste block printed by `traigent first-prompt`."""
    tool_line = (
        FIRST_PROMPT_TOOL_LINE[agent]
        if agent is not None
        else "Use your coding-agent tools to inspect code and propose the smallest safe change."
    )
    lines = [
        f"You are using Traigent in this project: {project_path}",
        "Read the agent guide first: https://traigent.ai/agent.md",
        tool_line,
        "Always run Traigent in dry-run/mock mode first.",
        "Never spend real provider tokens or start paid optimization without my approval.",
        "Treat scaffolded evals as drafts until I approve their dataset and metrics.",
        "When approved for real execution, keep cost estimates and limits visible.",
        "Finish with evidence, changed files, test results, and a PR-ready summary.",
    ]
    return "\n".join(lines)


def _build_non_tty_plan(
    *,
    cwd: Path,
    project_markers: list[str],
    detected_agents: list[AgentName],
    auth_status: str,
    mcp_available: bool,
    no_login: bool,
) -> dict[str, Any]:
    commands: list[dict[str, object]] = []
    steps: list[dict[str, object]] = [_python_version_step()]

    if project_markers:
        dependency_command = _dependency_command(project_markers)
        commands.append(
            {
                "id": "add_dependency",
                "command": _command_text(dependency_command),
                "argv": dependency_command,
                "requires_consent": True,
            }
        )
        steps.append(
            {
                "id": "project_dependency",
                "status": "planned",
                "description": "Offer adding traigent[integrations] to this Python project.",
                "commands": [_command_text(dependency_command)],
            }
        )
    else:
        steps.append(
            {
                "id": "project_dependency",
                "status": "skipped",
                "description": "No Python project marker detected in the current directory.",
            }
        )

    if auth_status == "authenticated":
        steps.append(
            {
                "id": "device_login",
                "status": "skipped",
                "description": "Existing credentials validated successfully.",
            }
        )
    elif no_login:
        steps.append(
            {
                "id": "device_login",
                "status": "skipped",
                "description": "--no-login was set.",
            }
        )
    else:
        login_command = ["traigent", "auth", "device-login"]
        commands.append(
            {
                "id": "device_login",
                "command": _command_text(login_command),
                "argv": login_command,
                "requires_consent": False,
                "requires_browser_action": True,
            }
        )
        steps.append(
            {
                "id": "device_login",
                "status": "planned",
                "description": "Start browser device login; the human approves in the browser.",
                "commands": [_command_text(login_command)],
            }
        )

    if detected_agents:
        skills_text = _command_text(SKILLS_COMMAND)
        commands.append(
            {
                "id": "install_agent_skills",
                "command": skills_text,
                "argv": SKILLS_COMMAND,
                "requires_consent": True,
                "agents": detected_agents,
            }
        )
        steps.append(
            {
                "id": "agent_skills",
                "status": "planned",
                "description": "Offer installing Traigent agent skills for each detected coding agent.",
                "commands": [skills_text],
            }
        )
    else:
        steps.append(
            {
                "id": "agent_skills",
                "status": "skipped",
                "description": "No supported coding agent markers detected.",
            }
        )

    if mcp_available and detected_agents:
        mcp_commands = [_mcp_registration_command(agent) for agent in detected_agents]
        for command in mcp_commands:
            commands.append(
                {
                    "id": "register_mcp",
                    "command": _command_text(command),
                    "argv": command,
                    "requires_consent": True,
                }
            )
        steps.append(
            {
                "id": "mcp_registration",
                "status": "planned",
                "description": "Offer MCP registration because `traigent mcp --help` succeeded.",
                "commands": [_command_text(command) for command in mcp_commands],
            }
        )
    else:
        steps.append(
            {
                "id": "mcp_registration",
                "status": "skipped",
                "description": "`traigent mcp` is not available yet; MCP registration skipped.",
            }
        )

    quickstart_command = ["traigent", "quickstart"]
    first_prompt_agent = _select_prompt_agent(detected_agents)
    first_prompt_command = ["traigent", "first-prompt", "--agent", first_prompt_agent]
    commands.extend(
        [
            {
                "id": "verify_quickstart",
                "command": _command_text(quickstart_command),
                "argv": quickstart_command,
                "requires_consent": False,
            },
            {
                "id": "first_prompt",
                "command": _command_text(first_prompt_command),
                "argv": first_prompt_command,
                "requires_consent": False,
            },
        ]
    )
    steps.extend(
        [
            {
                "id": "verification",
                "status": "planned",
                "description": "Run the packaged mock quickstart to verify zero-cost local execution.",
                "commands": [_command_text(quickstart_command)],
            },
            {
                "id": "first_prompt",
                "status": "ready",
                "description": "Print the coding-agent paste block.",
                "commands": [_command_text(first_prompt_command)],
            },
        ]
    )

    return {
        "project_path": str(cwd),
        "python_project": bool(project_markers),
        "project_markers": project_markers,
        "detected_agents": detected_agents,
        "auth_status": auth_status,
        "mcp_available": mcp_available,
        "steps": steps,
        "commands": commands,
    }


def _print_non_tty_plan(plan: dict[str, Any]) -> None:
    console.print("[bold]Traigent onboarding plan (non-interactive)[/bold]")
    console.print(f"Project: {plan['project_path']}")
    agents = plan["detected_agents"]
    console.print(
        "Detected agents: "
        + (", ".join(str(agent) for agent in agents) if agents else "none")
    )
    console.print(f"Auth status: {plan['auth_status']}")
    console.print("Commands are listed below for a caller to run with human consent.")
    for command in plan["commands"]:
        if isinstance(command, dict):
            console.print(f"  - {command['command']}")
    console.print(PLAN_JSON_BEGIN)
    click.echo(json.dumps(plan, indent=2))
    console.print(PLAN_JSON_END)


def _print_detection_table(project_markers: list[str], agents: list[AgentName]) -> None:
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row(
        "Python project",
        ", ".join(project_markers) if project_markers else "not detected",
    )
    table.add_row(
        "Coding agents",
        (
            ", ".join(AGENT_LABELS[agent] for agent in agents)
            if agents
            else "none detected"
        ),
    )
    console.print(table)


def _run_quickstart_verification() -> bool:
    console.print("\n[bold]Verification[/bold]")
    console.print("Running packaged mock quickstart...")
    try:
        from traigent.examples.quickstart.__main__ import main as run_quickstart

        run_quickstart()
    except Exception as exc:
        console.print(f"[red]Mock quickstart failed: {exc}[/red]")
        return False
    console.print("[green]Mock quickstart completed.[/green]")
    return True


def _run_interactive_onboard(no_login: bool, backend_url: str | None) -> bool:
    cwd = Path.cwd().resolve()
    project_markers = _detect_python_project(cwd)
    detected_agents = _detect_coding_agents(cwd)
    python_step = _python_version_step()

    console.print("\n[bold blue]Traigent Onboarding[/bold blue]")
    console.print(python_step["description"])
    if python_step["status"] != "passed":
        raise click.ClickException("Python >=3.11 is required.")

    _print_detection_table(project_markers, detected_agents)

    if project_markers:
        dependency_command = _dependency_command(project_markers)
        if click.confirm(
            f"Add the SDK to this project with `{_command_text(dependency_command)}`?",
            default=False,
        ):
            _run_command(dependency_command)
    else:
        console.print("No Python project marker found; dependency install skipped.")

    auth_cli = TraigentAuthCLI(backend_url_override=backend_url)
    current_auth_status = asyncio.run(_auth_status(auth_cli))
    if current_auth_status != "authenticated":
        if no_login:
            console.print("Device login skipped because --no-login was set.")
        else:
            if not asyncio.run(auth_cli.device_login()):
                return False

    for agent in detected_agents:
        if click.confirm(
            f"Install Traigent agent skills for {AGENT_LABELS[agent]}?",
            default=False,
        ):
            _run_command(SKILLS_COMMAND)

    if _mcp_help_succeeds():
        for agent in detected_agents:
            command = _mcp_registration_command(agent)
            if click.confirm(
                f"Register Traigent MCP for {AGENT_LABELS[agent]}?",
                default=False,
            ):
                _run_command(command)
    else:
        console.print(
            "`traigent mcp` is not available in this SDK build yet; "
            "MCP registration skipped."
        )

    if not _run_quickstart_verification():
        return False

    prompt_agent = _select_prompt_agent(detected_agents)
    console.print("\n[bold]First Prompt[/bold]")
    click.echo(build_first_prompt(prompt_agent, cwd))
    return True


@click.command()
@click.option("--no-login", is_flag=True, help="Skip device-code login.")
@click.option(
    "--backend-url",
    default=None,
    help="Backend URL to authenticate against during device login.",
)
def onboard(no_login: bool, backend_url: str | None) -> None:
    """Run guided setup for Traigent in this project."""
    cwd = Path.cwd().resolve()

    if not _stdin_is_tty():
        project_markers = _detect_python_project(cwd)
        detected_agents = _detect_coding_agents(cwd)
        auth_cli = TraigentAuthCLI(backend_url_override=backend_url)
        current_auth_status = asyncio.run(_auth_status(auth_cli))
        mcp_available = _mcp_help_succeeds()
        plan = _build_non_tty_plan(
            cwd=cwd,
            project_markers=project_markers,
            detected_agents=detected_agents,
            auth_status=current_auth_status,
            mcp_available=mcp_available,
            no_login=no_login,
        )
        _print_non_tty_plan(plan)
        if current_auth_status != "authenticated" and not no_login:
            success = asyncio.run(auth_cli.device_login())
            raise SystemExit(0 if success else 1)
        return

    success = _run_interactive_onboard(no_login=no_login, backend_url=backend_url)
    raise SystemExit(0 if success else 1)


@click.command("first-prompt")
@click.option(
    "--agent",
    type=click.Choice(["claude", "cursor", "codex"], case_sensitive=False),
    default=None,
    help="Coding agent flavor for the paste block.",
)
def first_prompt(agent: str | None) -> None:
    """Print the Traigent coding-agent first prompt."""
    normalized_agent = cast(AgentName, agent.lower()) if agent else None
    click.echo(build_first_prompt(normalized_agent, Path.cwd().resolve()))
