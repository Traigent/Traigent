"""CLI commands for agent build playbooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from traigent.playbook.loader import (
    DEFAULT_PLAYBOOK_FILENAME,
    _load_yaml_mapping,
    load_playbook,
)
from traigent.playbook.model import STAGE_ORDER, Stage, StageStatus
from traigent.playbook.scaffold import scaffold_playbook
from traigent.playbook.staleness import compute_staleness
from traigent.playbook.validator import validate_playbook

console = Console()


@click.group()
def playbook() -> None:
    """Manage the agent build playbook."""


@playbook.command("init")
@click.option("--name", prompt="Agent name", help="Agent name for the playbook.")
@click.option("--agent-type", default=None, help="Optional recommendation agent type.")
@click.option(
    "--entrypoint", default=None, help="Optional module:function or file path."
)
@click.option(
    "--path",
    "path",
    type=click.Path(path_type=Path),
    default=DEFAULT_PLAYBOOK_FILENAME,
    show_default=True,
    help="Path to write the agent build playbook.",
)
def init_playbook(
    name: str,
    agent_type: str | None,
    entrypoint: str | None,
    path: Path,
) -> None:
    """Create an initial agent build playbook."""
    if path.exists():
        raise click.ClickException(f"agent build playbook already exists: {path}")
    if path.parent and not path.parent.exists():
        raise click.ClickException(f"parent directory does not exist: {path.parent}")

    path.write_text(
        scaffold_playbook(name=name, agent_type=agent_type, entrypoint=entrypoint),
        encoding="utf-8",
    )
    click.echo(f"created agent build playbook: {path}")


@playbook.command("validate")
@click.option(
    "--path",
    "path",
    type=click.Path(path_type=Path),
    default=DEFAULT_PLAYBOOK_FILENAME,
    show_default=True,
    help="Path to the agent build playbook.",
)
def validate_playbook_command(path: Path) -> None:
    """Validate an agent build playbook."""
    try:
        payload = _load_yaml_mapping(path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    issues = validate_playbook(payload)
    if issues:
        for issue in issues:
            click.echo(f"{issue.location}: {issue.message}")
        raise SystemExit(1)

    click.echo("playbook valid")


@playbook.command("status")
@click.option(
    "--path",
    "path",
    type=click.Path(path_type=Path),
    default=DEFAULT_PLAYBOOK_FILENAME,
    show_default=True,
    help="Path to the agent build playbook.",
)
def playbook_status(path: Path) -> None:
    """Show agent build playbook stage status."""
    try:
        loaded = load_playbook(path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    stale_by_stage = compute_staleness(loaded)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Stale")
    table.add_column("Pin")

    for stage_name in STAGE_ORDER:
        stage = loaded.stages.get(stage_name)
        table.add_row(
            stage_name,
            _stage_status(stage),
            _stale_label(stage, stale_by_stage[stage_name]),
            _pin_summary(stage),
        )

    console.print(table)


def _stage_status(stage: Stage | None) -> str:
    if stage is None:
        return "missing"
    return stage.status.value


def _stale_label(stage: Stage | None, stale: bool) -> str:
    if (
        stage is None
        or stage.status is not StageStatus.PINNED
        or stage.pinned_at is None
    ):
        return ""
    return "yes" if stale else "no"


def _pin_summary(stage: Stage | None) -> str:
    if stage is None or not stage.pin:
        return ""
    return ", ".join(
        f"{key}={_shorten(value)}" for key, value in list(stage.pin.items())[:3]
    )


def _shorten(value: Any) -> str:
    text = str(value)
    return f"{text[:57]}..." if len(text) > 60 else text
