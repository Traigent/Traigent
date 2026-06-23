"""CLI command: traigent next-steps.

Fetches backend-provided, client-safe next-step recommendations for an
experiment run. Requires a backend version that exposes
GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from traigent.analytics.next_steps import NextStepsClient
from traigent.config.backend_config import DEFAULT_LOCAL_URL

console = Console(width=120)


@click.command("next-steps")
@click.argument("experiment_run_id")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output the raw backend payload as JSON.",
)
@click.option(
    "--backend-url",
    default=DEFAULT_LOCAL_URL,
    show_default=True,
    envvar=["TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL"],
    help=(
        "Backend API base URL (env: TRAIGENT_BACKEND_URL / TRAIGENT_API_URL). "
        "Requires a backend with the next-steps endpoint."
    ),
)
def next_steps(
    experiment_run_id: str,
    output_json: bool,
    backend_url: str,
) -> None:
    """Show next-step recommendations from a backend that supports them.

    Requires GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps
    on the configured backend.
    """
    try:
        payload = asyncio.run(_fetch_next_steps(experiment_run_id, backend_url))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return

    _print_next_steps_table(payload)


async def _fetch_next_steps(
    experiment_run_id: str,
    backend_url: str,
) -> dict[str, Any]:
    async with NextStepsClient(backend_url=backend_url) as client:
        return await client.get_next_steps(experiment_run_id)


def _print_next_steps_table(payload: dict[str, Any]) -> None:
    experiment_run_id = payload.get("experiment_run_id", "unknown")
    console.print(f"\n[bold blue]Next Steps for {experiment_run_id}[/bold blue]\n")

    rows = payload.get("next_steps") or []
    if rows:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Priority", justify="right", no_wrap=True)
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Rationale", overflow="fold")
        table.add_column("Action", no_wrap=True)

        for row in sorted(rows, key=_priority_sort_key):
            action = row.get("action") or {}
            table.add_row(
                str(row.get("priority", "")),
                str(row.get("category", "")),
                str(row.get("rationale", "")),
                str(action.get("command_template", "")),
            )
        console.print(table)
    else:
        console.print("[yellow]No next steps were returned by the backend.[/yellow]")

    console.print(f"\n[bold]Caveat:[/bold] {payload.get('caveat', '')}")


def _priority_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    priority = row.get("priority")
    if isinstance(priority, int):
        return priority, str(row.get("id", ""))
    return 10_000, str(row.get("id", ""))
