"""CLI command: traigent plan.

Fetches a backend-provided, client-safe pre-run optimization plan. Requires a
backend version that exposes POST /api/v1/optimization/plan.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from traigent.analytics.optimization_plan import OptimizationPlanClient
from traigent.config.backend_config import DEFAULT_LOCAL_URL

console = Console(width=120)


@click.command("plan")
@click.option(
    "--task-description",
    required=True,
    help="Short natural-language description of the optimization task.",
)
@click.option(
    "--dataset-size",
    required=True,
    type=int,
    help="Number of examples available for optimization planning.",
)
@click.option(
    "--has-holdout/--no-holdout",
    default=False,
    show_default=True,
    help="Whether the dataset already has a holdout split.",
)
@click.option(
    "--objective",
    "objectives",
    required=True,
    multiple=True,
    help="Objective metric name. Can be repeated.",
)
@click.option(
    "--max-trials",
    required=True,
    type=int,
    help="Maximum trial count requested for the planned run.",
)
@click.option(
    "--cost-limit",
    "cost_limit_usd",
    required=True,
    type=float,
    help="Maximum approved spend for the planned run, in USD.",
)
@click.option(
    "--task-type",
    default=None,
    help="Optional safe task category supplied to the backend.",
)
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
        "Requires the optimization plan endpoint."
    ),
)
def plan(
    task_description: str,
    dataset_size: int,
    has_holdout: bool,
    objectives: tuple[str, ...],
    max_trials: int,
    cost_limit_usd: float,
    task_type: str | None,
    output_json: bool,
    backend_url: str,
) -> None:
    """Show a backend-provided pre-run optimization plan."""
    try:
        payload = asyncio.run(
            _fetch_optimization_plan(
                task_description=task_description,
                dataset_size=dataset_size,
                dataset_has_holdout=has_holdout,
                objectives=objectives,
                max_trials=max_trials,
                cost_limit_usd=cost_limit_usd,
                task_type=task_type,
                backend_url=backend_url,
            )
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return

    _print_plan_table(payload)


async def _fetch_optimization_plan(
    *,
    task_description: str,
    dataset_size: int,
    dataset_has_holdout: bool,
    objectives: tuple[str, ...],
    max_trials: int,
    cost_limit_usd: float,
    task_type: str | None,
    backend_url: str,
) -> dict[str, Any]:
    async with OptimizationPlanClient(backend_url=backend_url) as client:
        return await client.get_optimization_plan(
            task_description=task_description,
            dataset_size=dataset_size,
            dataset_has_holdout=dataset_has_holdout,
            objectives=objectives,
            max_trials=max_trials,
            cost_limit_usd=cost_limit_usd,
            task_type=task_type,
        )


def _print_plan_table(payload: dict[str, Any]) -> None:
    plan_payload = payload.get("plan") or {}
    console.print("\n[bold blue]Optimization Plan[/bold blue]\n")
    console.print(f"[bold]Phase:[/bold] {payload.get('phase', '')}")
    console.print(f"[bold]Evidence:[/bold] {payload.get('evidence_level', '')}")
    console.print(f"[bold]Advisory:[/bold] {payload.get('advisory', '')}")

    if isinstance(plan_payload, dict):
        console.print(
            "[bold]Budget:[/bold] "
            f"{plan_payload.get('max_trials', '')} trials, "
            f"${plan_payload.get('cost_limit_usd', '')} limit"
        )
        console.print(f"[bold]Algorithm:[/bold] {plan_payload.get('algorithm', '')}")
        console.print(f"[bold]Offline:[/bold] {plan_payload.get('offline', '')}")

        _print_objectives_table(plan_payload.get("objectives") or [])
        _print_models_table(plan_payload.get("models") or [])
        _print_knobs_table(plan_payload.get("knobs") or [])

    _print_steps_table(payload.get("steps") or [])
    console.print(f"\n[bold]Caveat:[/bold] {payload.get('caveat', '')}")


def _print_objectives_table(objectives: list[dict[str, Any]]) -> None:
    if not objectives:
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Objective", style="cyan", no_wrap=True)
    table.add_column("Weight", justify="right", no_wrap=True)
    table.add_column("Orientation", no_wrap=True)
    for objective in objectives:
        table.add_row(
            str(objective.get("name", "")),
            str(objective.get("weight", "")),
            str(objective.get("orientation", "")),
        )
    console.print("\n[bold]Objectives[/bold]")
    console.print(table)


def _print_models_table(models: list[str]) -> None:
    if not models:
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Models", style="cyan")
    for model in models:
        table.add_row(str(model))
    console.print("\n[bold]Models[/bold]")
    console.print(table)


def _print_knobs_table(knobs: list[dict[str, Any]]) -> None:
    if not knobs:
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Knob", style="cyan", no_wrap=True)
    table.add_column("Values", overflow="fold")
    for knob in knobs:
        values = knob.get("values") or []
        table.add_row(str(knob.get("name", "")), ", ".join(map(str, values)))
    console.print("\n[bold]Knobs[/bold]")
    console.print(table)


def _print_steps_table(steps: list[dict[str, Any]]) -> None:
    if not steps:
        console.print("\n[yellow]No steps were returned by the backend.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Label", overflow="fold")
    table.add_column("Command", overflow="fold")
    for step in steps:
        table.add_row(
            str(step.get("id", "")),
            str(step.get("label", "")),
            str(step.get("command_template", "")),
        )
    console.print("\n[bold]Steps[/bold]")
    console.print(table)
