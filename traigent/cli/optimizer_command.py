"""CLI commands for the Traigent Optimizer adoption assistant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from traigent.optimizer.proposer import build_decorate_plan
from traigent.optimizer.scanner import scan_path

console = Console()


@click.group("optimizer")
def optimizer() -> None:
    """Adoption assistant for finding and preparing optimization targets."""


@optimizer.command("scan")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--function",
    "-f",
    "function_name",
    default=None,
    help="Analyze only this function name.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Print the full scan report as JSON.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the scan report JSON to this file.",
)
@click.option(
    "--top",
    type=int,
    default=None,
    help="Limit report candidates to the top N ranked functions.",
)
def scan(
    path: Path,
    function_name: str | None,
    output_json: bool,
    output_path: Path | None,
    top: int | None,
) -> None:
    """Scan a Python file or directory and rank optimizer candidates."""

    report = scan_path(path, function_name=function_name)
    if top is not None:
        report["candidates"] = report["candidates"][: max(top, 0)]

    if output_path is not None:
        output_path.write_text(_to_json(report), encoding="utf-8")

    if output_json:
        click.echo(_to_json(report))
        return

    _print_scan_summary(report)
    if output_path is not None:
        console.print(f"[green]Wrote scan report:[/green] {output_path}")


@optimizer.command("decorate")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--function",
    "-f",
    "function_name",
    required=True,
    help="Function name to prepare for @traigent.optimize.",
)
@click.option(
    "--objective",
    "objectives",
    multiple=True,
    help="Confirmed objective to include. May be passed multiple times.",
)
@click.option(
    "--dataset",
    "dataset_ref",
    default=None,
    help="Existing eval dataset path/URI to reference.",
)
@click.option(
    "--emit",
    "requested_emit_mode",
    type=click.Choice(["auto", "inline", "tvl", "tvl-only"]),
    default="auto",
    show_default=True,
    help="Requested output style for the eventual decorator/spec.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Print the decorate plan as JSON.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the decorate plan JSON to this file.",
)
@click.option(
    "--write",
    is_flag=True,
    default=False,
    help="Apply the plan. Not implemented in this initial dry-run slice.",
)
def decorate(
    path: Path,
    function_name: str,
    objectives: tuple[str, ...],
    dataset_ref: str | None,
    requested_emit_mode: str,
    output_json: bool,
    output_path: Path | None,
    write: bool,
) -> None:
    """Prepare a dry-run decorate plan for one function."""

    if write:
        raise click.ClickException(
            "optimizer decorate --write is not implemented in this slice; "
            "run without --write to review the plan."
        )

    try:
        plan = build_decorate_plan(
            path,
            function_name=function_name,
            objective_names=objectives,
            dataset_ref=dataset_ref,
            requested_emit_mode=requested_emit_mode,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_path is not None:
        output_path.write_text(_to_json(plan), encoding="utf-8")

    if output_json:
        click.echo(_to_json(plan))
        return

    _print_decorate_summary(plan)
    if output_path is not None:
        console.print(f"[green]Wrote decorate plan:[/green] {output_path}")


def _to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=False)


def _print_scan_summary(report: dict[str, Any]) -> None:
    candidates = report["candidates"]
    if not candidates:
        console.print("[yellow]No optimizer candidates detected.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right")
    table.add_column("Function")
    table.add_column("Score", justify="right")
    table.add_column("Signals")
    table.add_column("TVARs")
    table.add_column("Objectives")
    for index, candidate in enumerate(candidates, start=1):
        table.add_row(
            str(index),
            candidate["function"]["qualified_name"],
            f"{candidate['score']:.2f}",
            ", ".join(signal["kind"] for signal in candidate["signals"]) or "-",
            ", ".join(signal["tvar"]["name"] for signal in candidate["tvar_signals"])
            or "-",
            ", ".join(
                objective["name"] for objective in candidate["objective_candidates"]
            )
            or "-",
        )
    console.print(table)


def _print_decorate_summary(plan: dict[str, Any]) -> None:
    target = plan["target"]
    console.print(f"[bold]Decorate plan:[/bold] {target['function']}")
    console.print(f"Emit mode: {plan['resolved_emit_mode']}")
    console.print(
        "TVARs: "
        + (
            ", ".join(
                binding["tvar"]["name"] for binding in plan["proposed_tvar_bindings"]
            )
            or "-"
        )
    )
    console.print(
        "Selected objectives: "
        + (
            ", ".join(objective["name"] for objective in plan["selected_objectives"])
            or "-"
        )
    )
    if plan["warnings"]:
        for warning in plan["warnings"]:
            console.print(f"[yellow]Warning:[/yellow] {warning}")
