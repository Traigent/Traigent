"""CLI command: traigent detect-tvars

Scans Python files or directories for tuned variable candidates using
static analysis (AST) and optional LLM analysis.

Usage examples::

    # Scan a single file
    traigent detect-tvars my_agent.py

    # Scan a directory recursively
    traigent detect-tvars src/agents/

    # Scan a specific function only
    traigent detect-tvars my_agent.py --function answer_question

    # Output JSON for tooling integration
    traigent detect-tvars my_agent.py --json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from traigent.tuned_variables.detection_types import (
    DetectionConfidence,
    DetectionResult,
)
from traigent.tuned_variables.detector import TunedVariableDetector

console = Console()
stderr = Console(stderr=True)

_CONFIDENCE_COLORS = {
    DetectionConfidence.HIGH: "green",
    DetectionConfidence.MEDIUM: "yellow",
    DetectionConfidence.LOW: "dim",
}


@click.command("detect-tvars")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--function",
    "-f",
    "function_name",
    default=None,
    help="Analyze only this function (default: all functions).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output results as JSON for tooling integration.",
)
@click.option(
    "--min-confidence",
    type=click.Choice(["high", "medium", "low"], case_sensitive=False),
    default="low",
    show_default=True,
    help="Minimum confidence level to include in results.",
)
@click.option(
    "--show-suggestions",
    is_flag=True,
    default=True,
    show_default=True,
    help="Show suggested ParameterRange code in table output.",
)
def detect_tvars(
    path: Path,
    function_name: str | None,
    output_json: bool,
    min_confidence: str,
    show_suggestions: bool,
) -> None:
    """Detect tuned variable candidates in Python files.

    PATH can be a Python file or a directory. When a directory is given,
    all .py files are scanned recursively.
    """
    detector = TunedVariableDetector()
    results: list[DetectionResult] = []

    if path.is_file():
        if path.suffix != ".py":
            raise click.ClickException(f"Expected a .py file, got: {path}")
        results = detector.detect_from_file(path, function_name)
    else:
        # Directory: scan all .py files recursively
        py_files = sorted(path.rglob("*.py"))
        if not py_files:
            stderr.print(f"[yellow]No Python files found in {path}[/yellow]")
            sys.exit(0)

        for py_file in py_files:
            # Skip common non-source directories
            if any(
                part.startswith((".", "_", "__pycache__", "venv", "node_modules"))
                for part in py_file.parts
            ):
                continue
            results.extend(detector.detect_from_file(py_file, function_name))

    # Filter by confidence
    min_level = DetectionConfidence(min_confidence)
    _order = {
        DetectionConfidence.HIGH: 2,
        DetectionConfidence.MEDIUM: 1,
        DetectionConfidence.LOW: 0,
    }
    min_order = _order[min_level]
    filtered_results = [
        DetectionResult(
            function_name=r.function_name,
            candidates=tuple(
                c for c in r.candidates if _order[c.confidence] >= min_order
            ),
            warnings=r.warnings,
            source_hash=r.source_hash,
            detection_strategies_used=r.detection_strategies_used,
        )
        for r in results
        if any(_order[c.confidence] >= min_order for c in r.candidates)
    ]

    if output_json:
        _output_json(filtered_results)
    else:
        _output_table(filtered_results, show_suggestions)


def _output_json(results: list[DetectionResult]) -> None:
    """Print results as JSON array to stdout."""
    output: list[dict[str, Any]] = []
    for r in results:
        for c in r.candidates:
            entry: dict[str, Any] = {
                "function": r.function_name,
                "name": c.name,
                "type": c.candidate_type.value,
                "confidence": c.confidence.value,
                "line": c.location.line,
                "current_value": c.current_value,
                "canonical_name": c.canonical_name,
                "reasoning": c.reasoning,
                "detection_source": c.detection_source,
            }
            if c.suggested_range:
                entry["suggested_range"] = {
                    "range_type": c.suggested_range.range_type,
                    "kwargs": c.suggested_range.kwargs,
                    "code": c.suggested_range.to_parameter_range_code(),
                }
            output.append(entry)
    click.echo(json.dumps(output, indent=2))


def _build_function_table(result: DetectionResult, show_suggestions: bool) -> Table:
    """Build a rich Table for a single function's candidates."""
    table = Table(
        title=f"[cyan]{result.function_name}[/cyan]",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Variable", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Confidence")
    table.add_column("Value")
    if show_suggestions:
        table.add_column("Suggested Range", style="green")

    for c in result.candidates:
        color = _CONFIDENCE_COLORS.get(c.confidence, "")
        confidence_str = f"[{color}]{c.confidence.value}[/{color}]"
        value_str = repr(c.current_value) if c.current_value is not None else "—"
        suggestion_str = (
            c.suggested_range.to_parameter_range_code() if c.suggested_range else "—"
        )
        row = [c.name, c.candidate_type.value, confidence_str, value_str]
        if show_suggestions:
            row.append(suggestion_str)
        table.add_row(*row)
    return table


def _print_config_snippet(results: list[DetectionResult]) -> None:
    """Print a copy-paste-ready config space snippet."""
    config_snippet: dict[str, Any] = {}
    for r in results:
        config_snippet.update(r.to_configuration_space())

    if not config_snippet:
        return
    console.print("[bold]Suggested configuration_space snippet:[/bold]")
    console.print("[dim]# Add to your @traigent.optimize decorator:[/dim]")
    console.print("[dim]configuration_space = {[/dim]")
    for name, kwargs in config_snippet.items():
        console.print(f"[dim]    {name!r}: {kwargs!r},[/dim]")
    console.print("[dim]}[/dim]\n")


def _output_table(results: list[DetectionResult], show_suggestions: bool) -> None:
    """Print results as a rich table."""
    if not results:
        console.print("[yellow]No tunable variable candidates detected.[/yellow]")
        return

    total = sum(r.count for r in results)
    console.print(
        f"\n[bold]Detected {total} tunable variable candidate(s)[/bold] "
        f"across {len(results)} function(s)\n"
    )

    for result in results:
        if result.count == 0:
            continue
        console.print(_build_function_table(result, show_suggestions))
        for w in result.warnings:
            console.print(f"  [yellow]⚠ {w}[/yellow]")
        console.print()

    _print_config_snippet(results)
