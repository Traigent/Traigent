"""CLI command: ``traigent generate-config``.

Generates a complete ``@traigent.optimize(...)`` configuration from a
Python source file using preset heuristics and optional LLM enrichment.
"""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("generate-config")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--function",
    "-f",
    "function_name",
    default=None,
    help="Analyze only this function (default: all top-level functions).",
)
@click.option(
    "--enrich",
    is_flag=True,
    default=False,
    help="Use LLM for richer analysis (requires API key, budgeted).",
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM model for enrichment mode.",
)
@click.option(
    "--budget",
    type=float,
    default=0.10,
    show_default=True,
    help="Maximum LLM spend in USD for enrichment.",
)
@click.option(
    "--output",
    "-o",
    "output_format",
    type=click.Choice(["table", "python", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--only",
    "subsystems_str",
    default=None,
    help="Comma-separated subsystems to run (e.g., tvars,objectives,safety).",
)
def generate_config(
    path: Path,
    function_name: str | None,
    enrich: bool,
    model: str,
    budget: float,
    output_format: str,
    subsystems_str: str | None,
) -> None:
    """Generate a complete @traigent.optimize() configuration.

    PATH is a Python file to analyze. Detects tunable variables and
    generates ranges, objectives, safety constraints, structural
    constraints, benchmarks, and TVAR recommendations.

    \b
    Examples:
        traigent generate-config my_agent.py
        traigent generate-config my_agent.py --enrich
        traigent generate-config my_agent.py -f answer_question -o python
    """
    from traigent.config_generator import generate_config as gen_config
    from traigent.config_generator.pipeline import ALL_SUBSYSTEMS

    # Parse subsystems
    subsystems: frozenset[str] | None = None
    if subsystems_str:
        requested = frozenset(s.strip() for s in subsystems_str.split(","))
        invalid = requested - ALL_SUBSYSTEMS
        if invalid:
            click.echo(
                f"Unknown subsystems: {', '.join(sorted(invalid))}. "
                f"Valid: {', '.join(sorted(ALL_SUBSYSTEMS))}",
                err=True,
            )
            raise SystemExit(1)
        subsystems = requested

    try:
        result = gen_config(
            path,
            function_name=function_name,
            enrich=enrich,
            model=model,
            budget_usd=budget,
            subsystems=subsystems,
        )
    except FileNotFoundError:
        click.echo(f"File not found: {path}", err=True)
        raise SystemExit(1) from None
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from None

    # Format and display
    if output_format == "python":
        _output_python(result)
    elif output_format == "json":
        _output_json(result)
    else:
        _output_table(result)


def _output_python(result):
    """Print copy-pasteable Python decorator code."""
    click.echo(result.to_python_code())


def _output_json(result):
    """Print machine-readable JSON output."""
    data = {
        "agent_type": result.agent_type,
        "tvars": [
            {
                "name": t.name,
                "range_type": t.range_type,
                "range_kwargs": t.range_kwargs,
                "source": t.source,
                "confidence": t.confidence,
            }
            for t in result.tvars
        ],
        "objectives": [
            {
                "name": o.name,
                "orientation": o.orientation,
                "weight": o.weight,
            }
            for o in result.objectives
        ],
        "safety_constraints": [
            {
                "metric": sc.metric_name,
                "operator": sc.operator,
                "threshold": sc.threshold,
            }
            for sc in result.safety_constraints
        ],
        "structural_constraints": [
            {
                "description": c.description,
                "constraint_code": c.constraint_code,
            }
            for c in result.structural_constraints
        ],
        "benchmarks": [
            {"name": b.name, "description": b.description} for b in result.benchmarks
        ],
        "recommendations": [
            {
                "name": r.name,
                "range_code": r.to_range_code(),
                "category": r.category,
                "impact": r.impact_estimate,
                "reasoning": r.reasoning,
            }
            for r in result.recommendations
        ],
        "warnings": list(result.warnings),
        "llm_calls_made": result.llm_calls_made,
        "llm_cost_usd": result.llm_cost_usd,
    }
    click.echo(json.dumps(data, indent=2))


def _output_table(result):
    """Print a human-readable table summary."""
    if result.agent_type:
        click.echo(f"\nAgent type: {result.agent_type}")

    # TVars
    if result.tvars:
        click.echo(f"\n{'='*60}")
        click.echo("Tunable Variables")
        click.echo(f"{'='*60}")
        for t in result.tvars:
            click.echo(f"  {t.name}: {t.to_range_code()}  [{t.source}]")

    # Objectives
    if result.objectives:
        click.echo(f"\n{'='*60}")
        click.echo("Objectives")
        click.echo(f"{'='*60}")
        for o in result.objectives:
            click.echo(f"  {o.name} ({o.orientation}, weight={o.weight:.2f})")

    # Safety constraints
    if result.safety_constraints:
        click.echo(f"\n{'='*60}")
        click.echo("Safety Constraints")
        click.echo(f"{'='*60}")
        for sc in result.safety_constraints:
            click.echo(f"  {sc.metric_name} {sc.operator} {sc.threshold}")

    # Structural constraints
    if result.structural_constraints:
        click.echo(f"\n{'='*60}")
        click.echo("Structural Constraints")
        click.echo(f"{'='*60}")
        for c in result.structural_constraints:
            click.echo(f"  {c.description}")
            click.echo(f"    Code: {c.constraint_code.splitlines()[0]}")

    # Benchmarks
    if result.benchmarks:
        click.echo(f"\n{'='*60}")
        click.echo("Benchmarks")
        click.echo(f"{'='*60}")
        for b in result.benchmarks:
            click.echo(f"  {b.name}: {b.description[:80]}")

    # Recommendations
    if result.recommendations:
        click.echo(f"\n{'='*60}")
        click.echo("Recommended Additional TVars")
        click.echo(f"{'='*60}")
        for r in result.recommendations:
            click.echo(f"  [{r.impact_estimate}] {r.name}: {r.to_range_code()}")
            click.echo(f"    {r.reasoning}")

    # Warnings
    if result.warnings:
        click.echo(f"\n{'='*60}")
        click.echo("Warnings")
        click.echo(f"{'='*60}")
        for w in result.warnings:
            click.echo(f"  ! {w}")

    # LLM stats
    if result.llm_calls_made > 0:
        click.echo(
            f"\nLLM: {result.llm_calls_made} calls, "
            f"${result.llm_cost_usd:.4f} spent"
        )

    click.echo()
