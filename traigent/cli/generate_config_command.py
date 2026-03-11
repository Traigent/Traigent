"""CLI command: ``traigent generate-config``.

Generates a complete ``@traigent.optimize(...)`` configuration from a
Python source file using preset heuristics and optional LLM enrichment.
"""

from __future__ import annotations

import json
import os
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
    type=click.Choice(["table", "python", "json", "tvl"], case_sensitive=False),
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
@click.option(
    "--apply",
    "apply_decorator",
    is_flag=True,
    default=False,
    help="Apply the generated config as a decorator on the target function.",
)
@click.option(
    "--no-backup",
    is_flag=True,
    default=False,
    help="Skip creating .bak backup when using --apply.",
)
def generate_config(
    path: Path,
    function_name: str | None,
    enrich: bool,
    model: str,
    budget: float,
    output_format: str,
    subsystems_str: str | None,
    apply_decorator: bool,
    no_backup: bool,
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

    # --apply: insert/update decorator on the target function
    if apply_decorator:
        if not function_name:
            click.echo(
                "Error: --apply requires --function/-f to specify which function "
                "to decorate.",
                err=True,
            )
            raise SystemExit(1)
        from traigent.config_generator.apply import apply_config

        modified = apply_config(
            path,
            result,
            function_name,
            backup=not no_backup,
        )
        click.echo(f"Applied @traigent.optimize() to {function_name} in {modified}")
        if not no_backup:
            click.echo(f"Backup saved to {path.with_suffix('.py.bak')}")
        return

    # Format and display
    if output_format == "python":
        _output_python(result)
    elif output_format == "json":
        _output_json(result)
    elif output_format == "tvl":
        _output_tvl(result, path)
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


def _output_tvl(result, path: Path):
    """Write a .tvl.yml spec file next to the source file."""
    try:
        import yaml
    except ImportError:
        click.echo(
            "Error: PyYAML is required for TVL export. "
            "Install it with: pip install pyyaml",
            err=True,
        )
        raise SystemExit(1) from None

    safe_source = path.resolve()
    base_dir = Path(os.environ.get("TRAIGENT_CONFIG_BASE_DIR", Path.cwd())).resolve()
    try:
        safe_source.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(
            f"Path '{safe_source}' is outside the allowed base directory '{base_dir}'"
        ) from exc

    spec = result.to_tvl_spec(module_name=safe_source.stem)
    tvl_path = safe_source.with_suffix(".tvl.yml")
    tvl_path.write_text(yaml.dump(spec, default_flow_style=False, sort_keys=False))
    click.echo(f"TVL spec written to: {tvl_path}")


def _print_section(title: str, items, formatter) -> None:
    """Print a titled section with separator lines."""
    if not items:
        return
    click.echo(f"\n{'=' * 60}")
    click.echo(title)
    click.echo(f"{'=' * 60}")
    for item in items:
        formatter(item)


def _output_table(result):
    """Print a human-readable table summary."""
    if result.agent_type:
        click.echo(f"\nAgent type: {result.agent_type}")

    _print_section(
        "Tunable Variables",
        result.tvars,
        lambda t: click.echo(f"  {t.name}: {t.to_range_code()}  [{t.source}]"),
    )
    _print_section(
        "Objectives",
        result.objectives,
        lambda o: click.echo(f"  {o.name} ({o.orientation}, weight={o.weight:.2f})"),
    )
    _print_section(
        "Safety Constraints",
        result.safety_constraints,
        lambda sc: click.echo(f"  {sc.metric_name} {sc.operator} {sc.threshold}"),
    )
    _print_section(
        "Structural Constraints",
        result.structural_constraints,
        lambda c: (
            click.echo(f"  {c.description}"),
            click.echo(f"    Code: {c.constraint_code.splitlines()[0]}"),
        ),
    )
    _print_section(
        "Benchmarks",
        result.benchmarks,
        lambda b: click.echo(f"  {b.name}: {b.description[:80]}"),
    )
    _print_section(
        "Recommended Additional TVars",
        result.recommendations,
        lambda r: (
            click.echo(f"  [{r.impact_estimate}] {r.name}: {r.to_range_code()}"),
            click.echo(f"    {r.reasoning}"),
        ),
    )
    _print_section(
        "Warnings",
        result.warnings,
        lambda w: click.echo(f"  ! {w}"),
    )

    if result.llm_calls_made > 0:
        click.echo(
            f"\nLLM: {result.llm_calls_made} calls, ${result.llm_cost_usd:.4f} spent"
        )

    click.echo()
