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
from traigent.config.backend_config import DEFAULT_LOCAL_URL, BackendConfig

console = Console(width=120)


def _resolve_backend_url(explicit: str | None) -> tuple[str, bool]:
    """Resolve the backend URL.

    Precedence: explicit flag/env value, then env-var / stored-CLI-credential
    configuration (previously ignored; Traigent#1721), then the local default.
    We fall back to ``DEFAULT_LOCAL_URL`` rather than the prod cloud so an
    unconfigured local-dev user keeps hitting localhost as before.

    Returns (resolved_url, used_explicit_source).
    """
    if explicit:
        return explicit, True
    configured = BackendConfig.get_configured_backend_url()
    return (configured or DEFAULT_LOCAL_URL), False


def _resolve_api_key(explicit: str | None) -> str | None:
    """Resolve the API key the same way other authenticated commands do:
    ``TRAIGENT_API_KEY`` env var, then stored CLI credentials from
    ``traigent auth login`` (Traigent#1721)."""
    if explicit:
        return explicit
    return BackendConfig.get_api_key()


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
    default=None,
    envvar=["TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL"],
    help=(
        "Backend API base URL. Requires a backend with the next-steps endpoint. "
        "Defaults to the backend URL stored by `traigent auth login`, then "
        "the local default backend. "
        "(env: TRAIGENT_BACKEND_URL / TRAIGENT_API_URL)"
    ),
)
@click.option(
    "--api-key",
    default=None,
    help="API key (else TRAIGENT_API_KEY env var, then stored CLI credentials).",
)
@click.option(
    "--guidance-variant",
    type=click.Choice(["rules", "policy"], case_sensitive=False),
    default=None,
    envvar="TRAIGENT_GUIDANCE_VARIANT",
    help=(
        "Request the rules or policy guidance treatment. Invalid values fail "
        "locally. (env: TRAIGENT_GUIDANCE_VARIANT)"
    ),
)
@click.option(
    "--strict-experiment",
    is_flag=True,
    default=False,
    help=(
        "Fail unless the requested variant, actual engine, and authoritative "
        "decision provenance all match. Requires --guidance-variant or its env var."
    ),
)
def next_steps(
    experiment_run_id: str,
    output_json: bool,
    backend_url: str | None,
    api_key: str | None,
    guidance_variant: str | None,
    strict_experiment: bool,
) -> None:
    """Show next-step recommendations from a backend that supports them.

    Requires GET /api/v1/analytics/experiments/{experiment_run_id}/next-steps
    on the configured backend.
    """
    resolved_backend_url, backend_url_explicit = _resolve_backend_url(backend_url)
    resolved_api_key = _resolve_api_key(api_key)

    if not output_json and not backend_url_explicit:
        console.print(
            f"[dim]Using backend URL from stored CLI credentials/default: "
            f"{resolved_backend_url}[/dim]"
        )

    try:
        payload = asyncio.run(
            _fetch_next_steps(
                experiment_run_id,
                resolved_backend_url,
                resolved_api_key,
                guidance_variant=guidance_variant,
                strict_experiment=strict_experiment,
            )
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return

    _print_next_steps_table(payload)


async def _fetch_next_steps(
    experiment_run_id: str,
    backend_url: str,
    api_key: str | None,
    *,
    guidance_variant: str | None = None,
    strict_experiment: bool = False,
) -> dict[str, Any]:
    async with NextStepsClient(backend_url=backend_url, api_key=api_key) as client:
        return await client.get_next_steps(
            experiment_run_id,
            guidance_variant=guidance_variant,
            strict_experiment=strict_experiment,
        )


def _print_next_steps_table(payload: dict[str, Any]) -> None:
    experiment_run_id = payload.get("experiment_run_id", "unknown")
    console.print(f"\n[bold blue]Next Steps for {experiment_run_id}[/bold blue]\n")

    posture = payload.get("posture")
    if isinstance(posture, dict):
        summary_text = posture.get("summary_text")
        if summary_text:
            console.print(f"[bold]Posture:[/bold] {summary_text}\n")

    decision = payload.get("decision")
    rows = payload.get("next_steps") or []
    if isinstance(decision, dict):
        action = decision.get("action") or {}
        command = action.get("command_template") or "no action"
        console.print(
            "[bold]Decision:[/bold] "
            f"{decision.get('category', '')} — {decision.get('rationale', '')} "
            f"([dim]{command}[/dim])"
        )
    elif rows:
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

    guidance_meta = payload.get("guidance_meta")
    if isinstance(guidance_meta, dict):
        requested_variant = guidance_meta.get("requested_variant", "")
        served_variant = guidance_meta.get("served_variant", "")
        engine = guidance_meta.get("engine", "")
        fallback_reason = guidance_meta.get("fallback_reason")
        console.print(
            f"guidance: requested={requested_variant} "
            f"served={served_variant} engine={engine} "
            f"fallback={fallback_reason or 'none'}"
        )


def _priority_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    priority = row.get("priority")
    if isinstance(priority, int):
        return priority, str(row.get("id", ""))
    return 10_000, str(row.get("id", ""))
