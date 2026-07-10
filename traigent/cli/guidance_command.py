"""CLI surface for the additive SmartOps Planner V2 protocol."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click
from rich.console import Console

from traigent.analytics.planner import PlannerV2Client
from traigent.config.backend_config import DEFAULT_LOCAL_URL, BackendConfig

console = Console(width=120)


def _resolve_backend_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    return BackendConfig.get_configured_backend_url() or DEFAULT_LOCAL_URL


def _resolve_api_key(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return BackendConfig.get_api_key()


@click.group("guidance")
def guidance() -> None:
    """Fetch and execute authoritative Planner V2 decisions."""


@guidance.command("next")
@click.argument("run_id")
@click.option(
    "--profile",
    type=click.Choice(
        ["quality_first", "balanced", "cost_first"], case_sensitive=False
    ),
    default="balanced",
    show_default=True,
)
@click.option(
    "--treatment",
    type=click.Choice(["rules_control", "policy_override"], case_sensitive=False),
    default="policy_override",
    show_default=True,
)
@click.option(
    "--strict-experiment",
    is_flag=True,
    default=False,
    help="Reject fallback, treatment drift, and incomplete experiment provenance.",
)
@click.option("--json", "output_json", is_flag=True, default=False)
@click.option(
    "--backend-url",
    default=None,
    envvar=["TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL"],
    help="Backend URL (then stored CLI URL, then local default).",
)
@click.option("--api-key", default=None, help="API key (then stored credentials).")
def next_decision(
    run_id: str,
    profile: str,
    treatment: str,
    strict_experiment: bool,
    output_json: bool,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Fetch one authoritative Planner V2 decision for RUN_ID."""
    try:
        payload = asyncio.run(
            _fetch_next_decision(
                run_id,
                backend_url=_resolve_backend_url(backend_url),
                api_key=_resolve_api_key(api_key),
                profile=profile,
                treatment=treatment,
                strict_experiment=strict_experiment,
            )
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return
    _print_decision(payload)


@guidance.command("execute")
@click.option("--decision", "decision_id", required=True)
@click.option("--json", "output_json", is_flag=True, default=False)
@click.option(
    "--backend-url",
    default=None,
    envvar=["TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL"],
    help="Backend URL (then stored CLI URL, then local default).",
)
@click.option("--api-key", default=None, help="API key (then stored credentials).")
def execute_decision(
    decision_id: str,
    output_json: bool,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Resolve DECISION to its private, leased execution specification.

    The response is structured argv plus typed parameters.  It is never passed
    to a shell; a coding agent dispatches the declared operation through its
    installed Traigent skill and later records the authoritative result.
    """
    try:
        payload = asyncio.run(
            _resolve_decision(
                decision_id,
                backend_url=_resolve_backend_url(backend_url),
                api_key=_resolve_api_key(api_key),
            )
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return
    spec = payload["execution_spec"]
    console.print("\n[bold blue]Planner V2 execution resolved[/bold blue]\n")
    console.print(f"[bold]Operation:[/bold] {spec['operation_kind']}")
    console.print(f"[bold]Variant:[/bold] {spec['variant']}")
    console.print(f"[bold]Attempt:[/bold] {spec['attempt_id']}")
    console.print(f"[bold]Receipt URL:[/bold] {spec['receipt_url']}")
    console.print(f"[bold]Lease expires:[/bold] {spec['lease_expires_at']}")
    if spec.get("sample_limit") is not None:
        console.print(f"[bold]Sample limit:[/bold] {spec['sample_limit']}")
    if spec.get("max_trials") is not None:
        console.print(f"[bold]Max trials:[/bold] {spec['max_trials']}")
    if spec.get("reserved_cost_usd") is not None:
        console.print(f"[bold]Reserved cost (USD):[/bold] {spec['reserved_cost_usd']}")
    console.print(
        "[yellow]Dispatch this typed operation through the matching installed "
        "Traigent skill. Do not run server data through a shell.[/yellow]"
    )


@guidance.command("execute-resolved", hidden=True)
@click.option("--attempt", "attempt_id", required=True)
def reject_direct_resolved_execution(attempt_id: str) -> None:
    """Fail closed if a binding-only resolved argv is invoked directly.

    The backend emits this static argv to bind the private execution spec to an
    attempt. It is deliberately not an executable optimization command: coding
    agents dispatch the typed operation through the matching installed skill.
    """
    raise click.ClickException(
        f"attempt {attempt_id} is a binding token, not a shell command; "
        "dispatch the typed execution_spec through the matching Traigent skill"
    )


@guidance.command("receipt")
@click.option("--lifecycle", "lifecycle_id", required=True)
@click.option("--decision", "decision_id", required=True)
@click.option("--attempt", "attempt_id", required=True)
@click.option(
    "--status",
    type=click.Choice(["started", "submitted", "failed", "skipped"]),
    required=True,
)
@click.option("--successor-run", "successor_run_id", default=None)
@click.option("--result-ref", default=None)
@click.option("--json", "output_json", is_flag=True, default=False)
@click.option(
    "--backend-url",
    default=None,
    envvar=["TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL"],
    help="Backend URL (then stored CLI URL, then local default).",
)
@click.option("--api-key", default=None, help="API key (then stored credentials).")
def receipt(
    lifecycle_id: str,
    decision_id: str,
    attempt_id: str,
    status: str,
    successor_run_id: str | None,
    result_ref: str | None,
    output_json: bool,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Append an execution receipt without claiming server verification."""
    try:
        payload = asyncio.run(
            _record_receipt(
                lifecycle_id,
                decision_id,
                attempt_id=attempt_id,
                status=status,
                successor_run_id=successor_run_id,
                result_ref=result_ref,
                backend_url=_resolve_backend_url(backend_url),
                api_key=_resolve_api_key(api_key),
            )
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return
    console.print(
        "[bold blue]Planner V2 receipt recorded[/bold blue] "
        f"status={payload['status']} verification={payload['verification_status']}"
    )


@guidance.command("reopen")
@click.argument("lifecycle_id")
@click.option(
    "--reason",
    type=click.Choice(["new_artifact", "budget", "operator"]),
    required=True,
)
@click.option(
    "--expected-treatment",
    type=click.Choice(["rules_control", "policy_override"]),
    required=True,
    help="Stopped parent's precommitted treatment; fail if inheritance drifts.",
)
@click.option(
    "--expected-profile",
    type=click.Choice(["quality_first", "balanced", "cost_first"]),
    required=True,
    help="Stopped parent's utility profile; fail if inheritance drifts.",
)
@click.option("--json", "output_json", is_flag=True, default=False)
@click.option(
    "--backend-url",
    default=None,
    envvar=["TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL"],
    help="Backend URL (then stored CLI URL, then local default).",
)
@click.option("--api-key", default=None, help="API key (then stored credentials).")
def reopen(
    lifecycle_id: str,
    reason: str,
    expected_treatment: str,
    expected_profile: str,
    output_json: bool,
    backend_url: str | None,
    api_key: str | None,
) -> None:
    """Create a treatment- and profile-inheriting child lifecycle."""
    try:
        payload = asyncio.run(
            _reopen_lifecycle(
                lifecycle_id,
                reason=reason,
                expected_treatment=expected_treatment,
                expected_profile=expected_profile,
                backend_url=_resolve_backend_url(backend_url),
                api_key=_resolve_api_key(api_key),
            )
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        click.echo(json.dumps(payload, indent=2))
        return
    console.print(
        "[bold blue]Planner V2 lifecycle reopened[/bold blue] "
        f"lifecycle={payload['lifecycle_id']} reason={payload['reason']} "
        f"treatment={payload['requested_variant']} profile={payload['utility_profile']}"
    )


async def _fetch_next_decision(
    run_id: str,
    *,
    backend_url: str,
    api_key: str | None,
    profile: str,
    treatment: str,
    strict_experiment: bool,
) -> dict[str, Any]:
    async with PlannerV2Client(backend_url=backend_url, api_key=api_key) as client:
        return await client.get_next_decision(
            run_id,
            utility_profile=profile,
            treatment=treatment,
            strict_experiment=strict_experiment,
        )


async def _resolve_decision(
    decision_id: str,
    *,
    backend_url: str,
    api_key: str | None,
) -> dict[str, Any]:
    async with PlannerV2Client(backend_url=backend_url, api_key=api_key) as client:
        return await client.resolve_decision(decision_id)


async def _record_receipt(
    lifecycle_id: str,
    decision_id: str,
    *,
    attempt_id: str,
    status: str,
    successor_run_id: str | None,
    result_ref: str | None,
    backend_url: str,
    api_key: str | None,
) -> dict[str, Any]:
    async with PlannerV2Client(backend_url=backend_url, api_key=api_key) as client:
        return await client.record_receipt(
            lifecycle_id,
            decision_id,
            attempt_id=attempt_id,
            status=status,
            successor_run_id=successor_run_id,
            result_ref=result_ref,
        )


async def _reopen_lifecycle(
    lifecycle_id: str,
    *,
    reason: str,
    expected_treatment: str,
    expected_profile: str,
    backend_url: str,
    api_key: str | None,
) -> dict[str, Any]:
    async with PlannerV2Client(backend_url=backend_url, api_key=api_key) as client:
        return await client.reopen_lifecycle(
            lifecycle_id,
            reason=reason,
            expected_treatment=expected_treatment,
            expected_profile=expected_profile,
        )


def _print_decision(payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    meta = payload["meta"]
    action = decision["action"]
    console.print(
        f"\n[bold blue]Planner V2 decision for {payload['run_id']}[/bold blue]\n"
    )
    console.print(f"[bold]Lifecycle:[/bold] {payload['lifecycle_id']}")
    console.print(f"[bold]Category:[/bold] {decision['category']}")
    console.print(f"[bold]Mode:[/bold] {decision['mode']}")
    console.print(f"[bold]Variant:[/bold] {action['variant']}")
    console.print(f"[bold]Rationale:[/bold] {decision['rationale']}")
    console.print(
        "[bold]Evidence:[/bold] "
        f"{decision['evidence_level']} / {decision['advantage_label']}"
    )
    if not action["command_template"]:
        console.print("[yellow]This decision is non-executable.[/yellow]")
    else:
        console.print(f"[bold]Command:[/bold] {action['command_template']}")
    console.print(
        "[dim]"
        f"treatment={meta['served_variant']} engine={meta['selector_engine']} "
        f"profile={decision['utility_profile']} fallback="
        f"{meta['fallback_reason'] or 'none'}"
        "[/dim]"
    )
