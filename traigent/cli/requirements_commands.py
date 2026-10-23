"""CLI: ``traigent requirements validate|apply|status|watch|respond|explain``.

Every command that needs a declaration takes ``FILE --function NAME`` and finds
the function exactly the way ``traigent optimize`` does.  Output carries closed
codes only; server bodies and user content are never echoed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import click

from traigent.runtime import manifest
from traigent.runtime.apply import (
    EXIT_CONFLICT,
    ApplyConflict,
    apply_requirement,
    matching_heads,
)
from traigent.runtime.client import RuntimeClient
from traigent.runtime.model import (
    RuntimeRequirement,
    RuntimeRequirementError,
    one_line,
    safe_lines,
)
from traigent.runtime.respond import OccurrenceUsed
from traigent.runtime.respond import respond as run_respond
from traigent.runtime.watch import LOGGER_NAME as _LOGGER_NAME
from traigent.runtime.watch import poll_once

E_FUNCTION_NOT_FOUND = "function_not_found"
E_AMBIGUOUS_FUNCTION = "ambiguous_function"
E_NO_DECLARATION = "no_requirement_declared"
E_PROJECT_REQUIRED = "project_id_required"
E_AGENT_REQUIRED = "agent_id_required"


def _fail(exc: RuntimeRequirementError, status: int = 1) -> None:
    click.echo(f"error: {one_line(exc)}", err=True)
    raise click.exceptions.Exit(status)


def _find_function(file_path: Path, function: str) -> Any:
    """Load ``file_path`` and return the single function named ``function``."""
    import inspect
    import sys as _sys

    from traigent.cli.main import (
        _find_optimizable_functions,
        _load_user_module,
        _resolve_user_cli_path,
    )

    resolved = _resolve_user_cli_path(file_path, "File path", must_exist=True)
    _sys.path.insert(0, str(resolved.parent))
    try:
        module = _load_user_module(resolved)
    except RuntimeRequirementError:
        raise  # a closed error raised by a declaration keeps its own code
    except Exception:  # noqa: BLE001 - user module errors are not echoed
        raise RuntimeRequirementError("module_load_failed") from None
    finally:
        with_path = str(resolved.parent)
        if with_path in _sys.path:
            _sys.path.remove(with_path)
    if module is None:
        raise RuntimeRequirementError("module_load_failed")
    found = [obj for _name, obj in _find_optimizable_functions(module, function)]
    if not found:
        candidate = getattr(module, function, None)
        if (
            candidate is not None
            and getattr(candidate, "requirements", None)
            and not inspect.ismodule(candidate)
        ):
            found = [candidate]
    if not found:
        raise RuntimeRequirementError(E_FUNCTION_NOT_FOUND)
    if len(found) > 1:
        raise RuntimeRequirementError(E_AMBIGUOUS_FUNCTION)
    return found[0]


def _declared(fn: Any) -> RuntimeRequirement:
    items = getattr(fn, "requirements", None)
    if not items:
        raise RuntimeRequirementError(E_NO_DECLARATION)
    if len(items) != 1:
        raise RuntimeRequirementError("duplicate_requirement")
    declared: RuntimeRequirement = items[0]
    return declared


def _ids(
    requirement: RuntimeRequirement, project_id: str | None, agent_id: str | None
) -> tuple[str, str]:
    project = project_id or os.environ.get("TRAIGENT_PROJECT_ID")
    agent = agent_id or requirement.agent_id
    if not project:
        raise RuntimeRequirementError(E_PROJECT_REQUIRED)
    if not agent:
        raise RuntimeRequirementError(E_AGENT_REQUIRED)
    return project, agent


def _common(func: Any) -> Any:
    func = click.argument("file_path", type=click.Path(exists=True, path_type=Path))(
        func
    )
    func = click.option(
        "--function", "-f", "function", required=True, help="Function name."
    )(func)
    func = click.option(
        "--project-id", default=None, help="Project id (env: TRAIGENT_PROJECT_ID)."
    )(func)
    func = click.option(
        "--agent-id", default=None, help="Server agent id (never derived)."
    )(func)
    return func


def _interactive() -> bool:
    return sys.stdin.isatty()


def _state_dir() -> Path:
    return manifest.default_state_dir()


@click.group()
def requirements() -> None:
    """Declare, register and react to runtime requirements."""


@requirements.command("validate")
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--function", "-f", "function", required=True, help="Function name.")
def validate_cmd(file_path: Path, function: str) -> None:
    """Validate the declaration offline and print its TVL."""
    try:
        fn = _find_function(file_path, function)
        declared = _declared(fn)
        issues = fn.requirements.validate()
    except RuntimeRequirementError as exc:
        _fail(exc)
        return
    # Multi-line YAML by design: clean each line, keep the line breaks.
    click.echo(safe_lines(declared.to_tvl()))
    for issue in issues:
        click.echo(
            one_line(f"{issue.severity}: {issue.code} at {issue.path}"), err=True
        )
    if any(i.severity == "error" for i in issues):
        raise click.exceptions.Exit(1)


@requirements.command("apply")
@_common
@click.option(
    "--dry-run", is_flag=True, help="Validate and compile without any network call."
)
def apply_cmd(
    file_path: Path,
    function: str,
    project_id: str | None,
    agent_id: str | None,
    dry_run: bool,
) -> None:
    """Register the requirement (idempotent; fails closed on any ambiguity)."""
    try:
        fn = _find_function(file_path, function)
        declared = _declared(fn)
        project, agent = (
            _ids(declared, project_id, agent_id)
            if not dry_run
            else (project_id or "", agent_id or declared.agent_id or "")
        )
        from traigent.runtime.tvl_fragment import module_name_for

        result = apply_requirement(
            declared,
            project_id=project,
            agent_id=agent,
            dry_run=dry_run,
            state_dir=_state_dir(),
            module=module_name_for(fn.requirements.owner),
        )
    except ApplyConflict as conflict:
        click.echo(
            "error: requirement_exists (the registered profile differs)", err=True
        )
        for name, local, remote in conflict.diff:
            click.echo(
                one_line(f"  {name}: declared={local} registered={remote}"),
                err=True,
            )
        raise click.exceptions.Exit(EXIT_CONFLICT) from None
    except RuntimeRequirementError as exc:
        _fail(exc)
        return
    click.echo(
        one_line(
            f"{result.outcome} via {result.route}; "
            f"policy digest {result.policy_digest[:12]}"
        )
    )
    if result.receipt is not None:
        r = result.receipt
        click.echo(
            one_line(
                f"head={r.head_id} revision={r.revision_id} "
                f"number={r.revision_number} "
                f"provenance={r.provenance} status={r.status}"
            )
        )


@requirements.command("status")
@_common
def status_cmd(
    file_path: Path, function: str, project_id: str | None, agent_id: str | None
) -> None:
    """Show the registered head and open breaches for the declaration."""
    try:
        fn = _find_function(file_path, function)
        declared = _declared(fn)
        project, agent = _ids(declared, project_id, agent_id)
        api = RuntimeClient(project)
        heads = matching_heads(api.list_requirements(), declared, project, agent)
        rows, has_more = api.list_breaches(agent)
    except RuntimeRequirementError as exc:
        _fail(exc)
        return
    from traigent.runtime.watch import coverage_for

    if not heads:
        click.echo("not registered (run `traigent requirements apply`)")
    for head in heads:
        click.echo(
            one_line(
                f"head={head.get('head_id')} revision={head.get('revision_id')} "
                f"status={head.get('status')} provenance={head.get('provenance')}"
            )
        )
    open_rows = [
        r for r in rows if r.get("outcome") is None and r.get("closed_at") is None
    ]
    click.echo(
        f"open breaches: {len(open_rows)}; coverage: {coverage_for(len(rows), has_more)}"
    )


@requirements.command("watch")
@_common
@click.option(
    "--interval",
    type=float,
    default=0.0,
    show_default=True,
    help="Seconds between polls (0 = once).",
)
def watch_cmd(
    file_path: Path,
    function: str,
    project_id: str | None,
    agent_id: str | None,
    interval: float,
) -> None:
    """Best-effort snapshot reporting of breach transitions (not an event feed)."""
    import logging

    try:
        fn = _find_function(file_path, function)
        declared = _declared(fn)
        project, agent = _ids(declared, project_id, agent_id)
    except RuntimeRequirementError as exc:
        _fail(exc)
        return
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    failed = False
    while True:
        result = poll_once(
            [declared], project_id=project, agent_id=agent, state_dir=_state_dir()
        )
        click.echo(
            one_line(
                f"coverage: {result.coverage}; breaches seen: {result.rows}; "
                f"events: {len(result.events)}"
            )
        )
        if result.error:
            click.echo(f"error: {one_line(result.error)}", err=True)
            failed = True
        if interval <= 0:
            break
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break
    if failed and interval <= 0:
        raise click.exceptions.Exit(1)


@requirements.command("respond")
@_common
@click.option(
    "--max-fix-cost-usd",
    type=float,
    default=None,
    help="Cost ceiling for the optimize run.",
)
@click.option(
    "--agent-ref",
    default=None,
    help=(
        "Hosted agent ref (agentref_...): run the attempt a person granted in the "
        "portal (reserve, start, run, report) instead of the local approval prompt."
    ),
)
@click.option(
    "--attempt-ref",
    default=None,
    help="With --agent-ref: the attempt to run (default: the one approved attempt).",
)
def respond_cmd(
    file_path: Path,
    function: str,
    project_id: str | None,
    agent_id: str | None,
    max_fix_cost_usd: float | None,
    agent_ref: str | None,
    attempt_ref: str | None,
) -> None:
    """Start a human-approved fix (an optimize run) for an open breach. Never deploys."""

    def confirm(facts: dict[str, Any]) -> bool:
        click.echo("Open breach, restore requested:")
        for key, value in facts.items():
            click.echo(one_line(f"  {key}: {value}"))
        return bool(
            click.confirm("Run optimize under this cost ceiling?", default=False)
        )

    try:
        fn = _find_function(file_path, function)
        declared = _declared(fn)
        project, agent = _ids(declared, project_id, agent_id)
        if not callable(getattr(fn, "optimize", None)):
            raise RuntimeRequirementError("function_not_optimizable")
        result = asyncio.run(
            run_respond(
                fn,
                declared,
                project_id=project,
                agent_id=agent,
                confirm=confirm,
                interactive=_interactive(),
                max_fix_cost_usd=max_fix_cost_usd,
                state_dir=_state_dir(),
                agent_ref=agent_ref,
                attempt_ref=attempt_ref,
            )
        )
    except RuntimeRequirementError as exc:
        if isinstance(exc, OccurrenceUsed):
            click.echo(
                one_line(
                    f"occurrence already used: run id {exc.run_id or 'unavailable'}"
                ),
                err=True,
            )
        _fail(exc)
        return
    click.echo(
        one_line(
            f"optimization started for breach {result.breach_id} "
            f"(occurrence {result.occurrence}); "
            f"run id: {result.run_id or 'unavailable'}; "
            f"cost ceiling: {result.cost_ceiling_usd} USD"
        )
    )


@requirements.command("explain")
@click.argument("ref")
def explain_cmd(ref: str) -> None:
    """Resolve an opaque ref from a webhook or event back to its local name."""
    local = manifest.local_for(_state_dir(), "*", ref)
    if local is None:
        click.echo("error: ref_unknown", err=True)
        raise click.exceptions.Exit(1)
    click.echo(one_line(local))
