"""CLI: ``traigent tune compile|check|simulate`` (Continuous Tuning).

TVL is the source of truth: ``check`` prints the TVL reference linter's
diagnostics unchanged and ``simulate`` runs the TVL reference relation.  Both
refuse with ``E_TVL_VALIDATOR_UNAVAILABLE`` when the toolchain is missing or
does not support TVL 1.7 (there is no schema-only fallback).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import click
import yaml

from traigent.tuning.errors import (
    E_TVL_VALIDATOR_UNAVAILABLE,
    TuningError,
    TvlValidationFailed,
)
from traigent.tuning.validator import (
    conditions_trace,
    lint_document,
    require_validator,
)

EXIT_ERRORS = 1
EXIT_UNAVAILABLE = 2


def _die(message: str, status: int = EXIT_ERRORS) -> None:
    click.echo(f"error: {message}", err=True)
    raise click.exceptions.Exit(status)


def _load_yaml(path: Path, what: str) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        _die(f"{what} could not be read as YAML or JSON")


def _require_validator() -> None:
    try:
        require_validator()
    except TuningError as exc:
        _die(str(exc), EXIT_UNAVAILABLE)


def _module_doc(path: Path) -> dict[str, Any]:
    """A TVL module file; a conformance case file is unwrapped to its ``module``."""
    data = _load_yaml(path, "module file")
    if (
        isinstance(data, dict)
        and "tvl_version" not in data
        and isinstance(data.get("module"), dict)
    ):
        data = data["module"]
    if not isinstance(data, dict):
        _die("module file must hold a TVL module mapping")
    return dict(data)


@click.group()
def tune() -> None:
    """Compile, check and simulate Continuous Tuning policies (TVL 1.7)."""


@tune.command("compile")
@click.argument("target")
@click.option("-o", "--output", "output", type=click.Path(path_type=Path), default=None)
def compile_command(target: str, output: Path | None) -> None:
    """Emit the TVL module of ``MODULE.py:FUNCTION`` (decorated with observe(tune=...))."""
    file_part, sep, function = target.rpartition(":")
    if not sep or not file_part or not function:
        _die("target must be MODULE.py:FUNCTION")
    from traigent.cli.main import _load_user_module, _resolve_user_cli_path

    resolved = _resolve_user_cli_path(Path(file_part), "File path", must_exist=True)
    sys.path.insert(0, str(resolved.parent))
    try:
        module = _load_user_module(resolved)
    except Exception:  # noqa: BLE001 - user module errors are not echoed
        _die("module_load_failed")
    finally:
        if str(resolved.parent) in sys.path:
            sys.path.remove(str(resolved.parent))
    fn = getattr(module, function, None)
    state: Any = getattr(fn, "tuning", None)
    if state is None:
        _die("function_not_found_or_has_no_tuning_policy")
    text = yaml.safe_dump(state.tvl, sort_keys=True, default_flow_style=False)
    if output is None:
        click.echo(text, nl=False)
    else:
        output.write_text(text, encoding="utf-8")
        click.echo(f"wrote {output}")


@tune.command("check")
@click.argument(
    "file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--json", "as_json", is_flag=True, help="Print diagnostics as JSON.")
def check_command(file_path: Path, as_json: bool) -> None:
    """Run the TVL validator over FILE; exit 1 when it reports errors."""
    _require_validator()
    diagnostics = lint_document(_module_doc(file_path))
    if as_json:
        click.echo(
            json.dumps([d.to_dict() for d in diagnostics], indent=2, sort_keys=True)
        )
    else:
        for d in diagnostics:
            click.echo(f"{d.severity} {d.code}: {d.message}")
        if not diagnostics:
            click.echo("ok")
    if any(d.severity == "error" for d in diagnostics):
        raise click.exceptions.Exit(EXIT_ERRORS)


@tune.command("simulate")
@click.argument(
    "file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--history",
    "history_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--format", "fmt", type=click.Choice(["yaml", "json"]), default="yaml")
def simulate_command(file_path: Path, history_path: Path, fmt: str) -> None:
    """Run the TVL reference relation over a recorded history; print derived facts."""
    _require_validator()
    module = _module_doc(file_path)
    history = _load_yaml(history_path, "history file")
    if isinstance(history, dict) and "steps" not in history:
        inner = history.get("input", {})
        if isinstance(inner, dict) and isinstance(inner.get("history"), dict):
            history = inner["history"]
    if not isinstance(history, dict):
        _die("history file must hold a history mapping")
    errors = [d for d in lint_document(module) if d.severity == "error"]
    if errors:
        for d in errors:
            click.echo(f"{d.severity} {d.code}: {d.message}", err=True)
        raise click.exceptions.Exit(EXIT_ERRORS)
    try:
        trace = conditions_trace(module, history)
    except TuningError as exc:
        _die(
            str(exc),
            EXIT_UNAVAILABLE
            if exc.code == E_TVL_VALIDATOR_UNAVAILABLE
            else EXIT_ERRORS,
        )
    except Exception as exc:  # noqa: BLE001 - TVL closed errors carry a code
        _die(str(getattr(exc, "code", None) or "simulation_failed"))
    if fmt == "json":
        click.echo(json.dumps(trace, indent=2, sort_keys=True))
    else:
        click.echo(
            yaml.safe_dump(trace, sort_keys=False, default_flow_style=False), nl=False
        )


# ---------------------------------------------------------------------------------
# The local loop controller: record facts, step, approve, accept, status.
# ---------------------------------------------------------------------------------
def _controller_options(func: Any) -> Any:
    for option in reversed(
        [
            click.option(
                "--store",
                "store",
                type=click.Path(file_okay=False, path_type=Path),
                default=None,
                help="Store directory (default .traigent/tuning/<environment>/<agent>/).",
            ),
            click.option(
                "--tvl",
                "tvl_path",
                type=click.Path(exists=True, dir_okay=False, path_type=Path),
                default=None,
                help="TVL 1.7 module file the loop runs.",
            ),
            click.option(
                "--target",
                "target",
                default=None,
                help="MODULE.py:FUNCTION carrying the tuning policy (and .optimize()).",
            ),
            click.option("--agent", "agent", default=None, help="Agent reference."),
            click.option(
                "--authority",
                "authority_path",
                type=click.Path(exists=True, dir_okay=False, path_type=Path),
                envvar="TRAIGENT_TUNING_AUTHORITY",
                default=None,
                help=(
                    "Authority config (approvers, trusted event issuers). Needed "
                    "by approve and accept; without it no event is authenticated."
                ),
            ),
            click.option(
                "--now", "now_text", default=None, help="UTC instant (testing)."
            ),
        ]
    ):
        func = option(func)
    return func


def _now(text: str | None) -> Any:
    from datetime import datetime

    if text is None:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        _die("--now must be a UTC instant like 2026-11-20T10:00:00Z")


def _build_controller(
    store: Path | None,
    tvl_path: Path | None,
    target: str | None,
    agent: str | None,
    authority_path: Path | None = None,
) -> Any:
    from traigent.tuning.authority import load_authority
    from traigent.tuning.controller import Controller, traigent_optimize_fn
    from traigent.tuning.policy import Tuning
    from traigent.tuning.state import (
        TuningState,
        find_optimized,
        initial_config_of,
    )
    from traigent.utils.env_config import resolve_environment_name

    _require_validator()
    optimize_fn = None
    initial_config: dict[str, Any] = {}
    if target is not None:
        file_part, _, function = target.rpartition(":")
        from traigent.cli.main import _load_user_module, _resolve_user_cli_path

        resolved = _resolve_user_cli_path(Path(file_part), "File path", must_exist=True)
        sys.path.insert(0, str(resolved.parent))
        try:
            fn = getattr(_load_user_module(resolved), function, None)
        except Exception:  # noqa: BLE001 - user module errors are not echoed
            _die("module_load_failed")
        finally:
            if str(resolved.parent) in sys.path:
                sys.path.remove(str(resolved.parent))
        state: Any = getattr(fn, "tuning", None)
        if state is None:
            _die("function_not_found_or_has_no_tuning_policy")
        optimized = getattr(state, "optimized", None) or find_optimized(fn)
        if optimized is not None:
            optimize_fn = traigent_optimize_fn(optimized)
            initial_config = initial_config_of(optimized)
        if not initial_config:
            click.echo(
                "warning: no default configuration found for the target; "
                "the initial head config is {}",
                err=True,
            )
    elif tvl_path is not None:
        doc = _module_doc(tvl_path)
        diagnostics = lint_document(doc)  # TVL decides; its codes are shown unchanged
        failing = [d for d in diagnostics if d.is_error]
        if failing:
            _die(str(TvlValidationFailed(failing)))
        try:
            policy = Tuning.from_tvl(doc)
        except TuningError:
            # outside the builder's shape: the controller reads ``state.tvl`` and
            # a ``--tvl`` file carries no local checkers or holdout, so the policy
            # is a neutral one
            policy = Tuning()
        state = TuningState(
            agent=agent,
            policy=policy,
            tvl=doc,
            validated=not any(d.is_error for d in diagnostics),
            diagnostics=diagnostics,
        )
    else:
        _die("one of --tvl or --target is required")
    if agent is not None:
        state.agent = agent
    if state.agent is None and store is None:
        _die("--agent or --store is required")
    try:
        authority, verifier = (
            load_authority(authority_path) if authority_path else (None, None)
        )
        return Controller(
            state,
            store,
            optimize_fn,
            initial_config=initial_config or None,
            environment=getattr(state, "environment", None)
            or resolve_environment_name(default=None)
            or "production",
            authority=authority,
            event_verifier=verifier,
        )
    except TuningError as exc:
        _die(
            str(exc),
            EXIT_UNAVAILABLE
            if exc.code == E_TVL_VALIDATOR_UNAVAILABLE
            else EXIT_ERRORS,
        )


def _guard(call: Any) -> Any:
    try:
        return call()
    except TuningError as exc:
        _die(
            str(exc),
            EXIT_UNAVAILABLE
            if exc.code == E_TVL_VALIDATOR_UNAVAILABLE
            else EXIT_ERRORS,
        )


def _emit(value: Any) -> None:
    click.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _record_command(kind: str) -> Any:
    @tune.command(kind, help=f"Record a {kind} fact from FILE (JSON or YAML).")
    @click.argument(
        "file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    )
    @_controller_options
    def command(
        file_path: Path,
        store: Path | None,
        tvl_path: Path | None,
        target: str | None,
        agent: str | None,
        authority_path: Path | None,
        now_text: str | None,
    ) -> None:
        controller = _build_controller(store, tvl_path, target, agent, authority_path)
        data = _load_yaml(file_path, f"{kind} file")
        if not isinstance(data, dict):
            _die(f"{kind} file must hold a mapping")
        record = (
            controller.record_event if kind == "event" else controller.record_window
        )
        result = _guard(lambda: record(data, _now(now_text)))
        if kind == "event":
            if not result.authenticated:
                click.echo(
                    "warning: the event is not authenticated (no trusted issuer "
                    "signature); it was recorded as rejected(unauthenticated) and "
                    "fires nothing",
                    err=True,
                )
            _emit(
                {
                    "recorded": kind,
                    "instant": result.instant,
                    "authenticated": result.authenticated,
                }
            )
        else:
            _emit({"recorded": kind, "instant": result})

    return command


_record_command("event")
_record_command("window")


@tune.command("step")
@click.option(
    "--abandon-interrupted", is_flag=True, help="Cancel an interrupted attempt."
)
@click.option(
    "--close-ingestion",
    is_flag=True,
    help=(
        "Assert that no more events of the current second will arrive, so work may "
        "start in it (by default nothing is spent in the second an event was recorded)."
    ),
)
@_controller_options
def step_command(
    abandon_interrupted: bool,
    close_ingestion: bool,
    store: Path | None,
    tvl_path: Path | None,
    target: str | None,
    agent: str | None,
    authority_path: Path | None,
    now_text: str | None,
) -> None:
    """Derive facts over the full history and act on them (optimize)."""
    controller = _build_controller(store, tvl_path, target, agent, authority_path)
    report = _guard(
        lambda: controller.step(
            _now(now_text),
            abandon_interrupted=abandon_interrupted,
            ingestion_closed=True if close_ingestion else None,
        )
    )
    _emit(
        {
            "approved": report.approved,
            "started": report.started,
            "finished": report.finished,
            "deferred": report.deferred,
            "awaiting_approval": report.awaiting_approval,
            "head_written": report.head_written,
        }
    )


@tune.command("approve")
@click.argument("occurrence")
@click.option("--approver", required=True, help="Opaque approver reference.")
@_controller_options
def approve_command(
    occurrence: str,
    approver: str,
    store: Path | None,
    tvl_path: Path | None,
    target: str | None,
    agent: str | None,
    authority_path: Path | None,
    now_text: str | None,
) -> None:
    """Approve run: allow the optimization of OCCURRENCE to spend (needs --authority).

    OCCURRENCE is response/rule/scope_kind/scope/j; ``tune status`` shows its
    trigger and spend limit.  The approver must be on the authority's list; with a
    local-trusted authority the record is labelled local-trusted (no credential is
    checked; production needs the platform to authenticate the person).
    """
    controller = _build_controller(store, tvl_path, target, agent, authority_path)
    authority = _guard(controller.require_authority)
    _guard(
        lambda: controller.approve_optimization(occurrence, approver, _now(now_text))
    )
    _emit({"approved": occurrence, "authentication": authority.mode})


@tune.command("accept")
@click.argument("deploy")
@click.option("--generation", "generation", type=int, required=True)
@click.option("--approver", required=True, help="Opaque approver reference.")
@click.option("--reject", is_flag=True, help="Record a rejection instead.")
@_controller_options
def accept_command(
    deploy: str,
    generation: int,
    approver: str,
    reject: bool,
    store: Path | None,
    tvl_path: Path | None,
    target: str | None,
    agent: str | None,
    authority_path: Path | None,
    now_text: str | None,
) -> None:
    """Accept release: accept (or reject) the deploy awaiting DEPLOY (needs --authority).

    GENERATION is the head generation you saw in ``tune status``.  The result is the
    committed one: ``written`` means the serving head now has the new generation.
    """
    controller = _build_controller(store, tvl_path, target, agent, authority_path)
    authority = _guard(controller.require_authority)
    outcome = _guard(
        lambda: controller.accept(
            deploy,
            approver,
            generation,
            "reject" if reject else "accept",
            _now(now_text),
        )
    )
    _emit(
        {
            "acceptance": outcome.acceptance_id,
            "result": outcome.result,
            "reason": outcome.reason,
            "generation": outcome.generation,
            "authentication": authority.mode,
        }
    )
    if not outcome.ok:
        raise click.exceptions.Exit(EXIT_ERRORS)


@tune.command("status")
@click.option("--text", "as_text", is_flag=True, help="Print a readable summary.")
@_controller_options
def status_command(
    as_text: bool,
    store: Path | None,
    tvl_path: Path | None,
    target: str | None,
    agent: str | None,
    authority_path: Path | None,
    now_text: str | None,
) -> None:
    """Head, open attempts and pending deploys, with what each human decision shows.

    ``approve_run`` (Approve run): the trigger and the spend limit.
    ``accept_release`` (Accept release): incumbent, candidate, holdout counts, gate
    result and the observed head generation.
    """
    controller = _build_controller(store, tvl_path, target, agent, authority_path)
    status = _guard(controller.status)
    if as_text:
        click.echo(_status_text(status), nl=False)
    else:
        _emit(status)


def _status_text(status: dict[str, Any]) -> str:
    head = status["head"]
    lines = [f"head: generation {head['generation']} ({head['fingerprint']})"]
    for attempt in status["open_attempts"]:
        lines.append(f"attempt {attempt['attempt']}: {attempt['state']}")
        if attempt.get("approval"):
            lines.append(f"  {attempt['approval']}")
        view = attempt.get("approve_run")
        if view:
            lines.append(f"  Approve run  {attempt['occurrence']}")
            lines.append(
                f"    trigger:     {json.dumps(view['trigger'], sort_keys=True)}"
            )
            limit = view["spend_limit"]
            lines.append(
                "    spend limit: "
                + (json.dumps(limit, sort_keys=True) if limit else "none configured")
            )
            if view["admits"]:
                lines.append(
                    f"    admits:      {json.dumps(view['admits'], sort_keys=True)}"
                )
    for deploy in status["pending_deploys"]:
        view = deploy["accept_release"]
        lines.append(f"Accept release  {deploy['occurrence']}")
        inc = view["incumbent"]
        lines.append(
            f"    incumbent:  generation {inc['generation']} ({inc['fingerprint']})"
        )
        lines.append(f"    candidate:  {view['candidate']['fingerprint']}")
        lines.append(f"    code version: {view.get('code_version')}")
        hold = view["holdout"]
        lines.append(
            f"    holdout:    {hold['rows']} rows, {hold['unresolved_rows']} unresolved, "
            f"counts {json.dumps(hold['counts'], sort_keys=True)}, "
            f"used before: {hold['prior_uses']}"
        )
        gate = view["gate"]
        lines.append(
            f"    gate:       {gate['decision']} (comparison {gate['legacy_decision']}, "
            f"monitors {json.dumps(gate['monitors'], sort_keys=True)})"
        )
        lines.append(
            f"    observed generation: {view['observed_generation']} "
            f"(gate measured it: {view['gate_incumbent_current']})"
        )
    return "\n".join(lines) + "\n"
