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

from traigent.tuning.errors import E_TVL_VALIDATOR_UNAVAILABLE, TuningError
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
                help="Store directory (default .traigent/tuning/<agent>/).",
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
    store: Path | None, tvl_path: Path | None, target: str | None, agent: str | None
) -> Any:
    from traigent.tuning.controller import Controller, traigent_optimize_fn
    from traigent.tuning.policy import Tuning
    from traigent.tuning.state import TuningState

    _require_validator()
    optimize_fn = None
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
        if callable(getattr(fn, "optimize", None)):
            optimize_fn = traigent_optimize_fn(fn)
    elif tvl_path is not None:
        doc = _module_doc(tvl_path)
        state = TuningState(
            agent=agent, policy=Tuning.from_tvl(doc), tvl=doc, validated=True
        )
    else:
        _die("one of --tvl or --target is required")
    if agent is not None:
        state.agent = agent
    if state.agent is None and store is None:
        _die("--agent or --store is required")
    try:
        return Controller(state, store, optimize_fn)
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
        now_text: str | None,
    ) -> None:
        controller = _build_controller(store, tvl_path, target, agent)
        data = _load_yaml(file_path, f"{kind} file")
        if not isinstance(data, dict):
            _die(f"{kind} file must hold a mapping")
        record = (
            controller.record_event if kind == "event" else controller.record_window
        )
        instant = _guard(lambda: record(data, _now(now_text)))
        _emit({"recorded": kind, "instant": instant})

    return command


_record_command("event")
_record_command("window")


@tune.command("step")
@click.option(
    "--abandon-interrupted", is_flag=True, help="Cancel an interrupted attempt."
)
@_controller_options
def step_command(
    abandon_interrupted: bool,
    store: Path | None,
    tvl_path: Path | None,
    target: str | None,
    agent: str | None,
    now_text: str | None,
) -> None:
    """Derive facts over the full history and act on them (optimize, move the head)."""
    controller = _build_controller(store, tvl_path, target, agent)
    report = _guard(
        lambda: controller.step(_now(now_text), abandon_interrupted=abandon_interrupted)
    )
    _emit(
        {
            "approved": report.approved,
            "started": report.started,
            "finished": report.finished,
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
    now_text: str | None,
) -> None:
    """Approve the optimization of OCCURRENCE (response/rule/scope_kind/scope/j)."""
    controller = _build_controller(store, tvl_path, target, agent)
    _guard(
        lambda: controller.approve_optimization(occurrence, approver, _now(now_text))
    )
    _emit({"approved": occurrence})


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
    now_text: str | None,
) -> None:
    """Accept (or reject) the deploy awaiting DEPLOY at head GENERATION."""
    controller = _build_controller(store, tvl_path, target, agent)
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
        }
    )
    if not outcome.ok:
        raise click.exceptions.Exit(EXIT_ERRORS)


@tune.command("status")
@_controller_options
def status_command(
    store: Path | None,
    tvl_path: Path | None,
    target: str | None,
    agent: str | None,
    now_text: str | None,
) -> None:
    """Head, open attempts and pending deploys."""
    controller = _build_controller(store, tvl_path, target, agent)
    _emit(_guard(controller.status))
