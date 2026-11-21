"""CLI: ``traigent release ACCEPTANCE_ID`` and ``traigent release rollback GENERATION``.

A release moves an agent head to a configuration a person has accepted; it is not
``apply_best_config`` (a developer shortcut with no gate and no acceptance).
Output carries closed codes only; server text is never echoed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click

from traigent.runtime.client import RuntimeClient
from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.release import (
    NO_GATE_RUN,
    ReleaseReceipt,
    ReleaseRefused,
    release_agent_head,
    rollback_agent_head,
)


class _ReleaseGroup(click.Group):
    """``release ACCEPTANCE_ID`` runs the default command; ``release rollback`` is a subcommand."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["run", *args]
        elif args and args[0].startswith("-") and args[0] not in ("--help", "-h"):
            args = ["run", *args]
        return list(super().parse_args(ctx, args))


def _fail(exc: ReleaseRefused) -> None:
    click.echo(f"error: {exc}", err=True)
    raise click.exceptions.Exit(1)


def _show(verb: str, receipt: ReleaseReceipt) -> None:
    click.echo(
        f"{verb}: environment={receipt.environment} generation={receipt.generation}"
        + (
            f" version={receipt.best_config_version}"
            if receipt.best_config_version is not None
            else ""
        )
    )


def _client(project_id: str | None) -> RuntimeClient:
    project = (project_id or os.environ.get("TRAIGENT_PROJECT_ID") or "").strip()
    if not project:
        raise ReleaseRefused("project_required")
    return RuntimeClient(project)


def _agent_ref(agent_ref: str | None) -> str:
    value = (agent_ref or os.environ.get("TRAIGENT_AGENT_REF") or "").strip()
    if not value:
        raise ReleaseRefused("agent_ref_required")
    return value


@click.group("release", cls=_ReleaseGroup)
def release() -> None:
    """Release an accepted configuration to an agent head, or undo a release.

    \b
    traigent release ACCEPTANCE_ID --expected-head-generation N --file app.py -f agent
    traigent release rollback GENERATION --expected-head-generation N --environment production
    """


@release.command("run", hidden=True)
@click.argument("acceptance_id")
@click.option("--expected-head-generation", type=click.IntRange(min=0), default=None)
@click.option("--file", "file_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--function", "-f", "function", default=None, help="Function name in --file."
)
@click.option("--project-id", default=None, help="Env: TRAIGENT_PROJECT_ID.")
@click.option("--agent-ref", default=None, help="Env: TRAIGENT_AGENT_REF.")
@click.option("--environment", default=None, help="Environment for the head read.")
@click.option("--agent-code-version", default=None)
@click.option("--configuration-space-digest", default=None)
@click.option("--gate-result-digest", default=None, help="Default: no_gate_run.")
def release_cmd(
    acceptance_id: str,
    expected_head_generation: int | None,
    file_path: Path | None,
    function: str | None,
    project_id: str | None,
    agent_ref: str | None,
    environment: str | None,
    agent_code_version: str | None,
    configuration_space_digest: str | None,
    gate_result_digest: str | None,
) -> None:
    """Release the configuration a person accepted (needs a recorded acceptance)."""
    try:
        if file_path is not None:
            if not function:
                raise click.UsageError("--function is required with --file")
            from traigent.cli.requirements_commands import _find_function

            fn: Any = _find_function(file_path, function)
            receipt = fn.release(
                acceptance_id=acceptance_id,
                expected_head_generation=expected_head_generation,
                agent_ref=agent_ref,
                project_id=project_id,
                environment=environment,
                agent_code_version=agent_code_version,
                configuration_space_digest=configuration_space_digest,
                gate_result_digest=gate_result_digest,
            )
        else:
            if (
                expected_head_generation is None
                or agent_code_version is None
                or configuration_space_digest is None
            ):
                raise click.UsageError(
                    "without --file, give --expected-head-generation, "
                    "--agent-code-version and --configuration-space-digest"
                )
            receipt = release_agent_head(
                _client(project_id),
                _agent_ref(agent_ref),
                acceptance_id=acceptance_id,
                expected_head_generation=expected_head_generation,
                agent_code_version=agent_code_version,
                configuration_space_digest=configuration_space_digest,
                gate_result_digest=gate_result_digest or NO_GATE_RUN,
            )
    except ReleaseRefused as exc:
        _fail(exc)
        return
    except RuntimeRequirementError as exc:
        click.echo(f"error: {exc.code}", err=True)
        raise click.exceptions.Exit(1) from None
    _show("released", receipt)


@release.command("rollback")
@click.argument("generation", type=click.IntRange(min=0))
@click.option("--expected-head-generation", type=click.IntRange(min=0), required=True)
@click.option("--environment", required=True)
@click.option("--project-id", default=None, help="Env: TRAIGENT_PROJECT_ID.")
@click.option("--agent-ref", default=None, help="Env: TRAIGENT_AGENT_REF.")
def rollback_cmd(
    generation: int,
    expected_head_generation: int,
    environment: str,
    project_id: str | None,
    agent_ref: str | None,
) -> None:
    """Undo a release: move the head back to GENERATION (needs a backend with rollback)."""
    try:
        receipt = rollback_agent_head(
            _client(project_id),
            _agent_ref(agent_ref),
            environment=environment.strip().lower(),
            target_generation=generation,
            expected_head_generation=expected_head_generation,
        )
    except ReleaseRefused as exc:
        _fail(exc)
        return
    _show("rolled back", receipt)
