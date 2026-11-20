"""``traigent regression``: keep the examples a run got wrong as a regression set.

# Traceability: CONC-Layer-Data FUNC-STORAGE FUNC-ANALYTICS

Everything runs locally; no command here makes a network call.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from traigent.utils.optimization_result_persistence import (
    UNRESTORED_OPTIMIZATION_ID_PREFIX,
)
from traigent.utils.outcome_matrix import select_failure_basis
from traigent.utils.regression_set import (
    DEFAULT_REGRESSION_DIR,
    ON_ID_CONFLICT_CHOICES,
    RegressionSetError,
    RunEvidence,
    config_digest_for,
    merge_failures,
)


@click.group("regression")
def regression() -> None:
    """Keep failing examples as a regression set committed next to your code.

    The set is the file <dir>/regression.jsonl. It holds the example content
    (inputs and expected outputs) exactly as it is in your dataset: nothing in
    it is redacted, so a regression test can rerun the exact example. Nothing
    is uploaded; everything stays local. Review the file before you commit it,
    and do not commit it to a public repository unless the examples are public.
    """


def _run_identity(run_result: Any) -> str:
    """A stable id for the saved run, independent of its name and import date.

    The run's optimization id when it was persisted; otherwise (results saved
    before ids were persisted) a digest of the run's outcome matrix.
    """
    optimization_id = getattr(run_result, "optimization_id", None)
    if (
        isinstance(optimization_id, str)
        and optimization_id
        and not optimization_id.startswith(UNRESTORED_OPTIMIZATION_ID_PREFIX)
    ):
        return optimization_id
    text = json.dumps(
        run_result.example_matrix, sort_keys=True, default=str, separators=(",", ":")
    )
    return "matrix-sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _failure_basis(
    run_result: Any, failing_ids: list[str], threshold: float, mode: str
) -> dict[str, tuple[str, str]]:
    """What decided each failure; same winning-column rule as ``failing_examples``."""
    winning_trial_ids = None
    metadata = getattr(run_result, "metadata", None)
    summary = metadata.get("session_summary") if isinstance(metadata, dict) else None
    if isinstance(summary, dict):
        stamped = summary.get("winning_trial_ids")
        if isinstance(stamped, list) and stamped:
            winning_trial_ids = stamped
    return select_failure_basis(
        run_result.example_matrix,
        failing_ids,
        threshold=threshold,
        mode=mode,
        best_config=run_result.best_config,
        winning_trial_ids=winning_trial_ids,
    )


@regression.command("add")
@click.argument("run")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    help="The original JSONL/JSON dataset the run evaluated",
)
@click.option(
    "--dir",
    "directory",
    default=DEFAULT_REGRESSION_DIR,
    show_default=True,
    help="Directory holding regression.jsonl (created if absent)",
)
@click.option(
    "--mode",
    type=click.Choice(["all_configs", "best_config"]),
    default="best_config",
    show_default=True,
    help="best_config: failed in the winner's trials; all_configs: failed in every trial",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Per-example score below which an example fails (equality passes)",
)
@click.option(
    "--on-id-conflict",
    type=click.Choice(list(ON_ID_CONFLICT_CHOICES)),
    default="refuse",
    show_default=True,
    help=(
        "When a stored example id now names different content: refuse writes "
        "nothing; replace overwrites the stored row; keep-both stores the new "
        "content under a distinct id"
    ),
)
@click.option(
    "--storage-dir", "-d", default=".traigent", help="Traigent storage directory"
)
def regression_add(
    run: str,
    dataset_path: str,
    directory: str,
    mode: str,
    threshold: float,
    on_id_conflict: str,
    storage_dir: str,
) -> None:
    """Merge the examples RUN got wrong into the regression set.

    RUN is a saved result name (as in 'traigent results show'). Failing
    examples are selected and joined to rows of --dataset exactly as in
    'traigent results export-failures'. A new example becomes a row labelled
    seen_once; an example already stored gains a provenance entry (run id, mode,
    threshold, config digest, the field that decided the failure) instead of a
    duplicate row. Adding the same saved run again adds nothing.

    The file holds the example content exactly as it is in your dataset.
    Nothing in it is redacted (a regression test must rerun the exact
    example), and nothing is uploaded: everything stays local. Review the file
    before you commit it to your own repository.

    Example:
        traigent regression add my_run --dataset eval.jsonl
    """
    # Imported here: traigent.cli.main registers this group at import time.
    from traigent.cli.main import _resolve_user_cli_path, _select_run_failures
    from traigent.evaluators.base import Dataset
    from traigent.utils.regression_dataset import (
        UnresolvedExampleIdError,
        select_examples_with_run_ids,
    )

    out_dir = _resolve_user_cli_path(Path(directory), "Regression dir", for_write=True)
    if out_dir.exists() and not out_dir.is_dir():
        raise click.ClickException(f"Regression dir is not a directory: {out_dir}")

    try:
        dataset = Dataset.from_jsonl(dataset_path)
    except Exception as exc:
        # Closed message: a loader error may quote part of a row.
        raise click.ClickException(
            f"Cannot load dataset '{dataset_path}' ({type(exc).__name__}); check "
            "that it is valid JSONL or JSON with an input field on every row"
        ) from exc

    run_result, failing_ids = _select_run_failures(run, storage_dir, threshold, mode)
    if not failing_ids:
        click.echo("No failing examples; regression set unchanged")
        return

    try:
        subset = select_examples_with_run_ids(dataset, failing_ids, run_result)
    except UnresolvedExampleIdError as exc:
        raise click.ClickException(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"Cannot select failing examples ({type(exc).__name__}); the dataset "
            "rows do not match the run"
        ) from exc

    try:
        evidence = RunEvidence(
            run_id=_run_identity(run_result),
            run_name=run,
            imported=datetime.now(UTC).date().isoformat(),
            mode=mode,
            threshold=threshold,
            config_digest=config_digest_for(mode, run_result.best_config),
            basis=_failure_basis(run_result, failing_ids, threshold, mode),
        )
        report = merge_failures(
            out_dir,
            subset,
            evidence=evidence,
            on_id_conflict=on_id_conflict,  # type: ignore[arg-type]
        )
    except RegressionSetError as exc:
        raise click.ClickException(str(exc)) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(
            f"Cannot write regression set: {type(exc).__name__}"
        ) from exc

    click.echo(
        f"Added {report.rows_added} row(s), {report.provenance_added} new "
        f"provenance entr{'y' if report.provenance_added == 1 else 'ies'}, "
        f"replaced {report.rows_replaced}; file: {report.path}"
    )
    if report.gitignore_updated:
        click.echo(
            f"Added the lock file name to {report.path.parent / '.gitignore'} "
            "so it stays out of version control."
        )
    click.echo(
        "Note: this file holds unredacted example content and the examples' "
        "metadata. Review it before you commit it."
    )


def register_regression_commands(cli: click.Group) -> None:
    """Register the ``traigent regression`` group on the CLI group."""
    cli.add_command(regression)
