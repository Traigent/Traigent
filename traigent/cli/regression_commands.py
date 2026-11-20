"""``traigent regression``: keep the examples a run got wrong as a regression set.

# Traceability: CONC-Layer-Data FUNC-STORAGE FUNC-ANALYTICS

Everything runs locally; no command here makes a network call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from traigent.utils.regression_set import (
    DEFAULT_REGRESSION_DIR,
    ON_ID_CONFLICT_CHOICES,
    RegressionSetError,
    merge_failures,
)


@click.group("regression")
def regression() -> None:
    """Keep failing examples as a regression set committed next to your code.

    The set is the file <dir>/regression.jsonl. It holds YOUR example content
    (inputs and expected outputs, with only the SDK's pattern redaction
    applied) and is meant to be committed to your own repository. Do not
    commit it to a public repository unless the examples are public.
    Everything stays local: nothing is uploaded.
    """


def _metric_name(run_result: Any) -> str:
    objectives = getattr(run_result, "objectives", None) or []
    first = objectives[0] if objectives else None
    return first if isinstance(first, str) else "accuracy"


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
    seen_once; an example already stored gains a provenance entry (run, date,
    config digest, metric) instead of a duplicate row.

    The file holds your example content and is meant to be committed to your
    own repository. Everything stays local: nothing is uploaded.

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
        raise click.ClickException(
            f"Cannot load dataset '{dataset_path}': {exc}"
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
        raise click.ClickException(f"Cannot select failing examples: {exc}") from exc

    try:
        report = merge_failures(
            out_dir,
            subset,
            run=run,
            date=datetime.now(UTC).date().isoformat(),
            config=run_result.best_config if mode == "best_config" else None,
            metric=_metric_name(run_result),
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


def register_regression_commands(cli: click.Group) -> None:
    """Register the ``traigent regression`` group on the CLI group."""
    cli.add_command(regression)
