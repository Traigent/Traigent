"""``traigent regression``: keep the examples a run got wrong as a regression set.

# Traceability: CONC-Layer-Data FUNC-STORAGE FUNC-ANALYTICS

Everything runs locally; no command here makes a network call.

``confirm`` and ``check`` import and run the user's module in this process. Any
``BaseException`` that escapes it other than ``KeyboardInterrupt`` (a module
or function calling ``sys.exit(0)``, say) ends the command as the closed
inconclusive result, never as a success: success comes only from the verdict
logic at the end of ``check``. Limit: a process that ends itself with
``os._exit`` (or is killed) cannot be caught.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import click

from traigent.utils import regression_confirm as _confirm_engine
from traigent.utils.hashing import generate_config_hash
from traigent.utils.optimization_result_persistence import (
    UNRESTORED_OPTIMIZATION_ID_PREFIX,
)
from traigent.utils.outcome_matrix import select_failure_basis
from traigent.utils.output_boundary import CapturedOutput, capture_output
from traigent.utils.regression_confirm import (
    DEFAULT_CHECK_REPEATS,
    DEFAULT_CONFIRM_REPEATS,
    IdsNotAssignableError,
    Tally,
    example_id_of,
    observation_for,
    row_status,
    run_rows,
    tally,
)
from traigent.utils.regression_set import (
    CONFIRMED_SCHEMA_VERSION,
    DEFAULT_REGRESSION_DIR,
    ON_ID_CONFLICT_CHOICES,
    RegressionSetError,
    RunEvidence,
    config_digest_for,
    merge_failures,
    read_regression_rows,
    regression_file,
    update_rows_locked,
)
from traigent.utils.regression_status import (
    DISARM_REASONS,
    OUTCOME_DISARMED,
    STATUS_ARMED,
    STATUS_INTERMITTENT,
    STATUS_KNOWN_FAILURE,
    STATUS_UNCONFIRMED,
    STATUSES,
    append_observation,
    content_changed_since_measured,
    status_when_last_measured,
)


@click.group("regression")
def regression() -> None:
    """Keep failing examples as a regression set committed next to your code.

    The set is the file <dir>/regression.jsonl. It holds the example content
    (inputs and expected outputs) exactly as it is in your dataset: nothing in
    it is redacted, so a regression test can rerun the exact example. Nothing
    is uploaded; everything stays local. Review the file before you commit it,
    and do not commit it to a public repository unless the examples are public.

    \b
    add      writes the file: merges failing examples from a saved run.
    confirm  writes the file: re-runs rows under a configuration and records
             what it saw (rows are marked, never deleted).
    check    writes nothing: re-runs the armed rows and exits non-zero on a
             regression, for CI.
    """


def _outcome_state(cell: Any) -> str:
    """``pass``, ``fail`` or ``error`` for one outcome-matrix cell."""
    if not isinstance(cell, dict):
        return "unknown"
    if cell.get("error") is not None:
        return "error"
    return "pass" if cell.get("success") is True else "fail"


def _run_identity(run_result: Any, run_name: str) -> str:
    """A stable id for the saved run, independent of its name and import date.

    The run's optimization id when it was persisted. A result saved before ids
    were persisted has none, so its identity is built only from what the saved
    artifact keeps: the saved result's name plus a digest, in a canonical
    order, of each trial's config hash and its per-example pass/fail/error
    cells. Nothing time-dependent or object-dependent goes in (the matrix's
    ``created_at`` is the time it was rebuilt, so it is left out).

    Residual: two different saved results cannot share one name, so name plus
    outcome digest is unique per saved artifact and importing the same artifact
    again adds nothing. The same run re-saved under another name has a new name
    and counts as new evidence.
    """
    optimization_id = getattr(run_result, "optimization_id", None)
    saved_name = run_name
    if isinstance(optimization_id, str) and optimization_id:
        if not optimization_id.startswith(UNRESTORED_OPTIMIZATION_ID_PREFIX):
            return optimization_id
        saved_name = optimization_id[len(UNRESTORED_OPTIMIZATION_ID_PREFIX) :]
    matrix = run_result.example_matrix
    examples = matrix.get("examples") or []
    columns = []
    for trial in matrix.get("trials") or []:
        trial_id = trial.get("trial_id")
        cells = sorted(
            (str(row["example_id"]), _outcome_state(row["cells"].get(trial_id)))
            for row in examples
            if trial_id in row["cells"]
        )
        columns.append([trial.get("config_hash"), cells])
    columns.sort(key=lambda column: json.dumps(column, sort_keys=True))
    text = json.dumps(
        [saved_name, columns], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return "legacy-sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
            run_id=_run_identity(run_result, run),
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


# ------------------------------------------------------- confirm and check

#: ``check`` exit codes.
#: ``not_gated`` key for rows that were gated when last measured but whose
#: content has changed since.
CONTENT_CHANGED = "content_changed"
#: Closed reason listed with an inconclusive row whose scored attempts tied.
REASON_TIE = "tie"
CHECK_OK = 0
CHECK_REGRESSION = 1
CHECK_INCONCLUSIVE = 2
CHECK_BAD_FILE = 3
#: ``confirm`` exit codes (1 is any other failure).
CONFIRM_BAD_INPUT = 2
CONFIRM_BAD_FILE = 3
CONFIRM_STOPPED_EARLY = 4


class _Stop(Exception):
    """An input problem that ends a command with a closed reason and exit code."""

    def __init__(self, code: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.message = message


class _UserOutput:
    """Everything user code and the SDK wrote across a command's windows.

    ``window()`` contains one stretch of user code (module load, evaluation);
    ``report()`` runs once at the end and prints a counts-only line on stderr,
    or, with ``--show-output``, the collected text on stderr. Nothing is ever
    printed on stdout, because the text can hold example content and ``--json``
    owns stdout.
    """

    def __init__(self, show: bool) -> None:
        self.show = show
        self._captured: list[CapturedOutput] = []

    @contextmanager
    def window(self) -> Iterator[None]:
        with capture_output() as captured:
            self._captured.append(captured)
            yield

    def report(self) -> None:
        lines = sum(c.line_count for c in self._captured)
        if lines == 0:
            return
        if self.show:
            click.echo(
                "Output from your code and the SDK (may contain example content):",
                err=True,
            )
            for captured in self._captured:
                text = captured.text()
                if text:
                    click.echo(text, err=True)
            return
        click.echo(
            f"{lines} lines of output from your code and the SDK were suppressed; "
            "rerun with --show-output to see them on stderr",
            err=True,
        )


SHOW_OUTPUT_HELP = (
    "Print the output your code and the SDK produced while running (stdout, "
    "stderr, logs, warnings) on stderr after the run; it may contain example "
    "content. By default it is suppressed and only a line count is printed"
)


def _load_config(config_path: str) -> dict[str, Any]:
    from traigent.utils.secure_path import PathTraversalError, validate_user_path

    try:
        resolved = validate_user_path(Path(config_path).expanduser())
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, PathTraversalError) as exc:
        raise _Stop(
            2,
            "config_invalid",
            f"Cannot read config '{config_path}' ({type(exc).__name__}); it must "
            "be a JSON file holding one object of parameter values",
        ) from exc
    if not isinstance(data, dict):
        raise _Stop(
            2, "config_invalid", f"Config '{config_path}' must hold one JSON object"
        )
    return data


def _load_function(spec: str) -> Any:
    """Load the decorated function named by ``PATH_OR_MODULE:FUNCTION``."""
    from traigent.cli.function_discovery import _resolve_module_path
    from traigent.utils.module_loading import find_optimized_functions, load_user_module

    module_part, _, function_name = spec.rpartition(":")
    if not module_part or not function_name:
        raise _Stop(
            2,
            "module_load_failed",
            f"--module '{spec}' must look like PATH_OR_MODULE:FUNCTION",
        )
    try:
        module = load_user_module(_resolve_module_path(module_part, None))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 - user module code, even SystemExit
        # Closed message: a loader or user-module error may quote content. A
        # module that exits at import (sys.exit(0) included) never evaluated
        # anything, so it is an unusable function, never a success.
        raise _Stop(
            2,
            "module_load_failed",
            f"Cannot load '{module_part}' ({type(exc).__name__})",
        ) from exc
    found = find_optimized_functions(module, {function_name})
    if not found:
        raise _Stop(
            2,
            "module_load_failed",
            f"'{module_part}' has no @traigent.optimize function named "
            f"'{function_name}'",
        )
    return found[0][1]


def _regression_path(directory: str, code: int) -> Path:
    from traigent.utils.secure_path import PathTraversalError, validate_user_path

    try:
        return regression_file(validate_user_path(Path(directory).expanduser()))
    except (PathTraversalError, ValueError) as exc:
        raise _Stop(
            code,
            "file_invalid",
            f"Regression dir is not allowed ({type(exc).__name__})",
        ) from exc


def _read_rows(directory: str, code: int) -> tuple[Path, list[dict[str, Any]]]:
    path = _regression_path(directory, code)
    try:
        return path, read_regression_rows(path)
    except RegressionSetError as exc:
        raise _Stop(code, "file_invalid", str(exc)) from exc


def _exit_with(stop: _Stop, as_json: bool) -> NoReturn:
    if as_json:
        click.echo(
            json.dumps(
                {
                    "schema": "traigent.regression_check/1",
                    "verdict": "error",
                    "exit_code": stop.code,
                    "error": stop.reason,
                },
                sort_keys=True,
            )
        )
        click.echo(stop.message, err=True)
    else:
        click.echo(stop.message, err=True)
    raise SystemExit(stop.code)


def _interval(entry: Any) -> str:
    low, high = entry.pass_rate_low, entry.pass_rate_high
    return "n/a" if low is None else f"[{low}, {high}]"


@regression.command("confirm")
@click.option(
    "--module",
    "module_spec",
    default=None,
    help="PATH_OR_MODULE:FUNCTION of the @traigent.optimize function to run",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    help="JSON file holding the configuration to run the rows under",
)
@click.option(
    "--dir",
    "directory",
    default=DEFAULT_REGRESSION_DIR,
    show_default=True,
    help="Directory holding regression.jsonl",
)
@click.option(
    "--repeats",
    type=click.IntRange(min=2),
    default=DEFAULT_CONFIRM_REPEATS,
    show_default=True,
    help="Times each row runs (at least 2, so a row can be labelled)",
)
@click.option(
    "--max-cost",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help=(
        "Stop before starting more repeats once this many USD are spent; "
        "observations from the finished repeats are still written (exit 4)"
    ),
)
@click.option(
    "--only-id",
    "only_ids",
    multiple=True,
    help="Run only this row id (repeatable); default: every row",
)
@click.option(
    "--disarm",
    "disarm_ids",
    multiple=True,
    help="Return this armed row to known_failure (repeatable); runs nothing",
)
@click.option(
    "--reason",
    type=click.Choice(list(DISARM_REASONS)),
    default=None,
    help="Why a row is disarmed; required with --disarm",
)
@click.option(
    "--show-output",
    "show_output",
    is_flag=True,
    help=SHOW_OUTPUT_HELP,
)
def regression_confirm(
    module_spec: str | None,
    config_path: str | None,
    directory: str,
    repeats: int,
    max_cost: float | None,
    only_ids: tuple[str, ...],
    disarm_ids: tuple[str, ...],
    reason: str | None,
    show_output: bool,
) -> None:
    """Run the regression rows under a configuration and mark what they do.

    WRITES <dir>/regression.jsonl (one atomic rewrite at the end, under the
    file lock; the model calls happen outside the lock). Rows are marked,
    never deleted: each run appends an observation (config digest, date,
    passes, fails, errored, interval) and the row's status becomes one of
    known_failure (never seen passing), intermittent, armed (passed every
    repeat once; from then on a failure is a regression) or unconfirmed
    (errored). An armed row stays armed until you run --disarm ID --reason R.

    Everything is local: the command makes no call to the Traigent backend
    (your function calls your own model provider). Output has counts and ids,
    never example content. Anything your code or the SDK prints while running
    is suppressed (a line count is printed on stderr); --show-output prints it
    on stderr, where it may contain example content.

    Exit codes: 0 done; 1 other failure; 2 bad function, config or options;
    3 regression file missing, corrupt or edited by hand (nothing written);
    4 stopped early by --max-cost (finished repeats were written).

    Example:
        traigent regression confirm --module agent.py:answer --config c.json
    """
    try:
        if disarm_ids:
            _disarm(directory, disarm_ids, reason, module_spec, config_path, only_ids)
            return
        if reason is not None:
            raise click.UsageError("--reason only applies with --disarm")
        if not module_spec or not config_path:
            raise click.UsageError("--module and --config are required")
        _confirm(
            module_spec,
            config_path,
            directory,
            repeats,
            max_cost,
            list(only_ids),
            show_output,
        )
    except _Stop as stop:
        _exit_with(stop, as_json=False)


def _disarm(
    directory: str,
    ids: tuple[str, ...],
    reason: str | None,
    module_spec: str | None,
    config_path: str | None,
    only_ids: tuple[str, ...],
) -> None:
    if reason is None:
        raise click.UsageError(f"--disarm needs --reason ({'|'.join(DISARM_REASONS)})")
    if module_spec or config_path or only_ids:
        raise click.UsageError(
            "--disarm runs nothing: do not combine it with --module, --config "
            "or --only-id"
        )
    _read_rows(directory, CONFIRM_BAD_FILE)
    date = _confirm_engine.today()

    def apply(rows: list[dict[str, Any]]) -> list[str]:
        by_id = {example_id_of(row, i): row for i, row in enumerate(rows)}
        unknown = [eid for eid in ids if eid not in by_id]
        not_armed = [
            eid
            for eid in ids
            if eid in by_id and row_status(by_id[eid]) != STATUS_ARMED
        ]
        if unknown or not_armed:
            raise RegressionSetError(
                f"Nothing was written. Unknown ids: {unknown}; ids that are not "
                f"armed: {not_armed}."
            )
        for eid in ids:
            append_observation(
                by_id[eid]["metadata"]["regression"],
                {
                    "config_digest": None,
                    "content_digest": by_id[eid]["metadata"]["content_digest"],
                    "date": date,
                    "reps": 0,
                    "passes": 0,
                    "fails": 0,
                    "errored": 0,
                    "pass_rate_low": None,
                    "pass_rate_high": None,
                    "outcome": OUTCOME_DISARMED,
                    "reason": reason,
                },
                CONFIRMED_SCHEMA_VERSION,
            )
        return list(ids)

    try:
        done = update_rows_locked(_regression_dir(directory), apply)
    except RegressionSetError as exc:
        raise _Stop(CONFIRM_BAD_FILE, "file_invalid", str(exc)) from exc
    click.echo(
        f"Disarmed {len(done)} row(s) ({reason}): back to {STATUS_KNOWN_FAILURE}; "
        f"ids: {', '.join(done)}"
    )


def _regression_dir(directory: str) -> Path:
    return _regression_path(directory, CONFIRM_BAD_FILE).parent


def _confirm(
    module_spec: str,
    config_path: str,
    directory: str,
    repeats: int,
    max_cost: float | None,
    only_ids: list[str],
    show_output: bool,
) -> None:
    output = _UserOutput(show_output)
    try:
        _confirm_run(
            module_spec, config_path, directory, repeats, max_cost, only_ids, output
        )
    finally:
        output.report()


def _confirm_run(
    module_spec: str,
    config_path: str,
    directory: str,
    repeats: int,
    max_cost: float | None,
    only_ids: list[str],
    output: _UserOutput,
) -> None:
    _, rows = _read_rows(directory, CONFIRM_BAD_FILE)
    entries = [(example_id_of(row, i), row) for i, row in enumerate(rows)]
    known = {eid for eid, _ in entries}
    unknown = [eid for eid in only_ids if eid not in known]
    if unknown:
        raise click.UsageError(f"--only-id names no row: {unknown}")
    selected = [(e, r) for e, r in entries if not only_ids or e in only_ids]
    with output.window():
        function = _load_function(module_spec)
    config = _load_config(config_path)
    digest_of_config = generate_config_hash(config)
    before = {eid: row_status(row) for eid, row in selected}
    content = {eid: row["metadata"]["content_digest"] for eid, row in selected}

    try:
        with output.window():
            run = run_rows(
                function, config, selected, repeats=repeats, max_cost=max_cost
            )
    except IdsNotAssignableError as exc:
        raise _Stop(CONFIRM_BAD_INPUT, "ids_not_assignable", str(exc)) from exc
    except ValueError as exc:
        raise _Stop(
            CONFIRM_BAD_INPUT,
            "config_invalid",
            f"The run was refused ({type(exc).__name__}); is the config complete "
            "and inside the function's configuration space? Nothing was written.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - user code and SDK policy errors
        raise click.ClickException(
            f"The run failed ({type(exc).__name__}). Nothing was written."
        ) from exc
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 - e.g. the function calls sys.exit
        raise _Stop(
            CONFIRM_BAD_INPUT,
            "run_failed",
            f"The run was ended from inside the function ({type(exc).__name__}). "
            "Nothing was written.",
        ) from exc

    observations = {}
    for eid, _row in selected:
        if eid in run.unresolved:
            continue  # results not tied to this row: no observation
        tallied = tally(run.per_attempt, eid)
        if tallied.outcome is not None:
            observations[eid] = observation_for(tallied, config, content[eid])

    def apply(current: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
        # Re-read under the lock: merge by row, so rows another process added
        # or changed while the model calls ran are neither lost nor touched.
        by_id = {example_id_of(row, i): row for i, row in enumerate(current)}
        changed: dict[str, str] = {}
        skipped: list[str] = []
        for eid, observation in observations.items():
            row = by_id.get(eid)
            if row is None or row["metadata"]["content_digest"] != content[eid]:
                skipped.append(eid)
                continue
            _, new = append_observation(
                row["metadata"]["regression"], observation, CONFIRMED_SCHEMA_VERSION
            )
            changed[eid] = new
        return changed, skipped

    wrote = bool(observations)
    changed_to: dict[str, str] = {}
    skipped: list[str] = []
    if wrote:
        try:
            changed_to, skipped = update_rows_locked(_regression_dir(directory), apply)
        except RegressionSetError as exc:
            raise _Stop(CONFIRM_BAD_FILE, "file_invalid", str(exc)) from exc

    after = {eid: changed_to.get(eid, before[eid]) for eid, _ in selected}
    cost = "unmeasured" if run.total_cost is None else f"${run.total_cost:.4f}"
    click.echo(
        f"Confirmed {len(selected)} row(s) under config {digest_of_config}: "
        f"{len(run.per_attempt)} of {repeats} repeat(s) ran, cost {cost}."
    )
    click.echo("status           rows")
    for status in STATUSES:
        click.echo(f"{status:<16} {sum(1 for s in after.values() if s == status)}")
    moved = [
        f"{eid}: {before[eid]} -> {after[eid]}"
        for eid, _ in selected
        if after[eid] != before[eid]
    ]
    click.echo("status changes: " + ("; ".join(moved) if moved else "none"))
    if skipped:
        click.echo(
            f"skipped (row removed or its content changed while running): {skipped}"
        )
    if run.unresolved:
        click.echo(
            "not recorded (results could not be tied to the row): "
            + ", ".join(f"{eid} ({why})" for eid, why in sorted(run.unresolved.items()))
        )
    if not run.completed:
        click.echo(
            f"Stopped early ({run.stop_reason or 'fewer repeats than requested'}); "
            + (
                "observations from the finished repeats were written."
                if wrote
                else "fewer than 2 repeats finished, so nothing was written."
            ),
            err=True,
        )
        raise SystemExit(CONFIRM_STOPPED_EARLY)


@regression.command("check")
@click.option(
    "--module",
    "module_spec",
    required=True,
    help="PATH_OR_MODULE:FUNCTION of the @traigent.optimize function to run",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    help="JSON file holding the configuration to run the rows under",
)
@click.option(
    "--dir",
    "directory",
    default=DEFAULT_REGRESSION_DIR,
    show_default=True,
    help="Directory holding regression.jsonl",
)
@click.option(
    "--repeats",
    type=click.IntRange(min=1),
    default=DEFAULT_CHECK_REPEATS,
    show_default=True,
    help=(
        "Times each row runs; a row passes only when more scored attempts pass "
        "than fail, regresses when more fail than pass, and a tie is inconclusive"
    ),
)
@click.option(
    "--fail-on-intermittent",
    is_flag=True,
    help="Also gate intermittent rows (default: reported as not gated)",
)
@click.option("--json", "as_json", is_flag=True, help="Print one JSON object")
@click.option(
    "--show-output",
    "show_output",
    is_flag=True,
    help=SHOW_OUTPUT_HELP,
)
def regression_check(
    module_spec: str,
    config_path: str,
    directory: str,
    repeats: int,
    fail_on_intermittent: bool,
    as_json: bool,
    show_output: bool,
) -> None:
    """Re-run the ARMED regression rows and fail on a regression (for CI).

    WRITES NOTHING: the regression file, its lock file and its directory are
    never modified, so CI never writes to your repository. Arm rows first with
    'traigent regression confirm'. Known failures, unconfirmed rows and (unless
    --fail-on-intermittent) intermittent rows are listed as not gated.

    Everything is local: no call to the Traigent backend (your function calls
    your own model provider). Output has counts, ids and intervals, never
    example content. Anything your code or the SDK prints while running is
    suppressed (a line count is printed on stderr); --show-output prints it
    on stderr, where it may contain example content.

    Trust: the regression file is edited by people and is NOT authenticated
    evidence. A hand edit that stays internally consistent can arm or disarm a
    row. Changes to the file should go through your repository's normal review.
    A row whose content changed since it was confirmed is listed as not gated.

    Exit codes: 0 every armed row passed (or none are armed); 1 an armed row
    regressed (more of its scored attempts failed than passed); 2 inconclusive
    (an armed row with no scored attempt or with as many passes as fails, which
    is listed with reason "tie"; function or config unusable);
    3 regression file missing, corrupt or edited by hand.

    Example:
        traigent regression check --module agent.py:answer --config c.json
    """
    try:
        _check(
            module_spec,
            config_path,
            directory,
            repeats,
            fail_on_intermittent,
            as_json,
            show_output,
        )
    except _Stop as stop:
        _exit_with(stop, as_json)
    except (click.ClickException, click.exceptions.Exit, click.Abort):
        raise
    except Exception as exc:  # noqa: BLE001 - keep --json one object on any path
        _exit_with(
            _Stop(
                CHECK_INCONCLUSIVE,
                "internal_error",
                f"The check failed unexpectedly ({type(exc).__name__})",
            ),
            as_json,
        )


def _check(
    module_spec: str,
    config_path: str,
    directory: str,
    repeats: int,
    fail_on_intermittent: bool,
    as_json: bool,
    show_output: bool,
) -> None:
    output = _UserOutput(show_output)
    # check writes nothing: keep Python from leaving __pycache__ next to the
    # user's modules while it imports them (put back whatever happens).
    keep_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        _check_run(
            module_spec,
            config_path,
            directory,
            repeats,
            fail_on_intermittent,
            as_json,
            output,
        )
    finally:
        sys.dont_write_bytecode = keep_bytecode
        output.report()


def _check_run(
    module_spec: str,
    config_path: str,
    directory: str,
    repeats: int,
    fail_on_intermittent: bool,
    as_json: bool,
    output: _UserOutput,
) -> None:
    _, rows = _read_rows(directory, CHECK_BAD_FILE)
    with output.window():
        function = _load_function(module_spec)
    config = _load_config(config_path)
    gated_statuses = {STATUS_ARMED}
    if fail_on_intermittent:
        gated_statuses.add(STATUS_INTERMITTENT)
    entries = [(example_id_of(row, i), row) for i, row in enumerate(rows)]
    gated = [(e, r) for e, r in entries if row_status(r) in gated_statuses]
    # A row that was gated when last measured but whose content has changed
    # since no longer gates: say so, instead of letting it fall silently into
    # "unconfirmed".
    changed = [
        e
        for e, r in entries
        if status_when_last_measured(r["metadata"]["regression"]) in gated_statuses
        and content_changed_since_measured(
            r["metadata"]["regression"], r["metadata"]["content_digest"]
        )
    ]
    not_gated = {
        status: [e for e, r in entries if row_status(r) == status and e not in changed]
        for status in (STATUS_KNOWN_FAILURE, STATUS_INTERMITTENT, STATUS_UNCONFIRMED)
        if status not in gated_statuses
    }
    not_gated[CONTENT_CHANGED] = changed

    regressions: list[tuple[str, Any]] = []
    inconclusive: list[tuple[str, Any]] = []
    passed: list[str] = []
    unresolved: dict[str, str] = {}
    run_complete = True
    if gated:
        try:
            with output.window():
                run = run_rows(function, config, gated, repeats=repeats)
        except IdsNotAssignableError as exc:
            raise _Stop(CHECK_INCONCLUSIVE, "ids_not_assignable", str(exc)) from exc
        except ValueError as exc:
            raise _Stop(
                CHECK_INCONCLUSIVE,
                "config_invalid",
                f"The run was refused ({type(exc).__name__}); is the config "
                "complete and inside the function's configuration space?",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - user code and SDK policy errors
            raise _Stop(
                CHECK_INCONCLUSIVE,
                "run_failed",
                f"The run failed ({type(exc).__name__})",
            ) from exc
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - e.g. the function calls sys.exit
            raise _Stop(
                CHECK_INCONCLUSIVE,
                "run_failed",
                f"The run was ended from inside the function ({type(exc).__name__})",
            ) from exc
        run_complete = run.completed
        for eid, _row in gated:
            if eid in run.unresolved:
                unresolved[eid] = run.unresolved[eid]
                inconclusive.append(
                    (eid, Tally(len(run.per_attempt), 0, 0, 0, None, None, None))
                )
                continue
            tallied = tally(run.per_attempt, eid)
            if tallied.scored == 0:
                inconclusive.append((eid, tallied))
            elif tallied.regressed:
                regressions.append((eid, tallied))
            elif tallied.tied:
                inconclusive.append((eid, tallied))
                unresolved[eid] = REASON_TIE
            elif tallied.passed_strictly:
                passed.append(eid)
            else:  # unreachable: scored attempts are a pass, a fail or a tie
                inconclusive.append((eid, tallied))
    if regressions:
        code, verdict = CHECK_REGRESSION, "regression"
    elif inconclusive or not run_complete or len(passed) != len(gated):
        code, verdict = CHECK_INCONCLUSIVE, "inconclusive"
    else:
        code, verdict = CHECK_OK, "pass"

    def shape(item: tuple[str, Any]) -> dict[str, Any]:
        eid, t = item
        reason = {"reason": unresolved[eid]} if eid in unresolved else {}
        return {
            **reason,
            "id": eid,
            "attempts": t.attempts,
            "passes": t.passes,
            "fails": t.fails,
            "errored": t.errored,
            "pass_rate_low": t.pass_rate_low,
            "pass_rate_high": t.pass_rate_high,
        }

    if as_json:
        click.echo(
            json.dumps(
                {
                    "schema": "traigent.regression_check/1",
                    "verdict": verdict,
                    "exit_code": code,
                    "repeats": repeats,
                    "armed": len(gated),
                    "passed": len(passed),
                    "run_complete": run_complete,
                    "regressions": [shape(i) for i in regressions],
                    "inconclusive": [shape(i) for i in inconclusive],
                    "not_gated": not_gated,
                },
                sort_keys=True,
            )
        )
    else:
        _print_check(
            gated,
            passed,
            regressions,
            inconclusive,
            not_gated,
            run_complete,
            fail_on_intermittent,
            unresolved,
        )
    raise SystemExit(code)


def _print_check(
    gated: list[Any],
    passed: list[str],
    regressions: list[tuple[str, Any]],
    inconclusive: list[tuple[str, Any]],
    not_gated: dict[str, list[str]],
    run_complete: bool,
    fail_on_intermittent: bool,
    unresolved: dict[str, str],
) -> None:
    label = "gated row(s)" if fail_on_intermittent else "armed row(s)"
    unarmed = sum(len(v) for v in not_gated.values())
    if not gated:
        click.echo(
            f"0 armed rows; {len(not_gated.get(STATUS_KNOWN_FAILURE, []))} known "
            f"failures not gated ({unarmed} unarmed row(s) in all)"
        )
    else:
        click.echo(
            f"{len(gated)} {label}: {len(passed)} passed, {len(regressions)} "
            f"regression(s), {len(inconclusive)} inconclusive"
        )
    for eid, t in regressions:
        click.echo(
            f"REGRESSION {eid}: attempts={t.attempts} passes={t.passes} "
            f"fails={t.fails} errored={t.errored} pass_rate_95={_interval(t)}"
        )
    for eid, t in inconclusive:
        if unresolved.get(eid) == REASON_TIE:
            click.echo(
                f"INCONCLUSIVE {eid}: tie: attempts={t.attempts} passes={t.passes} "
                f"fails={t.fails} errored={t.errored} (as many passes as fails)"
            )
            continue
        if eid in unresolved:
            click.echo(
                f"INCONCLUSIVE {eid}: results could not be tied to the row "
                f"({unresolved[eid]})"
            )
            continue
        click.echo(
            f"INCONCLUSIVE {eid}: attempts={t.attempts} passes={t.passes} "
            f"fails={t.fails} errored={t.errored} (no scored attempt)"
        )
    if not run_complete:
        click.echo("The run stopped before every repeat finished.")
    for status, ids in not_gated.items():
        if not ids:
            continue
        if status == CONTENT_CHANGED:
            click.echo(
                "not gated (content changed since it was confirmed): "
                f"{len(ids)} row(s): {', '.join(ids)}"
            )
        else:
            click.echo(f"not gated ({status}): {', '.join(ids)}")


def register_regression_commands(cli: click.Group) -> None:
    """Register the ``traigent regression`` group on the CLI group."""
    cli.add_command(regression)
