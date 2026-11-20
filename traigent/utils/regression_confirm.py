"""Run regression rows under a fixed configuration and tally each row.

# Traceability: CONC-Layer-Data FUNC-STORAGE FUNC-ANALYTICS

The shared engine of ``traigent regression confirm`` and ``check``. It runs
``OptimizedFunction.compare([config], eval_dataset=<rows>, repeats=N)`` (local,
no backend session) and reads each repeat's per-example outcome through
``traigent.utils.example_consistency``: the pass/fail/errored rule, the
all_pass / all_fail / mixed label and the Wilson interval are that module's.
Tallies hold ids and counts only, never example content.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.example_consistency import (
    FAIL,
    MIN_ATTEMPTS,
    PASS,
    UNKNOWN,
    ExampleIdCollisionError,
    classify_consistency,
    trial_example_outcomes,
    wilson_interval,
)
from traigent.utils.hashing import generate_config_hash
from traigent.utils.regression_status import (
    OUTCOME_ERRORED,
    effective_status,
)

DEFAULT_CONFIRM_REPEATS = 5
DEFAULT_CHECK_REPEATS = 1


def today() -> str:
    """UTC date for an observation (a seam the tests replace)."""
    return datetime.now(UTC).date().isoformat()


def example_id_of(row: Mapping[str, Any], position: int) -> str:
    """The id a row has in a run: its stored id, else ``example_<position>``."""
    stored = row["metadata"]["example_id"]
    return stored if stored != "" else f"example_{position}"


def row_status(row: Mapping[str, Any]) -> str:
    """The row's status on its CURRENT content (older observations do not count)."""
    metadata = row["metadata"]
    return effective_status(metadata["regression"], metadata["content_digest"])


#: Closed reasons a row's outcomes could not be tied to it in a run.
UNRESOLVED_ID_MISSING = "id_missing"
UNRESOLVED_ID_DUPLICATED = "id_duplicated"
UNRESOLVED_ID_MISMATCH = "id_mismatch"
UNRESOLVED_REASONS: tuple[str, ...] = (
    UNRESOLVED_ID_MISSING,
    UNRESOLVED_ID_DUPLICATED,
    UNRESOLVED_ID_MISMATCH,
)


class IdsNotAssignableError(RuntimeError):
    """The evaluator behind the function chooses its own example ids.

    The ids this command assigns to the rows of a run never came back, so no
    result can be tied to a row. Carries no content.
    """

    def __init__(self) -> None:
        super().__init__(
            "The function's evaluator assigns its own example ids and ignores "
            "the ones this command sets (an external-service evaluator reads "
            "them from each input's example_id or input_id; a custom evaluator "
            "or a content-identity grant sets its own), so results cannot be "
            "matched to regression rows without guessing. Run the rows through "
            "a function that uses the built-in local evaluator. Nothing was "
            "written."
        )


@dataclass
class RunDataset:
    """The dataset a run evaluates plus the map from its ids back to rows.

    Every example carries an opaque, run-scoped id assigned here; the stored
    regression id and the ids an evaluator might derive from the customer's
    input play no part in matching a result to a row.
    """

    dataset: Dataset
    row_ids: Mapping[str, str]  # assigned id -> regression id


def build_run_dataset(rows: Sequence[tuple[str, Mapping[str, Any]]]) -> RunDataset:
    """A dataset of ``(regression id, row)`` pairs under fresh opaque ids."""
    token = secrets.token_hex(6)
    assigned = [f"rc_{token}_{position}" for position in range(len(rows))]
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data=row["input"],
                expected_output=row["output"],
                metadata={"example_id": run_id},
            )
            for run_id, (_, row) in zip(assigned, rows, strict=True)
        ],
        name="regression_rows",
        description="Regression rows selected for a run",
    )
    return RunDataset(
        dataset=dataset,
        row_ids={
            run_id: regression_id
            for run_id, (regression_id, _) in zip(assigned, rows, strict=True)
        },
    )


@dataclass
class RunResult:
    """Per-attempt outcomes of one configuration over a dataset.

    Attributes:
        per_attempt: One entry per repeat that ran: ``regression id -> pass |
            fail | unknown`` (only for rows whose result was tied to them), or
            ``None`` for a repeat that crashed.
        requested: Repeats asked for.
        stop_reason: Why the run ended before ``requested`` repeats, if it did.
        total_cost: Spend over the run; ``None`` when unmeasured.
        unresolved: ``regression id -> closed reason`` for rows whose result
            could not be tied to them in some repeat (missing, repeated or
            foreign ids). Such rows are inconclusive; their outcomes are not
            in ``per_attempt``.
    """

    per_attempt: Sequence[Mapping[str, str] | None] = field(default_factory=list)
    requested: int = 0
    stop_reason: str | None = None
    total_cost: float | None = None
    unresolved: Mapping[str, str] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.stop_reason is None and len(self.per_attempt) >= self.requested


def _row_value(row: Any, key: str) -> Any:
    return row.get(key) if isinstance(row, Mapping) else getattr(row, key, None)


def resolve_attempt(
    trial: Any, row_ids: Mapping[str, str]
) -> tuple[dict[str, str] | None, dict[str, str], bool, bool]:
    """Tie one repeat's results to rows by the ids this command assigned.

    Returns ``(outcomes, unresolved, foreign_seen, assigned_seen)``:
    ``outcomes`` maps regression ids to ``pass | fail | unknown`` for rows whose
    assigned id came back exactly once (``None`` when the trial has no
    per-example results); ``unresolved`` maps the other rows to a closed
    reason; ``foreign_seen`` says an id we did not assign came back;
    ``assigned_seen`` says one we did assign came back. The returned id set
    must equal the assigned one: any foreign id leaves every row of the repeat
    unresolved, because it shows an evaluator that rewrites ids.
    """
    metadata = getattr(trial, "metadata", None)
    results = metadata.get("example_results") if isinstance(metadata, Mapping) else None
    if not isinstance(results, list):
        return None, {}, False, False
    counts: dict[str, int] = {}
    for row in results:
        example_id = _row_value(row, "example_id")
        if example_id is not None:
            counts[str(example_id)] = counts.get(str(example_id), 0) + 1
    foreign = any(i not in row_ids for i in counts)
    assigned_seen = any(i in row_ids for i in counts)
    everything_mismatched = dict.fromkeys(row_ids.values(), UNRESOLVED_ID_MISMATCH)
    if foreign:
        return {}, everything_mismatched, True, assigned_seen
    try:
        outcomes = trial_example_outcomes(trial) or {}
    except ExampleIdCollisionError:
        return {}, everything_mismatched, True, assigned_seen
    resolved: dict[str, str] = {}
    unresolved: dict[str, str] = {}
    for run_id, regression_id in row_ids.items():
        seen = counts.get(run_id, 0)
        if seen == 1:
            resolved[regression_id] = outcomes.get(run_id, UNKNOWN)
        else:
            unresolved[regression_id] = (
                UNRESOLVED_ID_MISSING if seen == 0 else UNRESOLVED_ID_DUPLICATED
            )
    return resolved, unresolved, False, assigned_seen


def run_rows(
    function: Any,
    config: Mapping[str, Any],
    rows: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    repeats: int,
    max_cost: float | None = None,
) -> RunResult:
    """Evaluate ``config`` ``repeats`` times over ``rows`` via ``compare``.

    Results are tied to rows only through ids this command assigns for the run
    (see :func:`build_run_dataset`), never through an id an evaluator may
    derive from the customer's input. Verified against the code: the local
    evaluator uses ``metadata.example_id`` as the result id; the hybrid
    external-service evaluator uses ``input.example_id`` / ``input.input_id``
    whatever the dataset says (so those results come back under foreign ids);
    a custom evaluator and a content-identity grant set their own ids. Any
    path whose ids do not come back is refused (``IdsNotAssignableError``)
    rather than guessed; a row whose id is missing, repeated or accompanied by
    foreign ids in some repeat is left in ``RunResult.unresolved``.

    ``max_cost`` becomes a shared ``ExecutionBudget``, so the SDK stops before
    starting more repeats once the cap is reached and returns what finished.
    Policy errors raised by ``compare`` (an invalid config, strict-cost
    accounting) propagate.
    """
    budget = None
    if max_cost is not None:
        from traigent.core.execution_budget import ExecutionBudget

        budget = ExecutionBudget(max_cost_usd=max_cost)
    prepared = build_run_dataset(rows)
    comparison = function.compare(
        [dict(config)],
        eval_dataset=prepared.dataset,
        repeats=repeats,
        budget=budget,
    )
    compared = comparison.configs[0]
    per_attempt: list[Mapping[str, str] | None] = []
    unresolved: dict[str, str] = {}
    foreign_seen = assigned_seen = False
    for trial in compared.repeat_trials:
        if trial is None:
            per_attempt.append(None)
            continue
        outcomes, bad, foreign, matched = resolve_attempt(trial, prepared.row_ids)
        foreign_seen = foreign_seen or foreign
        assigned_seen = assigned_seen or matched
        for regression_id, reason in bad.items():
            unresolved.setdefault(regression_id, reason)
        per_attempt.append(outcomes)
    if foreign_seen and not assigned_seen:
        raise IdsNotAssignableError
    return RunResult(
        per_attempt=per_attempt,
        requested=repeats,
        stop_reason=comparison.stop_reason,
        total_cost=compared.total_cost,
        unresolved=unresolved,
    )


@dataclass(frozen=True)
class Tally:
    """One row's attempts under one configuration."""

    attempts: int
    passes: int
    fails: int
    errored: int
    pass_rate_low: float | None
    pass_rate_high: float | None
    outcome: str | None

    @property
    def scored(self) -> int:
        return self.passes + self.fails

    @property
    def majority_failed(self) -> bool:
        """More than half of the scored attempts failed."""
        return self.scored > 0 and self.fails * 2 > self.scored


def tally(per_attempt: Sequence[Mapping[str, str] | None], example_id: str) -> Tally:
    """Count one example's attempts; a missing or crashed attempt is errored."""
    marks = [
        UNKNOWN if outcomes is None else outcomes.get(example_id, UNKNOWN)
        for outcomes in per_attempt
    ]
    passes, fails = marks.count(PASS), marks.count(FAIL)
    low = high = None
    if passes + fails > 0:
        low, high = (round(v, 3) for v in wilson_interval(passes, passes + fails))
    label = classify_consistency(
        [True if m == PASS else False if m == FAIL else None for m in marks]
    )
    outcome: str | None
    if len(marks) < MIN_ATTEMPTS:
        outcome = None  # too few attempts to label: no observation
    else:
        outcome = label or OUTCOME_ERRORED
    return Tally(
        attempts=len(marks),
        passes=passes,
        fails=fails,
        errored=marks.count(UNKNOWN),
        pass_rate_low=low,
        pass_rate_high=high,
        outcome=outcome,
    )


def observation_for(
    tallied: Tally,
    config: Mapping[str, Any],
    content_digest: str,
    date: str | None = None,
) -> dict[str, Any]:
    """The observation a confirm run appends to a row.

    ``content_digest`` is the digest of the content the run measured; the
    status replay counts the observation only while the row still has it.
    """
    if tallied.outcome is None:
        raise ValueError("a tally with fewer than two attempts has no observation")
    return {
        "config_digest": generate_config_hash(dict(config)),
        "content_digest": content_digest,
        "date": date or today(),
        "reps": tallied.attempts,
        "passes": tallied.passes,
        "fails": tallied.fails,
        "errored": tallied.errored,
        "pass_rate_low": tallied.pass_rate_low,
        "pass_rate_high": tallied.pass_rate_high,
        "outcome": tallied.outcome,
    }
