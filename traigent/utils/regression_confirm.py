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
    return effective_status(row["metadata"]["regression"])


def build_dataset(rows: Sequence[tuple[str, Mapping[str, Any]]]) -> Dataset:
    """A dataset of ``(id, row)`` pairs; the id is the only metadata."""
    return Dataset(
        examples=[
            EvaluationExample(
                input_data=row["input"],
                expected_output=row["output"],
                metadata={"example_id": example_id},
            )
            for example_id, row in rows
        ],
        name="regression_rows",
        description="Regression rows selected for a run",
    )


@dataclass
class RunResult:
    """Per-attempt outcomes of one configuration over a dataset.

    Attributes:
        per_attempt: One entry per repeat that ran: ``example_id -> pass | fail
            | unknown``, or ``None`` for a repeat that crashed.
        requested: Repeats asked for.
        stop_reason: Why the run ended before ``requested`` repeats, if it did.
        total_cost: Spend over the run; ``None`` when unmeasured.
    """

    per_attempt: Sequence[Mapping[str, str] | None] = field(default_factory=list)
    requested: int = 0
    stop_reason: str | None = None
    total_cost: float | None = None

    @property
    def completed(self) -> bool:
        return self.stop_reason is None and len(self.per_attempt) >= self.requested


def run_config(
    function: Any,
    config: Mapping[str, Any],
    dataset: Dataset,
    *,
    repeats: int,
    max_cost: float | None = None,
) -> RunResult:
    """Evaluate ``config`` ``repeats`` times over ``dataset`` via ``compare``.

    ``max_cost`` becomes a shared ``ExecutionBudget``, so the SDK stops before
    starting more repeats once the cap is reached and returns what finished.
    Policy errors raised by ``compare`` (an invalid config, strict-cost
    accounting) propagate.
    """
    budget = None
    if max_cost is not None:
        from traigent.core.execution_budget import ExecutionBudget

        budget = ExecutionBudget(max_cost_usd=max_cost)
    comparison = function.compare(
        [dict(config)], eval_dataset=dataset, repeats=repeats, budget=budget
    )
    compared = comparison.configs[0]
    per_attempt = [
        None if trial is None else trial_example_outcomes(trial)
        for trial in compared.repeat_trials
    ]
    return RunResult(
        per_attempt=per_attempt,
        requested=repeats,
        stop_reason=comparison.stop_reason,
        total_cost=compared.total_cost,
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
    tallied: Tally, config: Mapping[str, Any], date: str | None = None
) -> dict[str, Any]:
    """The observation a confirm run appends to a row."""
    if tallied.outcome is None:
        raise ValueError("a tally with fewer than two attempts has no observation")
    return {
        "config_digest": generate_config_hash(dict(config)),
        "date": date or today(),
        "reps": tallied.attempts,
        "passes": tallied.passes,
        "fails": tallied.fails,
        "errored": tallied.errored,
        "pass_rate_low": tallied.pass_rate_low,
        "pass_rate_high": tallied.pass_rate_high,
        "outcome": tallied.outcome,
    }
