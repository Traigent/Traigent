"""Per-example consistency labels across repeated evaluations of one config.

# Traceability: CONC-Layer-Data CONC-Quality-Observability FUNC-ANALYTICS

When the same configuration is evaluated more than once on the same examples
(winner-stability reruns, ``compare(..., repeats=N)``), each example gets an
outcome per attempt. This module turns those outcomes into one label per
example:

* ``consistent_pass``: every scored attempt passed
* ``consistent_fail``: every scored attempt failed
* ``flaky``: both outcomes occurred

For every example the report keeps ``attempted``, ``passes``, ``fails`` and
``errored`` attempt counts.

An attempt is *errored* (unknown) when it produced no judgment: its cell carries
an error (``error`` is not ``None``, an empty string included), the evaluation
reported ``success=False``, or there is no finite score. Both an ``error=""``
cell and a ``success=False`` cell with ``error=None`` are therefore errored; a
cell that ran and scored below the threshold is a fail. A rerun that crashed,
or returned no per-example results, is one errored attempt for every example of
that rerun. An errored attempt never counts as a pass or a fail.

An example whose errored attempts are at least half of its attempts gets no
label: it is counted under ``errored`` instead, so a mostly-errored example is
never reported as a clean pass. An example attempted fewer than twice gets
nothing.

The score rule is the one :meth:`OptimizationResult.failing_examples` uses
(``outcome_matrix._classify_cell`` at threshold 0.5), with one deliberate
difference: ``failing_examples`` counts an errored cell as failing, while this
view calls it unknown because whether it would pass is not known. Only example
ids and counts are kept: no inputs, outputs or error text.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from traigent.utils.outcome_matrix import _classify_cell, _extract_cell, _get

CONSISTENT_PASS = "consistent_pass"
CONSISTENT_FAIL = "consistent_fail"
FLAKY = "flaky"
CONSISTENCY_LABELS: tuple[str, ...] = (CONSISTENT_PASS, CONSISTENT_FAIL, FLAKY)

#: Same default threshold as ``OptimizationResult.failing_examples``.
DEFAULT_PASS_THRESHOLD = 0.5

#: Fewest attempts an example needs before it is reported at all.
MIN_ATTEMPTS = 2

#: ``counts`` key for examples with no label because errored attempts made up
#: at least half of their attempts.
ERRORED = "errored"

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"


def cell_outcome(cell: Any, threshold: float = DEFAULT_PASS_THRESHOLD) -> str:
    """Return ``pass``, ``fail`` or ``unknown`` for one outcome-matrix cell.

    Uses the failing-examples score rule (``_classify_cell``), except that an
    attempt which produced no judgment is ``unknown``: a cell whose ``error`` is
    not ``None`` (an empty string counts) or whose ``success`` is ``False``.
    Such an attempt says nothing about whether the answer is stable.
    """
    if isinstance(cell, Mapping) and (
        cell.get("error") is not None or cell.get("success") is False
    ):
        return UNKNOWN
    return _classify_cell(cell, threshold)


def classify_consistency(outcomes: Iterable[bool | None]) -> str | None:
    """Label one example from its per-attempt outcomes.

    Args:
        outcomes: One entry per attempt: ``True`` passed, ``False`` failed,
            ``None`` errored (unknown). Errored attempts never count as a pass
            or a fail.

    Returns:
        ``"consistent_pass"``, ``"consistent_fail"`` or ``"flaky"``; ``None``
        when there are fewer than two attempts or errored attempts are at least
        half of them (the example is then reported as errored, not labelled).
    """
    attempts = list(outcomes)
    errored = sum(1 for o in attempts if o is None)
    if len(attempts) < MIN_ATTEMPTS or errored * 2 >= len(attempts):
        return None
    passes = sum(1 for o in attempts if o)
    scored = len(attempts) - errored
    if passes == scored:
        return CONSISTENT_PASS
    if passes == 0:
        return CONSISTENT_FAIL
    return FLAKY


class ExampleIdCollisionError(ValueError):
    """Two distinct example ids have the same text form (e.g. ``1`` and ``"1"``).

    Stored reports key examples by text, so merging them would report one
    example's outcomes under the other's id. Nothing is merged: the caller
    gets this error naming both ids.
    """


def _same_example_id(first: Any, second: Any) -> bool:
    return type(first) is type(second) and first == second


def trial_example_outcomes(
    trial: Any, threshold: float = DEFAULT_PASS_THRESHOLD
) -> dict[str, str] | None:
    """Per-example ``pass``/``fail``/``unknown`` for one trial.

    Returns ``None`` when the trial carries no per-example results, so the
    caller can tell "no evidence" from "every example unknown".

    Raises:
        ExampleIdCollisionError: two distinct ids (such as ``1`` and ``"1"``)
            share a text form. Nothing is merged.
    """
    metadata = getattr(trial, "metadata", None)
    rows = metadata.get("example_results") if isinstance(metadata, Mapping) else None
    if not isinstance(rows, list):
        return None
    outcomes: dict[str, str] = {}
    originals: dict[str, Any] = {}
    for row in rows:
        example_id = _get(row, "example_id")
        if example_id is None:
            continue
        key = str(example_id)
        if key in originals and not _same_example_id(originals[key], example_id):
            raise ExampleIdCollisionError(
                f"Example ids {originals[key]!r} ({type(originals[key]).__name__}) "
                f"and {example_id!r} ({type(example_id).__name__}) have the same "
                f"text form {key!r}, so their per-example outcomes cannot be "
                "kept apart. Give every example an id that is distinct as text."
            )
        originals[key] = example_id
        outcomes[key] = cell_outcome(_extract_cell(row), threshold)
    return outcomes


def summarize_example_outcomes(
    per_attempt: Sequence[Mapping[str, str] | None],
) -> dict[str, dict[str, Any]]:
    """Fold per-attempt outcome maps into per-example counts.

    ``None`` stands for an attempt that crashed or returned no per-example
    results: it is one errored attempt for every example. An example missing
    from an attempt's map is errored in that attempt too. Nothing is returned
    for fewer than two attempts. Examples whose errored attempts are at least
    half of their attempts are kept with ``label=None``.
    """
    if len(per_attempt) < MIN_ATTEMPTS:
        return {}
    ids: list[str] = []
    seen: set[str] = set()
    for outcomes in per_attempt:
        for example_id in outcomes or ():
            if example_id not in seen:
                seen.add(example_id)
                ids.append(example_id)
    summary: dict[str, dict[str, Any]] = {}
    for example_id in ids:
        marks = [
            UNKNOWN if outcomes is None else outcomes.get(example_id, UNKNOWN)
            for outcomes in per_attempt
        ]
        summary[example_id] = {
            "label": classify_consistency(
                [True if m == PASS else False if m == FAIL else None for m in marks]
            ),
            "attempted": len(marks),
            "passes": marks.count(PASS),
            "fails": marks.count(FAIL),
            "errored": marks.count(UNKNOWN),
        }
    return summary


def build_consistency_report(
    examples: Mapping[str, Mapping[str, Any]] | None,
    *,
    reps: int = 0,
    config_hash: str | None = None,
) -> dict[str, Any]:
    """Shape examples as ``{"counts", "examples", "reps", "config_hash"}``.

    ``counts`` has one entry per label plus ``errored`` (examples with no label
    because errored attempts were at least half of their attempts).
    """
    counts = dict.fromkeys((*CONSISTENCY_LABELS, ERRORED), 0)
    kept: dict[str, dict[str, Any]] = {}
    for example_id, entry in (examples or {}).items():
        label = entry.get("label")
        if label in CONSISTENCY_LABELS:
            counts[label] += 1
        elif label is None:
            counts[ERRORED] += 1
        else:
            continue
        kept[str(example_id)] = {
            "label": label,
            "attempted": int(entry.get("attempted", 0)),
            "passes": int(entry.get("passes", 0)),
            "fails": int(entry.get("fails", 0)),
            "errored": int(entry.get("errored", 0)),
        }
    return {
        "counts": counts,
        "examples": kept,
        "reps": int(reps) if kept else 0,
        "config_hash": config_hash if kept else None,
    }


def format_consistency_line(report: Mapping[str, Any], scope: str) -> str | None:
    """The one-line CLI summary, or ``None`` when nothing was reported."""
    if not report.get("examples"):
        return None
    counts = report["counts"]
    parts = ", ".join(f"{counts[label]} {label}" for label in CONSISTENCY_LABELS)
    if counts.get(ERRORED):
        parts += f", {counts[ERRORED]} {ERRORED}"
    return f"{parts} ({scope}, {report['reps']} reps)"
