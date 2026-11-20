"""Per-example consistency labels across repeated evaluations of one config.

# Traceability: CONC-Layer-Data CONC-Quality-Observability FUNC-ANALYTICS

When the same configuration is evaluated more than once on the same examples
(winner-stability reruns, ``compare(..., repeats=N)``), each example gets a
pass/fail outcome per evaluation. This module turns those outcomes into one
label per example:

* ``consistent_pass``: every known evaluation passed
* ``consistent_fail``: every known evaluation failed
* ``flaky``: both outcomes occurred

An evaluation is *unknown* when its cell has an error or no finite score. An
unknown evaluation never counts as a pass, and an example with fewer than two
known evaluations gets no label at all.

The pass/fail rule is the one :meth:`OptimizationResult.failing_examples`
uses (``outcome_matrix._classify_cell`` at threshold 0.5), so the two views
never disagree about what a passing cell is. Only example ids and counts are
kept: no inputs, outputs or error text.
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

#: ``result.metadata`` key holding the stored counts. Local only; never uploaded.
EXAMPLE_CONSISTENCY_METADATA_KEY = "example_consistency"

#: ``result.metadata`` keys that stay on the user's machine. Every code path
#: that copies result metadata to a remote destination (tracking services,
#: uploaded artifacts) must drop these via :func:`without_local_only_metadata`.
LOCAL_ONLY_RESULT_METADATA_KEYS: frozenset[str] = frozenset(
    {EXAMPLE_CONSISTENCY_METADATA_KEY}
)

#: Fewest known evaluations an example needs to be labelled.
MIN_KNOWN_EVALUATIONS = 2

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"


def without_local_only_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow copy of ``metadata`` minus the local-only keys.

    The input is never mutated, so the result keeps its labels locally.
    """
    return {
        key: value
        for key, value in (metadata or {}).items()
        if key not in LOCAL_ONLY_RESULT_METADATA_KEYS
    }


def cell_outcome(cell: Any, threshold: float = DEFAULT_PASS_THRESHOLD) -> str:
    """Return ``pass``, ``fail`` or ``unknown`` for one outcome-matrix cell.

    Uses the failing-examples rule (``_classify_cell``), except that a cell
    carrying an error is ``unknown``: an evaluation that did not produce an
    answer says nothing about whether the answer is stable.
    """
    if isinstance(cell, Mapping) and cell.get("error") is not None:
        return UNKNOWN
    return _classify_cell(cell, threshold)


def classify_consistency(outcomes: Iterable[bool | None]) -> str | None:
    """Label one example from its per-evaluation outcomes.

    Args:
        outcomes: One entry per evaluation: ``True`` passed, ``False`` failed,
            ``None`` unknown (error or unscorable). Unknowns are ignored and
            never count as a pass.

    Returns:
        ``"consistent_pass"``, ``"consistent_fail"`` or ``"flaky"``; ``None``
        when fewer than two evaluations are known.
    """
    known = [o for o in outcomes if o is not None]
    if len(known) < MIN_KNOWN_EVALUATIONS:
        return None
    passes = sum(1 for o in known if o)
    if passes == len(known):
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
    per_evaluation: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Fold per-evaluation outcome maps into labelled per-example counts.

    Only examples that earn a label are returned. An example missing from an
    evaluation counts as unknown in that evaluation.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for outcomes in per_evaluation:
        for example_id in outcomes:
            if example_id not in seen:
                seen.add(example_id)
                ids.append(example_id)
    summary: dict[str, dict[str, Any]] = {}
    for example_id in ids:
        marks = [outcomes.get(example_id, UNKNOWN) for outcomes in per_evaluation]
        label = classify_consistency(
            [True if m == PASS else False if m == FAIL else None for m in marks]
        )
        if label is None:
            continue
        summary[example_id] = {
            "label": label,
            "passes": marks.count(PASS),
            "fails": marks.count(FAIL),
            "unknowns": marks.count(UNKNOWN),
        }
    return summary


def build_consistency_report(
    examples: Mapping[str, Mapping[str, Any]] | None,
    *,
    reps: int = 0,
    config_hash: str | None = None,
) -> dict[str, Any]:
    """Shape labelled examples as ``{"counts", "examples", "reps", "config_hash"}``."""
    counts = dict.fromkeys(CONSISTENCY_LABELS, 0)
    kept: dict[str, dict[str, Any]] = {}
    for example_id, entry in (examples or {}).items():
        label = entry.get("label")
        if label not in counts:
            continue
        counts[label] += 1
        kept[str(example_id)] = {
            "label": label,
            "passes": int(entry.get("passes", 0)),
            "fails": int(entry.get("fails", 0)),
            "unknowns": int(entry.get("unknowns", 0)),
        }
    return {
        "counts": counts,
        "examples": kept,
        "reps": int(reps) if kept else 0,
        "config_hash": config_hash if kept else None,
    }


def format_consistency_line(report: Mapping[str, Any], scope: str) -> str | None:
    """The one-line CLI summary, or ``None`` when no example has a label."""
    if not report.get("examples"):
        return None
    counts = report["counts"]
    parts = ", ".join(f"{counts[label]} {label}" for label in CONSISTENCY_LABELS)
    return f"{parts} ({scope}, {report['reps']} reps)"
