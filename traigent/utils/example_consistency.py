"""Per-example consistency labels across repeated evaluations of one config.

# Traceability: CONC-Layer-Data CONC-Quality-Observability FUNC-ANALYTICS

When the same configuration is evaluated more than once on the same examples
(winner-stability reruns, ``compare(..., repeats=N)``), each example gets an
outcome per attempt. This module turns those outcomes into one label per
example:

* ``all_pass``: every scored attempt passed
* ``all_fail``: every scored attempt failed
* ``mixed``: both outcomes occurred

The labels describe what was observed and claim nothing beyond it: ``all_pass``
over five attempts does not mean the example always passes. For every example
the report keeps ``attempted``, ``passes``, ``fails`` and ``errored`` attempt
counts and a 95% Wilson score interval on the pass rate over the scored
attempts (``pass_rate_low``, ``pass_rate_high``, three decimals).

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

What counts as a pass: a cell passes when its per-example ``score`` (else
``accuracy``) is at least ``DEFAULT_PASS_THRESHOLD`` (0.5). That cut-off is only
meaningful for a higher-is-better score in ``[0, 1]``. The rule does not know a
metric's direction, so for a lower-is-better or differently scaled metric the
pass/fail reading is not meaningful. The report therefore records ``metric``
(which field decided: ``score``, ``accuracy`` or ``score_or_accuracy``) and
``threshold``, so a reader can see what "pass" meant.

The score rule is the one :meth:`OptimizationResult.failing_examples` uses
(``outcome_matrix._classify_cell`` at threshold 0.5), with one deliberate
difference: ``failing_examples`` counts an errored cell as failing, while this
view calls it unknown because whether it would pass is not known. Only example
ids and counts are kept: no inputs, outputs or error text.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from traigent.utils.outcome_matrix import _classify_cell, _extract_cell, _get

ALL_PASS = "all_pass"
ALL_FAIL = "all_fail"
MIXED = "mixed"
CONSISTENCY_LABELS: tuple[str, ...] = (ALL_PASS, ALL_FAIL, MIXED)

#: z for a two-sided 95% interval.
Z_95 = 1.959963984540054

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

#: ``metric`` values: the per-example field that decided pass or fail.
METRIC_SCORE = "score"
METRIC_ACCURACY = "accuracy"
METRIC_SCORE_OR_ACCURACY = "score_or_accuracy"
_METRICS = frozenset({METRIC_SCORE, METRIC_ACCURACY, METRIC_SCORE_OR_ACCURACY})


def wilson_interval(passes: int, scored: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``passes`` out of ``scored`` attempts.

    Closed form, standard library only. ``scored`` must be positive.
    """
    if scored <= 0 or not 0 <= passes <= scored:
        raise ValueError("wilson_interval needs 0 <= passes <= scored and scored > 0")
    p = passes / scored
    z2 = z * z
    denominator = 1 + z2 / scored
    center = (p + z2 / (2 * scored)) / denominator
    half = z * math.sqrt(p * (1 - p) / scored + z2 / (4 * scored * scored))
    half /= denominator
    return max(0.0, center - half), min(1.0, center + half)


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
        ``"all_pass"``, ``"all_fail"`` or ``"mixed"``; ``None``
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
        return ALL_PASS
    if passes == 0:
        return ALL_FAIL
    return MIXED


class TrialOutcomes(dict[str, str]):
    """Per-example outcomes of one trial, plus which fields decided them."""

    metrics_used: frozenset[str] = frozenset()


def _cell_metric(cell: Any) -> str | None:
    """The field ``_classify_cell`` read the score from, or ``None``."""
    if not isinstance(cell, Mapping):
        return None
    if cell.get("score") is not None:
        return METRIC_SCORE
    if cell.get("accuracy") is not None:
        return METRIC_ACCURACY
    return None


def scoring_metric(per_attempt: Iterable[Mapping[str, str] | None]) -> str | None:
    """Which per-example field decided the pass/fail outcomes, across attempts."""
    used: set[str] = set()
    for outcomes in per_attempt:
        used.update(getattr(outcomes, "metrics_used", ()))
    if not used:
        return None
    return next(iter(used)) if len(used) == 1 else METRIC_SCORE_OR_ACCURACY


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
    outcomes = TrialOutcomes()
    metrics: set[str] = set()
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
        cell = _extract_cell(row)
        outcomes[key] = cell_outcome(cell, threshold)
        metric = _cell_metric(cell)
        if outcomes[key] != UNKNOWN and metric is not None:
            metrics.add(metric)
    outcomes.metrics_used = frozenset(metrics)
    return outcomes


def _pass_rate_interval(passes: int, scored: int) -> tuple[float | None, float | None]:
    """95% Wilson interval rounded to three decimals; ``None`` with nothing scored."""
    if scored <= 0:
        return None, None
    low, high = wilson_interval(passes, scored)
    return round(low, 3), round(high, 3)


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
        passes, fails = marks.count(PASS), marks.count(FAIL)
        low, high = _pass_rate_interval(passes, passes + fails)
        summary[example_id] = {
            "label": classify_consistency(
                [True if m == PASS else False if m == FAIL else None for m in marks]
            ),
            "attempted": len(marks),
            "passes": passes,
            "fails": fails,
            "errored": marks.count(UNKNOWN),
            "pass_rate_low": low,
            "pass_rate_high": high,
        }
    return summary


def build_consistency_report(
    examples: Mapping[str, Mapping[str, Any]] | None,
    *,
    reps: int = 0,
    config_hash: str | None = None,
    metric: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Shape examples as the stored report dictionary.

    Keys: ``counts``, ``examples``, ``reps``, ``config_hash``, ``metric`` and
    ``threshold`` (what decided pass or fail; see the module docstring).

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
            "pass_rate_low": entry.get("pass_rate_low"),
            "pass_rate_high": entry.get("pass_rate_high"),
        }
    return {
        "counts": counts,
        "examples": kept,
        "reps": int(reps) if kept else 0,
        "config_hash": config_hash if kept else None,
        "metric": metric if kept else None,
        "threshold": threshold if kept else None,
    }


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_rate(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _is_known_label(label: Any) -> bool:
    # Type-check first: an unhashable stored value must not reach a membership test.
    return label is None or (isinstance(label, str) and label in CONSISTENCY_LABELS)


def _is_readable_entry(example_id: Any, entry: Any, reps: int) -> bool:
    if not (
        isinstance(example_id, str)
        and isinstance(entry, Mapping)
        and _is_known_label(entry.get("label"))
        and all(
            _is_count(entry.get(key))
            for key in ("attempted", "passes", "fails", "errored")
        )
        and _is_rate(entry.get("pass_rate_low"))
        and _is_rate(entry.get("pass_rate_high"))
    ):
        return False
    passes, fails, errored = entry["passes"], entry["fails"], entry["errored"]
    attempted = entry["attempted"]
    if attempted != passes + fails + errored or attempted > reps:
        return False
    # The label must be the one the counts produce.
    expected = classify_consistency(
        [True] * passes + [False] * fails + [None] * errored
    )
    if entry["label"] != expected:
        return False
    low, high = entry.get("pass_rate_low"), entry.get("pass_rate_high")
    return low is None or high is None or low <= high


def normalize_stored_report(stored: Any) -> dict[str, Any] | None:
    """Validate a stored report and return it in canonical shape.

    Returns ``None`` when ``stored`` is not a readable report: not a mapping,
    entries that are not mappings, labels outside the closed set, counts that
    are not non-negative integers (a ``bool`` is not an integer here), rates
    outside ``[0, 1]``, an interval whose low bound exceeds its high bound, an
    entry whose counts do not add up to ``attempted``, whose ``attempted``
    exceeds ``reps``, or whose label contradicts its counts, and ``reps`` below
    one when there are entries. A damaged report is refused whole, never partly
    read. Never raises.
    """
    try:
        return _normalize_stored_report(stored)
    except Exception:  # noqa: BLE001 - any odd stored value is an unreadable report
        return None


def _normalize_stored_report(stored: Any) -> dict[str, Any] | None:
    if not isinstance(stored, Mapping):
        return None
    examples = stored.get("examples")
    reps = stored.get("reps")
    config_hash = stored.get("config_hash")
    metric = stored.get("metric")
    threshold = stored.get("threshold")
    if (
        not isinstance(examples, Mapping)
        or not isinstance(reps, int)
        or not _is_count(reps)
        or not (config_hash is None or isinstance(config_hash, str))
        or not (metric is None or (isinstance(metric, str) and metric in _METRICS))
        or not (threshold is None or _is_finite_number(threshold))
        or (bool(examples) and reps < 1)
        or not all(
            _is_readable_entry(key, entry, reps) for key, entry in examples.items()
        )
    ):
        return None
    return build_consistency_report(
        examples,
        reps=reps,
        config_hash=config_hash,
        metric=metric,
        threshold=None if threshold is None else float(threshold),
    )


def format_consistency_line(report: Mapping[str, Any], scope: str) -> str | None:
    """The one-line CLI summary, or ``None`` when nothing was reported.

    The trailing clause shows what the best possible result means: an example
    that passed every one of ``reps`` attempts has a 95% Wilson interval of
    ``low-high`` on its pass rate.
    """
    try:
        return _format_consistency_line(report, scope)
    except Exception:  # noqa: BLE001 - an optional summary line never raises
        return None


def _format_consistency_line(report: Mapping[str, Any], scope: str) -> str | None:
    if not report.get("examples"):
        return None
    counts = report["counts"]
    reps = int(report["reps"])
    if reps < 1:
        return None
    parts = ", ".join(f"{counts[key]} {key}" for key in (*CONSISTENCY_LABELS, ERRORED))
    low, high = wilson_interval(reps, reps)
    return (
        f"Repeat outcomes ({scope}, {reps} reps): {parts}; "
        f"{ALL_PASS} at {reps}/{reps} means a 95% interval of "
        f"{low:.2f}-{high:.2f} on the pass rate"
    )
