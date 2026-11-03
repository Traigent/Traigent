"""Relative improvement of a best score over a baseline, honouring direction."""

from __future__ import annotations

import math
from typing import Any

_REASON_BASELINE_MISSING = "baseline missing"
_REASON_BASELINE_ZERO = "baseline 0"
_REASON_BASELINE_NON_FINITE = "baseline not finite"
_REASON_BEST_MISSING = "best score missing"
_REASON_BEST_NON_FINITE = "best score not finite"
_REASON_BAND = "target-banded objective has no direction"
_REASON_ORIENTATION = "objective direction unknown"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def relative_change_with_reason(
    baseline: Any, best: Any, orientation: str | None
) -> tuple[float | None, str | None]:
    """Return ``(change, reason)``; exactly one of the two is ``None``.

    ``change`` is ``(best - baseline) / abs(baseline)`` for a maximized
    objective and the negation for a minimized one, so a positive value always
    means "better than baseline". It is undefined (``None`` plus a reason) when
    the baseline is missing, zero or non-finite, when the best score is missing
    or non-finite, or when the objective is banded or has no known direction.
    A best score of ``0.0`` is a real value, never "missing".
    """
    if orientation == "band":
        return None, _REASON_BAND
    if orientation not in ("maximize", "minimize"):
        return None, _REASON_ORIENTATION
    base = _number(baseline)
    if base is None:
        return None, _REASON_BASELINE_MISSING
    if not math.isfinite(base):
        return None, _REASON_BASELINE_NON_FINITE
    if base == 0:
        return None, _REASON_BASELINE_ZERO
    top = _number(best)
    if top is None:
        return None, _REASON_BEST_MISSING
    if not math.isfinite(top):
        return None, _REASON_BEST_NON_FINITE
    change = (top - base) / abs(base)
    return (-change if orientation == "minimize" else change), None


def relative_change(baseline: Any, best: Any, orientation: str | None) -> float | None:
    """Relative improvement of ``best`` over ``baseline``, or ``None`` if undefined."""
    return relative_change_with_reason(baseline, best, orientation)[0]


# A baseline of exactly zero has no relative scale. Comparisons against it use
# this floor as the denominator (the same scale as a 0.01 baseline), so a move
# from 0 to a materially non-zero value is reported with a finite, large change
# instead of being skipped (#2847).
ZERO_BASELINE_DENOMINATOR = 0.01


def signed_relative_delta(baseline: float, current: float) -> float:
    """Signed change of ``current`` from ``baseline`` as a fraction of the baseline.

    The denominator is ``abs(baseline)``, so the sign always says whether the
    value went up (positive) or down (negative), also for negative baselines.
    A zero baseline uses :data:`ZERO_BASELINE_DENOMINATOR`; ``0 -> 0`` is 0.0.
    """
    delta = current - baseline
    if delta == 0:
        return 0.0
    denominator = abs(baseline) if baseline != 0 else ZERO_BASELINE_DENOMINATOR
    return delta / denominator
