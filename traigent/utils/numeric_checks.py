"""Shared strict numeric checks for public constructors and validators."""

from __future__ import annotations

import math
import numbers
from typing import Any


def is_finite_number(value: Any) -> bool:
    """True for a real, finite, non-bool number (bool is an int subclass)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def require_finite_number(name: str, value: Any, *, integer: bool = False) -> None:
    """Reject bool, non-numbers, NaN/inf and (when ``integer``) fractions.

    Raises ``TypeError`` for a wrong type and ``ValueError`` for a bad value,
    both naming ``name`` but never echoing the value.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        kind = "an integer" if integer else "a number"
        raise TypeError(f"{name} must be {kind}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    if integer and not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
