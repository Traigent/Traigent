from __future__ import annotations

from traigent.analytics.render import _as_float


def test_as_float_rejects_non_finite_numbers() -> None:
    assert _as_float(float("nan")) is None
    assert _as_float(float("inf")) is None
    assert _as_float("-inf") is None


def test_as_float_accepts_finite_numbers() -> None:
    assert _as_float("1.25") == 1.25
    assert _as_float(0) == 0.0
