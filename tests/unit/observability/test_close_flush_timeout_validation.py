"""ObservabilityClient.close/flush timeout validation."""

from __future__ import annotations

import pytest

from traigent import ObservabilityClient


@pytest.mark.parametrize("method", ["close", "flush"])
def test_inf_timeout_treated_as_unbounded(method) -> None:
    c = ObservabilityClient()
    result = getattr(c, method)(timeout=float("inf"))
    assert result.success is True


@pytest.mark.parametrize("method", ["close", "flush"])
@pytest.mark.parametrize("bad", [float("nan"), -1, -0.5])
def test_nan_and_negative_timeout_rejected(method, bad) -> None:
    c = ObservabilityClient()
    with pytest.raises(ValueError, match="timeout"):
        getattr(c, method)(timeout=bad)


@pytest.mark.parametrize("method", ["close", "flush"])
def test_huge_timeout_clamped(method) -> None:
    c = ObservabilityClient()
    assert getattr(c, method)(timeout=10**400).success is True
    c2 = ObservabilityClient()
    assert getattr(c2, method)(timeout=1e300).success is True


@pytest.mark.parametrize("method", ["close", "flush"])
def test_non_numeric_timeout_rejected(method) -> None:
    c = ObservabilityClient()
    with pytest.raises(ValueError, match="timeout"):
        getattr(c, method)(timeout="5")


@pytest.mark.parametrize("method", ["close", "flush"])
def test_numpy_and_fraction_timeouts_accepted(method) -> None:
    """numpy and Fraction timeouts are real numbers; base accepted them."""
    from fractions import Fraction

    np = pytest.importorskip("numpy")
    for value in (np.int64(5), np.float32(2.5), Fraction(3, 2)):
        assert getattr(ObservabilityClient(), method)(timeout=value).success is True
    with pytest.raises(ValueError, match="timeout"):
        getattr(ObservabilityClient(), method)(timeout=np.float64("nan"))
