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
