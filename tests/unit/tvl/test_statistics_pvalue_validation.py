"""Multiple-testing adjusters reject invalid p-values loudly."""

from __future__ import annotations

import pytest

from traigent.tvl.statistics import (
    benjamini_hochberg_adjust,
    bonferroni_adjust,
    holm_bonferroni_adjust,
)

ADJ = [benjamini_hochberg_adjust, bonferroni_adjust, holm_bonferroni_adjust]


@pytest.mark.parametrize("fn", ADJ)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5])
def test_invalid_p_value_raises(fn, bad) -> None:
    with pytest.raises(ValueError, match="p-value"):
        fn([bad, 0.01])
    with pytest.raises(ValueError, match="p-value"):
        fn([bad])


@pytest.mark.parametrize("fn", ADJ)
def test_valid_inputs_unchanged(fn) -> None:
    assert fn([]) == []
    assert fn([0.0, 1.0, 0.5]) is not None


def test_bh_known_values() -> None:
    out = benjamini_hochberg_adjust([0.01, 0.04, 0.03, 0.005, 0.5, 0.5])
    assert out == pytest.approx([0.03, 0.06, 0.06, 0.03, 0.5, 0.5])
