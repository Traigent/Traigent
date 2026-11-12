"""Traigent#3110: NaN values are never reported as the best configuration."""

from __future__ import annotations

import numpy as np
import pytest

from traigent.api.types import ConfigurationComparison, ParetoFront, SensitivityAnalysis

NAN = float("nan")


def _cmp(values: list[float]) -> ConfigurationComparison:
    return ConfigurationComparison(
        [{"a": i} for i in range(len(values))], {"accuracy": values}, {}, [], 0.95
    )


def test_best_configuration_skips_nan() -> None:
    assert _cmp([NAN, 0.5, 0.9]).get_best_configuration("accuracy")[0] == 2
    assert _cmp([0.1, NAN, 0.9]).get_best_configuration("accuracy")[0] == 2
    assert _cmp([NAN, 0.5, 0.9]).get_best_configuration("accuracy", "minimize")[0] == 1


def test_best_configuration_all_nan_raises() -> None:
    with pytest.raises(ValueError):
        _cmp([NAN, NAN]).get_best_configuration("accuracy")


def test_balanced_config_excludes_nan_rows() -> None:
    front = ParetoFront(
        [{"a": 1}, {"a": 2}, {"a": 3}],
        np.array([[NAN, 1.0], [0.5, 0.5], [0.4, 0.4]]),
        ["x", "y"],
        [True, True],
    )
    assert front.get_best_balanced_config() == {"a": 2}


def test_top_parameters_sorts_nan_last() -> None:
    sa = SensitivityAnalysis.__new__(SensitivityAnalysis)
    object.__setattr__(
        sa, "parameter_importance", {"a": NAN, "b": 2.0, "c": 3.0, "d": NAN}
    )
    names = [n for n, _ in sa.get_top_parameters(4)]
    assert names[:2] == ["c", "b"]
    assert set(names[2:]) == {"a", "d"}
