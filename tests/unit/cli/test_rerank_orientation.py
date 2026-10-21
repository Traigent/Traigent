"""results rerank honours objective direction (#2655)."""

from __future__ import annotations

from types import SimpleNamespace as N
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from traigent.cli.main import _calculate_trial_score, cli


def test_minimize_metric_lowers_score():
    w = {"accuracy": 0.8, "cost": 0.2}
    cheap = _calculate_trial_score(N(metrics={"accuracy": 0.9, "cost": 0.01}), w)
    pricey = _calculate_trial_score(N(metrics={"accuracy": 0.9, "cost": 0.5}), w)
    assert cheap > pricey


def test_explicit_orientations_override_defaults():
    w = {"quality": 1.0, "lag": 1.0}
    orient = {"quality": "maximize", "lag": "minimize"}
    fast = _calculate_trial_score(N(metrics={"quality": 0.5, "lag": 1.0}), w, orient)
    slow = _calculate_trial_score(N(metrics={"quality": 0.5, "lag": 9.0}), w, orient)
    assert fast > slow


def test_unknown_custom_metric_without_orientation_fails_closed():
    with pytest.raises(ValueError):
        _calculate_trial_score(N(metrics={"mystery": 1.0}), {"mystery": 1.0})


def test_rerank_picks_cheaper_of_two_equally_accurate_trials():
    cheap = N(
        trial_id="a", config={"m": "cheap"}, metrics={"accuracy": 0.9, "cost": 0.01}
    )
    pricey = N(
        trial_id="b", config={"m": "pricey"}, metrics={"accuracy": 0.9, "cost": 0.5}
    )
    result = N(
        trials=[pricey, cheap],
        best_config={"m": "pricey"},
        objectives=["accuracy"],
        metadata={},
    )
    with patch("traigent.cli.main.PersistenceManager") as pm:
        pm.return_value.load_result.return_value = result
        out = CliRunner().invoke(
            cli, ["results", "rerank", "run", "-w", "accuracy=0.8,cost=0.2"]
        )
    assert out.exit_code == 0, out.output
    new_best = out.output.split("New Best Configuration")[1]
    assert "cheap" in new_best.split("Note:")[0]


def test_rerank_unknown_metric_orientation_exits_nonzero():
    t = N(trial_id="a", config={}, metrics={"mystery": 1.0})
    result = N(trials=[t], best_config={}, objectives=[], metadata={})
    with patch("traigent.cli.main.PersistenceManager") as pm:
        pm.return_value.load_result.return_value = result
        out = CliRunner().invoke(cli, ["results", "rerank", "run", "-w", "mystery=1"])
    assert out.exit_code != 0
    assert "orientation" in out.output.lower()
