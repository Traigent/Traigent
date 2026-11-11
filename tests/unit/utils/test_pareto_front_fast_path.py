"""#3273: the two-objective Pareto front is O(n log n) and matches the scan."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from traigent.utils.multi_objective import ParetoFrontCalculator, ParetoPoint


def _trial(metrics):
    return SimpleNamespace(status="completed", metrics=metrics, config={})


def _reference(calc, trials, objectives):
    """Pre-#3273 quadratic scan, kept verbatim as the behavioural oracle."""
    points = []
    for trial in trials:
        point = calc._trial_to_pareto_point(trial, objectives)
        if point is not None:
            points.append(point)
    return [
        p
        for i, p in enumerate(points)
        if not any(
            i != j and q.dominates(p, calc.maximize, objectives=objectives)
            for j, q in enumerate(points)
        )
    ]


_BASE = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 2, 3, True, False]
_JITTER = [0.0, 0.0, 1e-12, -1e-12, 5e-11, 1e-10, 2e-10, -1e-10, 1e-9]


def _value(rng, special):
    v = rng.choice(_BASE) + rng.choice(_JITTER)
    if special and rng.random() < 0.06:
        return rng.choice([float("nan"), float("inf"), float("-inf")])
    return v


@pytest.mark.parametrize("special", [False, True])
def test_two_objective_front_equals_quadratic_scan(special):
    rng = random.Random(3273)
    for case in range(400):
        maximize = {"a": rng.random() < 0.5, "b": rng.random() < 0.5}
        calc = ParetoFrontCalculator(maximize=maximize)
        n = rng.randint(1, 40)
        trials = [
            _trial({"a": _value(rng, special), "b": _value(rng, special)})
            for _ in range(n)
        ]
        names = rng.choice([["a", "b"], ["b", "a"]])
        got = calc.calculate_pareto_front(trials, names)
        want = _reference(calc, trials, names)
        assert [p.trial for p in got] == [p.trial for p in want], case


def test_duplicates_all_stay_on_the_front():
    calc = ParetoFrontCalculator(maximize={"a": True, "b": False})
    trials = [_trial({"a": 1.0, "b": 1.0}) for _ in range(3)]
    assert len(calc.calculate_pareto_front(trials, ["a", "b"])) == 3


def test_two_objective_front_makes_no_pairwise_comparisons(monkeypatch):
    calls = 0
    original = ParetoPoint.dominates

    def counting(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ParetoPoint, "dominates", counting)
    rng = random.Random(1)
    calc = ParetoFrontCalculator(maximize={"a": True, "b": False})
    trials = [_trial({"a": rng.random(), "b": rng.random()}) for _ in range(2000)]
    front = calc.calculate_pareto_front(trials, ["a", "b"])
    assert calls == 0
    assert 0 < len(front) < 2000


def test_three_objectives_keep_the_exact_scan():
    rng = random.Random(7)
    calc = ParetoFrontCalculator(maximize={"a": True, "b": False, "c": True})
    trials = [_trial({k: round(rng.random(), 1) for k in "abc"}) for _ in range(60)]
    got = calc.calculate_pareto_front(trials, ["a", "b", "c"])
    want = _reference(calc, trials, ["a", "b", "c"])
    assert [p.trial for p in got] == [p.trial for p in want]
    assert not any(math.isnan(v) for p in got for v in p.objectives.values())
