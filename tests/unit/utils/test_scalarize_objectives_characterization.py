"""Characterization of ``scalarize_objectives``.

``oracle_scalarize_objectives`` is a frozen copy of the pre-refactor
implementation. Results (including float bit patterns), raised exceptions, the
sequence of reads on the inputs, and the number of orientation lookups are all
compared between the oracle and the live function.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.core.objectives import ObjectiveSchema
from traigent.utils import multi_objective as mo
from traigent.utils.multi_objective import scalarize_objectives


# --- frozen pre-refactor implementation ---
def oracle_scalarize_objectives(
    objectives: dict[str, float],
    weights: dict[str, float],
    minimize_objectives: list[str] | None = None,
    objective_schema: ObjectiveSchema | None = None,
) -> float:
    """Scalarize multiple objectives using weighted sum.

    Args:
        objectives: Dictionary of objective values
        weights: Dictionary of objective weights
        minimize_objectives: Explicit list of objectives to minimize. An empty
            list deliberately declares all objectives as maximize. When None,
            exact SDK-owned defaults are resolved and unknown names raise.
        objective_schema: ObjectiveSchema with orientations and weights (overrides other params).

    Returns:
        Scalarized objective value
    """
    if not objectives:
        return 0.0

    # Use ObjectiveSchema if provided (overrides other params)
    if objective_schema is not None:
        declared_values: dict[str, float] = {}
        weights = {}
        minimize_objectives = []

        for obj_def in objective_schema.objectives:
            weights[obj_def.name] = obj_def.weight
            if obj_def.name not in objectives:
                continue
            value = objectives[obj_def.name]
            if obj_def.orientation == "band":
                band = obj_def.band
                if band is None or band.low is None or band.high is None:
                    raise ValueError(
                        f"Banded objective {obj_def.name!r} requires a complete target band"
                    )
                if band.low <= value <= band.high:
                    declared_values[obj_def.name] = 0.0
                else:
                    declared_values[obj_def.name] = -float(
                        min(abs(value - band.low), abs(value - band.high))
                    )
                continue
            declared_values[obj_def.name] = value
            if obj_def.orientation == "minimize":
                minimize_objectives.append(obj_def.name)
        objectives = declared_values

    # No explicit list: resolve exact SDK-owned defaults. An explicitly empty
    # list remains the deliberate all-maximize declaration.
    elif minimize_objectives is None:
        minimize_objectives = [
            name
            for name in objectives
            if mo.resolve_objective_orientation(name) == "minimize"
        ]

    total_score = 0.0
    total_weight = 0.0

    for obj_name, obj_value in objectives.items():
        # Skip None values
        if obj_value is None:
            continue

        weight = weights.get(obj_name, 1.0)

        # For minimization objectives, we invert the value by using (1 - obj_value)
        # assuming objectives are normalized between 0 and 1, or use negative weight
        if obj_name in minimize_objectives:
            # Use negative objective value for minimization
            total_score += weight * (-obj_value)
        else:
            # Normal maximization
            total_score += weight * obj_value

        total_weight += weight

    # Handle case where total weight is zero (use equal weights)
    if total_weight == 0:
        # Use equal weights for all non-None objectives
        valid_objectives = {k: v for k, v in objectives.items() if v is not None}
        if not valid_objectives:
            return 0.0

        # Apply minimization logic for equal weights too
        total_value = 0.0
        for obj_name, obj_value in valid_objectives.items():
            if obj_name in minimize_objectives:
                total_value += -obj_value
            else:
                total_value += obj_value

        return total_value / len(valid_objectives)

    return total_score / total_weight


class _LoggedDict(dict):
    def __init__(self, data, log, tag):
        super().__init__(data)
        self._log = log
        self._tag = tag

    def get(self, key, *default):
        self._log.append((self._tag, "get", key))
        return super().get(key, *default)

    def __getitem__(self, key):
        self._log.append((self._tag, "item", key))
        return super().__getitem__(key)

    def __contains__(self, key):
        self._log.append((self._tag, "in", key))
        return super().__contains__(key)

    def __iter__(self):
        self._log.append((self._tag, "iter"))
        return super().__iter__()

    def items(self):
        self._log.append((self._tag, "items"))
        return super().items()


class _LoggedList(list):
    def __init__(self, data, log):
        super().__init__(data)
        self._log = log

    def __contains__(self, key):
        self._log.append(("minimize", "in", key))
        return super().__contains__(key)

    def append(self, value):
        self._log.append(("minimize", "append", value))
        super().append(value)


class _Logged:
    """Attribute reads are logged; used for objective definitions and bands."""

    def __init__(self, tag, log, **attrs):
        object.__setattr__(self, "_tag", tag)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_attrs", attrs)

    def __getattr__(self, name):
        attrs = object.__getattribute__(self, "_attrs")
        if name not in attrs:
            raise AttributeError(name)
        object.__getattribute__(self, "_log").append(
            (object.__getattribute__(self, "_tag"), "attr", name)
        )
        return attrs[name]


def _schema(log, defs):
    objects = []
    for idx, spec in enumerate(defs):
        attrs = dict(spec)
        band = attrs.get("band")
        if isinstance(band, dict):
            attrs["band"] = _Logged(f"band{idx}", log, **band)
        objects.append(_Logged(f"def{idx}", log, **attrs))
    return SimpleNamespace(objectives=objects)


def _run(fn, objectives, weights, minimize, defs, monkeypatch):
    log: list = []
    lookups: list = []

    def counting_orientation(name):
        lookups.append(name)
        return "minimize" if name in ("cost", "latency") else "maximize"

    monkeypatch.setattr(mo, "resolve_objective_orientation", counting_orientation)
    kwargs: dict[str, Any] = {}
    obj_arg = _LoggedDict(objectives, log, "objectives")
    w_arg = _LoggedDict(weights, log, "weights")
    m_arg = None if minimize is None else _LoggedList(minimize, log)
    if defs is not None:
        kwargs["objective_schema"] = _schema(log, defs)
    try:
        result = ("ok", repr(fn(obj_arg, w_arg, m_arg, **kwargs)))
    except Exception as exc:  # noqa: BLE001 - characterizing any exception
        result = ("err", type(exc).__name__, str(exc))
    # snapshot caller-visible state to prove no mutation differs
    after = (dict(obj_arg), dict(w_arg), None if m_arg is None else list(m_arg))
    return result, log, lookups, after


_OBJECTIVES = [
    {},
    {"accuracy": 0.9},
    {"accuracy": 0.9, "cost": 0.2},
    {"accuracy": 0.9, "cost": 0.2, "latency": 1.5, "f1": None},
    {"accuracy": None},
    {"accuracy": None, "cost": None},
    {"a": 0.1, "b": 0.7, "c": 0.3333333333333333, "d": 1e-17},
    {"accuracy": 0.0, "cost": 0.0},
    {"accuracy": math.inf, "cost": 1.0},
    {"accuracy": float("nan"), "cost": 1.0},
]
_WEIGHTS = [
    {},
    {"accuracy": 2.0},
    {"accuracy": 0.7, "cost": 0.3, "latency": 0.1},
    {"accuracy": 0.0, "cost": 0.0, "a": 0.0},
    {"a": 0.5, "b": 0.25, "c": 0.125, "d": 3.0},
    {"accuracy": 1.0, "cost": -1.0},
]
_MINIMIZE = [None, [], ["cost"], ["accuracy", "cost", "a"], ["unknown"]]


@pytest.mark.parametrize("objectives", _OBJECTIVES, ids=repr)
def test_plain_path_matches_oracle(objectives, monkeypatch):
    mismatches = []
    for weights, minimize in itertools.product(_WEIGHTS, _MINIMIZE):
        got = _run(
            scalarize_objectives, objectives, weights, minimize, None, monkeypatch
        )
        want = _run(
            oracle_scalarize_objectives,
            objectives,
            weights,
            minimize,
            None,
            monkeypatch,
        )
        if got != want:
            mismatches.append((weights, minimize, got, want))
    assert not mismatches


_BAND = {"low": 0.4, "high": 0.6}


def _defs_cases():
    names = ["accuracy", "cost", "latency", "quality"]
    orientations = ["maximize", "minimize", "band"]
    bands = [None, _BAND, {"low": None, "high": 1.0}, {"low": 0.1, "high": None}]
    weights = [1.0, 0.5, 0.0]
    cases = [[]]
    for orient in orientations:
        for band in bands:
            for w in weights:
                cases.append(
                    [
                        {
                            "name": "quality",
                            "orientation": orient,
                            "weight": w,
                            "band": band,
                        }
                    ]
                )
    for combo in itertools.product(orientations, repeat=3):
        cases.append(
            [
                {"name": names[i], "orientation": o, "weight": w, "band": _BAND}
                for i, (o, w) in enumerate(zip(combo, (1.0, 0.25, 0.75), strict=True))
            ]
        )
    cases.append(
        [
            {
                "name": "accuracy",
                "orientation": "maximize",
                "weight": 0.0,
                "band": None,
            },
            {"name": "cost", "orientation": "minimize", "weight": 0.0, "band": None},
        ]
    )
    return cases


_SCHEMA_OBJECTIVES = [
    {},
    {"accuracy": 0.9},
    {"accuracy": 0.9, "cost": 0.2},
    {"accuracy": 0.9, "cost": 0.2, "latency": 1.5, "quality": 0.5},
    {"quality": 0.9, "other": 4.0},
    {"quality": 0.1},
    {"quality": 0.55},
    {"quality": None, "accuracy": None},
    {"accuracy": 0.5, "unknown": 0.4},
]


def test_schema_path_matches_oracle(monkeypatch):
    mismatches = []
    for defs in _defs_cases():
        for objectives in _SCHEMA_OBJECTIVES:
            for minimize in (None, ["accuracy"]):
                got = _run(
                    scalarize_objectives,
                    objectives,
                    {"accuracy": 9.0},
                    minimize,
                    defs,
                    monkeypatch,
                )
                want = _run(
                    oracle_scalarize_objectives,
                    objectives,
                    {"accuracy": 9.0},
                    minimize,
                    defs,
                    monkeypatch,
                )
                if got != want:
                    mismatches.append((defs, objectives, minimize, got, want))
    assert not mismatches


@pytest.mark.parametrize(
    ("objectives", "weights", "minimize", "expected"),
    [
        ({}, {}, None, 0.0),
        ({"a": 1.0, "b": 3.0}, {}, [], 2.0),
        ({"a": 1.0, "b": 3.0}, {}, ["b"], -1.0),
        ({"a": 1.0, "b": 3.0}, {"a": 3.0, "b": 1.0}, [], 1.5),
        ({"a": None}, {}, [], 0.0),
        ({"a": 1.0, "b": 3.0}, {"a": 0.0, "b": 0.0}, ["a"], 1.0),
    ],
)
def test_documented_values(objectives, weights, minimize, expected):
    assert scalarize_objectives(objectives, weights, minimize) == expected
    assert oracle_scalarize_objectives(objectives, weights, minimize) == expected


def test_band_without_complete_target_raises():
    schema = SimpleNamespace(
        objectives=[
            SimpleNamespace(
                name="q",
                orientation="band",
                weight=1.0,
                band=SimpleNamespace(low=None, high=1.0),
            )
        ]
    )
    with pytest.raises(ValueError, match="requires a complete target band"):
        scalarize_objectives({"q": 0.5}, {}, None, schema)
