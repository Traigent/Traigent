"""Characterization of the discrete config-parameter helpers.

The ``oracle_*`` functions below are frozen copies of the pre-refactor
implementations. The tests compare results, raised exceptions, and the exact
sequence of operations performed on the definition (so evaluation order is
pinned, not just the return value).
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from traigent.utils import discrete_domains as dd
from traigent.utils.discrete_domains import (
    discrete_cardinality_for_config_param,
    discrete_values_for_config_param,
    stepped_float_cardinality,
    stepped_float_values,
    stepped_int_cardinality,
    stepped_int_values,
)

# --- frozen pre-refactor implementations ---


def oracle_cardinality(definition: Any) -> int | None:
    """Return finite cardinality for a config-space parameter, or None if continuous."""

    if isinstance(definition, list):
        return len(definition)

    if isinstance(definition, tuple) and len(definition) == 2:
        return None

    if isinstance(definition, dict):
        param_type = str(definition.get("type") or "").lower()
        has_range = "low" in definition and "high" in definition
        has_categorical_values = "choices" in definition or "values" in definition

        if param_type in {"categorical", "choice"} or (
            not param_type and not has_range and has_categorical_values
        ):
            choices = (
                definition["choices"]
                if "choices" in definition
                else definition.get("values", ())
            )
            try:
                return len(choices)
            except TypeError:
                return None

        if param_type in {"fixed", "constant"}:
            return 1

        if has_range:
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                return None

            if param_type in {"int", "integer"}:
                step = definition.get("step", 1)
                if step is None:
                    step = 1
                try:
                    low_i = int(low)
                    high_i = int(high)
                    step_i = int(step)
                except (TypeError, ValueError):
                    return None
                if step_i <= 0 or low_i > high_i:
                    return None
                return dd.stepped_int_cardinality(low_i, high_i, step_i)

            step = definition.get("step")
            if step is None:
                return None
            try:
                low_f = float(low)
                high_f = float(high)
                step_f = float(step)
            except (TypeError, ValueError):
                return None
            if step_f <= 0 or low_f > high_f:
                return None
            return dd.stepped_float_cardinality(low_f, high_f, step_f)

        return 1

    return 1


def oracle_values(definition: Any) -> tuple[Any, ...] | None:
    """Return finite values for a config-space parameter, or None if continuous."""

    if isinstance(definition, list):
        return tuple(definition)

    if isinstance(definition, tuple) and len(definition) == 2:
        return None

    if isinstance(definition, dict):
        param_type = str(definition.get("type") or "").lower()
        has_range = "low" in definition and "high" in definition
        has_categorical_values = "choices" in definition or "values" in definition

        if param_type in {"categorical", "choice"} or (
            not param_type and not has_range and has_categorical_values
        ):
            choices = (
                definition["choices"]
                if "choices" in definition
                else definition.get("values", ())
            )
            return tuple(choices)

        if param_type in {"fixed", "constant"}:
            if "value" in definition:
                return (definition["value"],)
            if "default" in definition:
                return (definition["default"],)
            return (definition,)

        if has_range:
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                return None

            if param_type in {"int", "integer"}:
                step = definition.get("step", 1)
                if step is None:
                    step = 1
                try:
                    low_i = int(low)
                    high_i = int(high)
                    step_i = int(step)
                except (TypeError, ValueError):
                    return None
                if step_i <= 0 or low_i > high_i:
                    return ()
                return dd.stepped_int_values(low_i, high_i, step_i)

            step = definition.get("step")
            if step is None:
                return None
            try:
                low_f = float(low)
                high_f = float(high)
                step_f = float(step)
            except (TypeError, ValueError):
                return None
            if step_f <= 0 or low_f > high_f:
                return ()
            return dd.stepped_float_values(low_f, high_f, step_f)

        return (definition,)

    return (definition,)


class _Recording(dict):
    """dict that logs every read-style operation the code under test performs."""

    def __init__(self, data, log):
        super().__init__(data)
        self._log = log

    def get(self, key, *default):
        self._log.append(("get", key) + tuple(default))
        return super().get(key, *default)

    def __contains__(self, key):
        self._log.append(("in", key))
        return super().__contains__(key)

    def __getitem__(self, key):
        self._log.append(("item", key))
        return super().__getitem__(key)


class _Num:
    """Numeric-like object whose conversions are logged."""

    def __init__(self, name, value, log):
        self.name = name
        self.value = value
        self.log = log

    def __int__(self):
        self.log.append(("int", self.name))
        return int(self.value)

    def __float__(self):
        self.log.append(("float", self.name))
        return float(self.value)


def _run(fn, definition_factory):
    """Run ``fn`` on a fresh definition, recording stepped-helper calls too."""
    log: list = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in (
            "stepped_int_cardinality",
            "stepped_float_cardinality",
            "stepped_int_values",
            "stepped_float_values",
        ):
            real = getattr(dd, name)

            def wrapper(*args, _real=real, _name=name):
                log.append(("call", _name, args))
                return _real(*args)

            monkeypatch.setattr(dd, name, wrapper)
        definition = definition_factory(log)
        try:
            result = ("ok", repr(fn(definition)))
        except Exception as exc:  # noqa: BLE001 - characterizing any exception
            result = ("err", type(exc).__name__)
    return result, log


class _RaisingRead(dict):
    """dict whose read of one key raises, to pin where each read happens."""

    def __init__(self, data, log, key, error):
        super().__init__(data)
        self._log = log
        self._key = key
        self._error = error

    def __getitem__(self, key):
        self._log.append(("item", key))
        if key == self._key:
            raise self._error
        return super().__getitem__(key)

    def get(self, key, *default):
        self._log.append(("get", key) + tuple(default))
        if key == self._key:
            raise self._error
        return super().get(key, *default)


_ABSENT = object()
_TYPES = [
    _ABSENT,
    None,
    0,
    False,
    "",
    "int",
    "Integer",
    "float",
    "Categorical",
    "choice",
    "fixed",
    "CONSTANT",
    "other",
    7,
]
_LOWS = [_ABSENT, None, 0, 0.5, "2", "x", -3]
_HIGHS = [_ABSENT, None, 10, 3.2, "9", "y", -5]
_STEPS = [_ABSENT, None, 1, 0, -1, 0.25, "2", "z"]
_CHOICES = [_ABSENT, [1, 2], None, 5, ()]
_VALUES = [_ABSENT, (3,)]
_FIXED = [_ABSENT, "value", "default", "both"]


def _definitions():
    for t, lo, hi, st, ch, va, fx in itertools.product(
        _TYPES, _LOWS, _HIGHS, _STEPS, _CHOICES, _VALUES, _FIXED
    ):
        spec = {}
        for key, val in (
            ("type", t),
            ("low", lo),
            ("high", hi),
            ("step", st),
            ("choices", ch),
            ("values", va),
        ):
            if val is not _ABSENT:
                spec[key] = val
        if fx in ("value", "both"):
            spec["value"] = 11
        if fx in ("default", "both"):
            spec["default"] = 12
        yield spec


def _stride_sample(items, stride):
    return [item for idx, item in enumerate(items) if idx % stride == 0]


_ALL_DEFS = list(_definitions())


@pytest.mark.parametrize(
    ("new", "oracle"),
    [
        (discrete_cardinality_for_config_param, oracle_cardinality),
        (discrete_values_for_config_param, oracle_values),
    ],
    ids=["cardinality", "values"],
)
def test_matches_oracle_for_dict_definitions(new, oracle):
    mismatches = []
    for spec in _ALL_DEFS:
        got = _run(new, lambda log, s=spec: _Recording(s, log))
        want = _run(oracle, lambda log, s=spec: _Recording(s, log))
        if got != want:
            mismatches.append((spec, got, want))
            if len(mismatches) >= 3:
                break
    assert not mismatches


@pytest.mark.parametrize(
    ("new", "oracle"),
    [
        (discrete_cardinality_for_config_param, oracle_cardinality),
        (discrete_values_for_config_param, oracle_values),
    ],
    ids=["cardinality", "values"],
)
@pytest.mark.parametrize("kind", ["int", "float"])
def test_conversion_order_is_pinned(new, oracle, kind):
    for step in (_ABSENT, None, 2, "bad"):
        for low_bad, high_bad in itertools.product((False, True), repeat=2):

            def factory(log, step=step, low_bad=low_bad, high_bad=high_bad):
                spec = {
                    "type": kind,
                    "low": _Num("low", "x" if low_bad else 1, log),
                    "high": _Num("high", "y" if high_bad else 9, log),
                }
                if step is not _ABSENT:
                    spec["step"] = (
                        _Num("step", 0 if step == "bad" else step, log)
                        if step not in (None,)
                        else None
                    )
                return _Recording(spec, log)

            assert _run(new, factory) == _run(oracle, factory)


_NON_DICT = [
    [],
    [1, 2, 3],
    (1, 2),
    (1, 2, 3),
    (),
    None,
    0,
    5,
    2.5,
    "text",
    {1, 2},
    b"ab",
    object,
    range(3),
]


@pytest.mark.parametrize(
    ("new", "oracle"),
    [
        (discrete_cardinality_for_config_param, oracle_cardinality),
        (discrete_values_for_config_param, oracle_values),
    ],
    ids=["cardinality", "values"],
)
@pytest.mark.parametrize("definition", _NON_DICT, ids=repr)
def test_non_dict_definitions(new, oracle, definition):
    assert _run(new, lambda log: definition) == _run(oracle, lambda log: definition)


@pytest.mark.parametrize(
    ("definition", "cardinality", "values"),
    [
        ([1, 2, 3], 3, (1, 2, 3)),
        ((0.0, 1.0), None, None),
        ({"type": "int", "low": 0, "high": 10, "step": 4}, 4, (0, 4, 8, 10)),
        ({"type": "float", "low": 0, "high": 1, "step": 0.5}, 3, (0.0, 0.5, 1.0)),
        ({"type": "float", "low": 0, "high": 1}, None, None),
        ({"type": "int", "low": 5, "high": 1}, None, ()),
        ({"type": "fixed", "value": 3}, 1, (3,)),
        ({"choices": ["a", "b"]}, 2, ("a", "b")),
        ({"low": 0, "high": 1, "step": 0.5}, 3, (0.0, 0.5, 1.0)),
        ({"anything": 1}, 1, ({"anything": 1},)),
        (7, 1, (7,)),
    ],
)
def test_documented_examples(definition, cardinality, values):
    assert discrete_cardinality_for_config_param(definition) == cardinality
    assert discrete_values_for_config_param(definition) == values
    assert stepped_int_cardinality(0, 10, 4) == 4
    assert stepped_int_values(0, 10, 4) == (0, 4, 8, 10)
    assert stepped_float_cardinality(0, 1, 0.5) == 3
    assert stepped_float_values(0, 1, 0.5) == (0.0, 0.5, 1.0)


@pytest.mark.parametrize(
    ("new", "oracle"),
    [
        (discrete_cardinality_for_config_param, oracle_cardinality),
        (discrete_values_for_config_param, oracle_values),
    ],
    ids=["cardinality", "values"],
)
@pytest.mark.parametrize(
    "error", [TypeError("boom"), ValueError("boom"), KeyError("k")]
)
@pytest.mark.parametrize(
    "key", ["choices", "values", "type", "low", "high", "step", "value"]
)
@pytest.mark.parametrize(
    "data",
    [
        {"type": "choice", "choices": [1, 2]},
        {"choices": [1, 2], "values": (3,)},
        {"values": (3,)},
        {"type": "fixed", "value": 1, "default": 2},
        {"type": "int", "low": 0, "high": 9, "step": 3},
        {"type": "float", "low": 0, "high": 1, "step": 0.5},
    ],
)
def test_exceptions_raised_by_reads_surface_identically(new, oracle, data, key, error):
    def factory(log):
        return _RaisingRead(data, log, key, error)

    assert _run(new, factory) == _run(oracle, factory)
