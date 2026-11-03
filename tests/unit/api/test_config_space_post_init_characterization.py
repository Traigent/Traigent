"""Golden characterization for ConfigSpace.__post_init__ knob/tvar coherence."""

from __future__ import annotations

import pickle
from types import MappingProxyType

import pytest

from traigent.api.config_space import ConfigSpace
from traigent.api.parameter_ranges import Choices, Range
from traigent.knobs import Knob
from traigent.knobs.bindings import Fixed, Tuned


def _tuned(name, rng):
    return Knob(name=name, binding=Tuned(range=rng, default=rng.get_default()))


def test_tvars_only_synthesises_all_tuned_knobs():
    t = Range(0.0, 1.0, name="t")
    c = Choices(["a", "b"], name="c")
    space = ConfigSpace(tvars={"t": t, "c": c}, constraints=[])
    assert isinstance(space.tvars, MappingProxyType)
    assert isinstance(space.knobs, MappingProxyType)
    assert list(space.tvars) == ["t", "c"]
    assert list(space.knobs) == ["t", "c"]
    assert all(k.is_tuned() for k in space.knobs.values())
    assert space.knobs["t"].binding.range is t
    assert space.constraints == ()
    assert isinstance(space.constraints, tuple)


def test_empty_defaults():
    space = ConfigSpace()
    assert dict(space.tvars) == {} and dict(space.knobs) == {}
    assert space.constraints == ()


def test_knobs_only_derives_tuned_projection():
    t = Range(0.0, 1.0, name="t")
    knobs = {"t": _tuned("t", t), "f": Knob(name="f", binding=Fixed(value=3))}
    space = ConfigSpace(knobs=knobs)
    assert list(space.tvars) == ["t"]
    assert space.tvars["t"] is t
    assert list(space.knobs) == ["t", "f"]
    assert isinstance(space.knobs, MappingProxyType)
    assert isinstance(space.tvars, MappingProxyType)


def test_knobs_with_matching_tvars_is_accepted():
    t = Range(0.0, 1.0, name="t")
    space = ConfigSpace(tvars={"t": t}, knobs={"t": _tuned("t", t)})
    assert space.tvars["t"] is t


def test_knobs_with_mismatching_tvars_rejected():
    t = Range(0.0, 1.0, name="t")
    other = Range(0.0, 2.0, name="t")
    with pytest.raises(ValueError, match="tvars must be the Tuned-only projection"):
        ConfigSpace(tvars={"t": other}, knobs={"t": _tuned("t", t)})
    with pytest.raises(ValueError, match="tvars must be the Tuned-only projection"):
        ConfigSpace(tvars={"x": t}, knobs={"f": Knob(name="f", binding=Fixed(value=1))})


def test_knob_validation_errors():
    t = Range(0.0, 1.0, name="t")
    with pytest.raises(TypeError, match=r"knobs\['t'\] is not a Knob"):
        ConfigSpace(knobs={"t": object()})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match=r"knob mapping key 'x' != knob.name 't'"):
        ConfigSpace(knobs={"x": _tuned("t", t)})
    # knobs are checked in order: the first bad one decides the error
    with pytest.raises(ValueError):
        ConfigSpace(knobs={"x": _tuned("t", t), "y": 5})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        ConfigSpace(knobs={"y": 5, "x": _tuned("t", t)})  # type: ignore[dict-item]


def test_constraints_become_tuple_and_input_dict_is_copied():
    t = Range(0.0, 1.0, name="t")
    src = {"t": t}
    space = ConfigSpace(tvars=src, constraints=iter([]))  # type: ignore[arg-type]
    src["u"] = t
    assert list(space.tvars) == ["t"]
    assert space.constraints == ()


def test_pickle_roundtrip_keeps_surfaces():
    t = Range(0.0, 1.0, name="t")
    space = ConfigSpace(tvars={"t": t})
    clone = pickle.loads(pickle.dumps(space))
    assert list(clone.tvars) == ["t"] and list(clone.knobs) == ["t"]
