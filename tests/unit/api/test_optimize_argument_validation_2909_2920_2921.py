"""#2909 / #2920 / #2921: @traigent.optimize argument handling fails early and clearly."""

from __future__ import annotations

import pytest

import traigent
from traigent.optimizers.registry import get_optimizer
from traigent.utils.exceptions import OptimizationError

_DATA = [{"input": {"x": 1}, "output": 1}]


def _decorate(**kwargs):
    kwargs.setdefault("configuration_space", {"a": [1, 2]})
    kwargs.setdefault("objectives", ["accuracy"])
    kwargs.setdefault("eval_dataset", _DATA)

    def fn(x, a=1):
        return a

    return traigent.optimize(**kwargs)(fn)


def test_unknown_optimizer_lists_only_selectable_names_2909():
    with pytest.raises(OptimizationError) as exc:
        get_optimizer("foo", {"x": [1]}, ["accuracy"])
    msg = str(exc.value)
    assert "remote" not in msg
    assert "algorithm" in msg
    assert "traigent algorithms" in msg
    assert "grid" in msg and "random" in msg


@pytest.mark.parametrize(
    "arg", ["agent_name", "experiment_name", "run_title", "run_description"]
)
@pytest.mark.parametrize("bad", [5, ["a"], True])
def test_non_string_text_args_raise_typeerror_2920(arg, bad):
    with pytest.raises(TypeError) as exc:
        _decorate(**{arg: bad})
    assert arg in str(exc.value)
    assert repr(bad) not in str(exc.value)


def test_string_text_args_still_accepted_2920():
    _decorate(agent_name="a", run_title="t", run_description="d")
    _decorate(agent_name=None)


@pytest.mark.parametrize(
    "key", ["", " ", "a b", "1x", "a-b", "a..b", ".a", "a.", "a/b"]
)
def test_invalid_configuration_space_keys_rejected_2921(key):
    with pytest.raises(ValueError) as exc:
        _decorate(configuration_space={key: [1, 2]})
    assert "configuration_space" in str(exc.value)
    if key.strip():
        assert key not in str(exc.value)


@pytest.mark.parametrize("key", ["a", "_a", "model_name", "a.b", "temp2"])
def test_valid_configuration_space_keys_accepted_2921(key):
    _decorate(configuration_space={key: [1, 2]})
