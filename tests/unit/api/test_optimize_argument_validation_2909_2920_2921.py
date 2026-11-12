"""#2909 / #2920 / #2921: @traigent.optimize argument handling fails early and clearly."""

from __future__ import annotations

import ast

import pytest

import traigent
from traigent.api.config_space import ConfigSpace
from traigent.api.parameter_ranges import IntRange
from traigent.config.types import accepted_algorithm_values
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
    assert "parallel_batch" not in msg
    assert "algorithm" in msg
    assert "traigent algorithms" in msg
    listed = ast.literal_eval(
        msg.split("Available optimizers: ")[1].split("]")[0] + "]"
    )
    assert listed == list(accepted_algorithm_values())
    # The CLI renders the same source.
    from click.testing import CliRunner

    from traigent.cli.main import cli

    out = CliRunner().invoke(cli, ["algorithms"]).output
    assert all(name in out for name in listed)
    assert "parallel_batch" not in out


def test_batch_optimizer_still_works_when_named_2909():
    opt = get_optimizer("parallel_batch", {"x": [1, 2]}, ["accuracy"])
    assert opt is not None


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


@pytest.mark.parametrize("key", ["a-b", "1x", "a b", ""])
def test_config_space_object_rejects_invalid_keys_2921(key):
    with pytest.raises(ValueError):
        ConfigSpace(tvars={key: IntRange(0, 5, name=key)})
    with pytest.raises(ValueError):
        ConfigSpace.from_decorator_args(configuration_space={key: IntRange(0, 5)})


def test_config_space_object_accepts_valid_keys_2921():
    ConfigSpace(tvars={"k": IntRange(0, 5, name="k")})


_TVL_BAD = """
configuration_space:
  "a-b":
    type: categorical
    values: [1, 2]
objectives:
  - name: accuracy
    direction: maximize
"""

_TVL_BAD_TVARS = """
tvl_version: "0.9"
tvars:
  - name: a-b
    type: enum[int]
    domain: [1, 2]
objectives:
  - name: accuracy
    direction: maximize
"""


@pytest.mark.parametrize("text", [_TVL_BAD, _TVL_BAD_TVARS])
def test_tvl_loader_rejects_invalid_keys_2921(tmp_path, text):
    from traigent.tvl.spec_loader import load_tvl_spec
    from traigent.utils.exceptions import TVLValidationError

    spec = tmp_path / "bad.tvl.yml"
    spec.write_text(text)
    with pytest.raises(TVLValidationError) as exc:
        load_tvl_spec(spec_path=spec)
    assert "a-b" not in str(exc.value)


def test_decorator_with_tvl_spec_bad_key_rejected_2921(tmp_path):
    from traigent.utils.exceptions import ValidationError

    spec = tmp_path / "bad.tvl.yml"
    spec.write_text(_TVL_BAD)
    with pytest.raises((ValueError, ValidationError)):
        traigent.optimize(tvl_spec=str(spec), eval_dataset=_DATA)(lambda x: x)
