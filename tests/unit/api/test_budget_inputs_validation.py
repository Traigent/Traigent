"""max_trials and timeout reject NaN, bool and non-numeric values by name."""

from __future__ import annotations

import pytest

import traigent
from traigent.core.orchestrator_helpers import validate_constructor_arguments

NAN = float("nan")
INF = float("inf")

# --- max_trials / timeout ----------------------------------------------------


def _decorate(**kw):
    @traigent.optimize(
        eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
        configuration_space={"x": [1, 2]},
        objectives=["accuracy"],
        **kw,
    )
    def fn(q):
        return q

    return fn


@pytest.mark.parametrize("bad", [NAN, INF, True, 2.5, "5", 0, -1])
def test_decorator_rejects_bad_max_trials(bad):
    with pytest.raises((ValueError, TypeError), match="max_trials"):
        _decorate(max_trials=bad)


@pytest.mark.parametrize("bad", [NAN, True, 2.5, "5", 0, -1])
def test_optimize_inputs_reject_bad_max_trials(bad):
    with pytest.raises((ValueError, TypeError), match="max_trials"):
        _call_helper(max_trials=bad)


@pytest.mark.parametrize("bad", [NAN, INF, True, "3", -1])
def test_optimize_inputs_reject_bad_timeout(bad):
    with pytest.raises((ValueError, TypeError), match="timeout"):
        _call_helper(timeout=bad)


def _call_helper(**kw):
    from unittest.mock import MagicMock

    from traigent.evaluators.base import BaseEvaluator
    from traigent.optimizers.base import BaseOptimizer

    return validate_constructor_arguments(
        MagicMock(spec=BaseOptimizer), MagicMock(spec=BaseEvaluator), **kw
    )


def test_valid_max_trials_and_timeout_still_accepted():
    _decorate(max_trials=3)
    _call_helper(timeout=0, max_trials=2)
    _call_helper(timeout=1.5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kw",
    [
        {"max_trials": NAN},
        {"max_trials": True},
        {"max_trials": 2.5},
        {"timeout": NAN},
        {"timeout": True},
        {"timeout": "3"},
    ],
)
async def test_call_time_overrides_validated_with_field_name(kw):
    fn = _decorate(max_trials=2)
    with pytest.raises((ValueError, TypeError), match=next(iter(kw))):
        await fn.optimize(algorithm="random", **kw)


def test_timeout_too_large_for_a_float_names_the_field() -> None:
    """math.isfinite(10**400) raises OverflowError; the error must name timeout."""
    from traigent.utils.validation import validate_timeout_value

    with pytest.raises(ValueError, match="timeout"):
        validate_timeout_value(10**400)


def test_timeout_with_more_digits_than_str_allows_still_names_the_field() -> None:
    """repr() of an int past Python's 4300-digit limit raises; the message must not."""
    from traigent.utils.validation import Validators, validate_timeout_value

    with pytest.raises(ValueError, match="timeout"):
        validate_timeout_value(10**5000)
    assert not Validators.validate_number(10**5000, "x").is_valid
