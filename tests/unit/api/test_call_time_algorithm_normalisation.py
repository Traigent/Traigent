"""optimize(algorithm=...) normalises like the decorator."""

from __future__ import annotations

import pytest

import traigent
from traigent.utils.exceptions import OptimizationError


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


# --- algorithm normalisation ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["RANDOM", " random ", "Random"])
async def test_call_time_algorithm_is_normalised(name, tmp_path):
    fn = _decorate(max_trials=1)
    result = await fn.optimize(algorithm=name, max_trials=1)
    assert len(result.trials) == 1


@pytest.mark.asyncio
async def test_call_time_unknown_algorithm_names_choices():
    fn = _decorate(max_trials=1)
    with pytest.raises(OptimizationError, match="Unknown optimizer"):
        await fn.optimize(algorithm="nonsense", max_trials=1)
