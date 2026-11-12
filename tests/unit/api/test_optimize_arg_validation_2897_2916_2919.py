"""Argument validation for @traigent.optimize and its stop conditions (#2897, #2916, #2919)."""

from __future__ import annotations

import math

import pytest

import traigent
from traigent.core import stop_conditions as _sc

DATASET = [{"input": {"x": 1}, "output": "a"}]
SPACE = {"t": [0.0, 1.0]}


def _decorate(**kwargs):
    def fn(x):
        return "a"

    return traigent.optimize(
        configuration_space=SPACE,
        objectives=kwargs.pop("objectives", ["accuracy"]),
        eval_dataset=DATASET,
        **kwargs,
    )(fn)


class TestStopConditionValidators:
    @pytest.mark.parametrize("bad", [True, False, "x", "1e-3", None, [1]])
    def test_epsilon_rejects_non_numbers(self, bad):
        with pytest.raises(TypeError, match="plateau_epsilon"):
            _sc.validate_plateau_epsilon(bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.5])
    def test_epsilon_rejects_nonfinite_and_negative(self, bad):
        with pytest.raises(ValueError, match="plateau_epsilon"):
            _sc.validate_plateau_epsilon(bad)

    def test_epsilon_zero_stays_zero(self):
        assert _sc.validate_plateau_epsilon(0) == 0.0
        assert _sc.validate_plateau_epsilon(1e-3) == 1e-3

    @pytest.mark.parametrize("bad", [True, "5", [1]])
    def test_limit_rejects_non_numbers(self, bad):
        with pytest.raises(TypeError, match="metric_limit"):
            _sc.validate_metric_limit(bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, 0, -1.0])
    def test_limit_rejects_nonfinite_and_non_positive(self, bad):
        with pytest.raises(ValueError, match="metric_limit"):
            _sc.validate_metric_limit(bad)

    def test_limit_accepts_positive(self):
        assert _sc.validate_metric_limit(5) == 5.0

    @pytest.mark.parametrize(
        "key,value",
        [("plateau_epsilon", math.nan), ("metric_limit", True)],
    )
    def test_decorator_rejects_bad_override(self, key, value):
        with pytest.raises((TypeError, ValueError), match=key):
            _decorate(**{key: value})


class TestFlatBoolFlags:
    @pytest.mark.parametrize("bad", ["false", "true", 0, 1, "no"])
    def test_content_less_non_bool_rejected(self, bad):
        with pytest.raises(TypeError, match="content_less"):
            _decorate(content_less=bad)

    def test_content_less_real_bool_accepted(self):
        assert _decorate(content_less=False) is not None
        assert _decorate(content_less=True) is not None

    @pytest.mark.parametrize("bad", ["false", "true", 0, 1, None])
    def test_direct_construction_effectuation_must_be_bool(self, bad):
        from traigent.core.optimized_function import OptimizedFunction

        with pytest.raises(TypeError, match="effectuation"):
            OptimizedFunction(
                func=lambda x: "a",
                configuration_space=SPACE,
                eval_dataset=DATASET,
                effectuation=bad,
            )

    def test_decorator_string_false_is_off(self):
        # Decorator path coerces through the injection bundle (#2922): 'false' is off.
        assert _decorate(effectuation="false").effectuation is False
        assert (
            _decorate(auto_override_frameworks="false").auto_override_frameworks
            is False
        )


class TestEmptyObjectives:
    def test_decorator_empty_list_raises(self):
        with pytest.raises(ValueError, match="objectives"):
            _decorate(objectives=[])

    def test_decorator_omitted_keeps_default(self):
        def fn(x):
            return "a"

        f = traigent.optimize(configuration_space=SPACE, eval_dataset=DATASET)(fn)
        assert f.objectives == ["accuracy"]

    @pytest.mark.asyncio
    async def test_call_time_empty_list_raises(self):
        f = _decorate()
        with pytest.raises(ValueError, match="objectives"):
            await f.optimize(objectives=[], max_trials=1)


SENTINEL = "zz-distinct-9137"


class TestMessagesDoNotEchoValues:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: _sc.validate_plateau_epsilon(SENTINEL),
            lambda: _sc.validate_plateau_epsilon(-912345.5),
            lambda: _sc.validate_metric_limit(SENTINEL),
            lambda: _sc.validate_metric_limit(-912345.5),
            lambda: _sc.validate_metric_limit(float("nan")),
            lambda: _decorate(content_less=SENTINEL),
        ],
    )
    def test_value_absent_from_message(self, call):
        with pytest.raises((TypeError, ValueError)) as info:
            call()
        text = str(info.value)
        assert "zz-distinct-9137" not in text
        assert "912345" not in text


class TestCallTimeStopOverrides:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"plateau_epsilon": float("nan")},
            {"plateau_epsilon": -1.0},
            {"plateau_window": 0, "plateau_epsilon": "x"},
            {"metric_limit": float("nan"), "metric_name": "accuracy"},
        ],
    )
    async def test_bad_call_time_value_rejected_without_plateau_window(self, kwargs):
        f = _decorate()
        with pytest.raises((TypeError, ValueError)):
            await f.optimize(max_trials=1, **kwargs)


class TestEmptyObjectivesLeaveNoTvlState:
    @pytest.mark.asyncio
    async def test_empty_objectives_rejected_before_tvl_applied(self):
        from unittest.mock import patch

        f = _decorate()
        with patch.object(f, "_process_tvl_options") as process:
            with pytest.raises(ValueError, match="objectives"):
                await f.optimize(objectives=[], tvl_spec="demo.tvl.yml", max_trials=1)
        process.assert_not_called()
