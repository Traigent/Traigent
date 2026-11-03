"""Golden characterization for TraigentCloudClient._fallback_optimization."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from traigent.cloud.client import CloudServiceError, TraigentCloudClient
from traigent.evaluators.base import Dataset, EvaluationExample


def _client() -> TraigentCloudClient:
    return TraigentCloudClient(
        api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
        base_url="http://localhost",
    )


def _dataset() -> Dataset:
    return Dataset([EvaluationExample({"text": "x"}, "x")], name="d")


def _run(
    result: Any,
    *,
    function_name: str = "dummy",
    local_function: Any = lambda text: text,
    objectives: list[str] | None = None,
    max_trials: int = 4,
) -> tuple[Any, dict[str, Any]]:
    seen: dict[str, Any] = {"calls": []}
    objectives = ["accuracy"] if objectives is None else objectives

    def fake_get_optimizer(*args: Any, **kwargs: Any) -> str:
        seen["calls"].append("get_optimizer")
        seen["optimizer_kwargs"] = kwargs
        return "opt"

    async def go() -> Any:
        with (
            patch("traigent.core.orchestrator.OptimizationOrchestrator") as orch_cls,
            patch("traigent.evaluators.local.LocalEvaluator"),
            patch("traigent.optimizers.registry.get_optimizer", fake_get_optimizer),
        ):
            orch_cls.return_value.optimize = AsyncMock(return_value=result)
            seen["orch_cls"] = orch_cls
            return await _client()._fallback_optimization(
                function_name=function_name,
                dataset=_dataset(),
                configuration_space={"k": ["v"]},
                objectives=objectives,
                max_trials=max_trials,
                local_function=local_function,
            )

    return asyncio.run(go()), seen


def _res(**kw: Any) -> SimpleNamespace:
    base = {"trials": [], "best_config": {}, "best_metrics": {}, "best_score": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_best_metrics_skips_non_numeric_and_stringifies_keys() -> None:
    out, _ = _run(
        _res(best_metrics={1: "2.5", "bad": "x", "none": None, "ok": 3}),
    )
    assert out.best_metrics == {"1": 2.5, "ok": 3.0}
    assert out.subset_used is False
    assert out.cost_reduction == 0.0
    assert out.trials_count == 0


def test_best_config_non_mapping_becomes_empty_and_keys_stringified() -> None:
    assert _run(_res(best_config=[1, 2]))[0].best_config == {}
    assert _run(_res(best_config={1: "a"}))[0].best_config == {"1": "a"}


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (None, {}),
        ("0.75", {"accuracy": 0.75}),
        (2, {"accuracy": 2.0}),
        ("nope", {}),
        (object(), {}),
        (float("nan"), None),  # nan compares unequal; checked separately
    ],
)
def test_best_score_fallback(score: Any, expected: Any) -> None:
    out, _ = _run(_res(best_score=score))
    if expected is None:
        assert list(out.best_metrics) == ["accuracy"]
        assert out.best_metrics["accuracy"] != out.best_metrics["accuracy"]
    else:
        assert out.best_metrics == expected


def test_best_score_fallback_not_used_when_metrics_present_or_no_objectives() -> None:
    out, _ = _run(_res(best_metrics={"m": 1}, best_score=9))
    assert out.best_metrics == {"m": 1.0}
    out, _ = _run(_res(best_score=9), objectives=[])
    assert out.best_metrics == {}
    # metrics that all fail conversion count as empty -> fallback applies
    out, _ = _run(_res(best_metrics={"m": "x"}, best_score=9))
    assert out.best_metrics == {"accuracy": 9.0}


def test_best_metrics_non_mapping_falls_back_to_score() -> None:
    out, _ = _run(_res(best_metrics=5, best_score=1))
    assert out.best_metrics == {"accuracy": 1.0}


@pytest.mark.parametrize(("requested", "expected"), [(0, 1), (-5, 1), (7, 7), (50, 20)])
def test_max_trials_clamped(requested: int, expected: int) -> None:
    _, seen = _run(_res(), max_trials=requested)
    assert seen["optimizer_kwargs"]["max_trials"] == expected
    assert seen["orch_cls"].call_args.kwargs["max_trials"] == expected


def test_trials_none_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _run(_res(trials=None))


def test_missing_local_function_without_dotted_name_errors_before_optimizer() -> None:
    seen: dict[str, Any] = {}

    async def go() -> None:
        with patch(
            "traigent.optimizers.registry.get_optimizer",
            side_effect=AssertionError("must not be reached"),
        ):
            await _client()._fallback_optimization(
                "plainname", _dataset(), {"k": [1]}, ["accuracy"], 3
            )

    with pytest.raises(CloudServiceError, match="requires local_function"):
        asyncio.run(go())
    assert seen == {}


def test_non_callable_local_function_is_treated_as_missing() -> None:
    with pytest.raises(CloudServiceError, match="requires local_function"):
        _run(_res(), local_function="not callable")


def test_dotted_path_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("golden_fb_mod")
    mod.fn = lambda text: text  # type: ignore[attr-defined]
    mod.value = 3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "golden_fb_mod", mod)

    out, seen = _run(
        _res(trials=[1, 2]), function_name="golden_fb_mod.fn", local_function=None
    )
    assert out.trials_count == 2
    assert (
        seen["orch_cls"].return_value.optimize.call_args.kwargs["func"] is mod.fn  # type: ignore[attr-defined]
    )

    with pytest.raises(CloudServiceError, match="is not callable"):
        _run(_res(), function_name="golden_fb_mod.value", local_function=None)
    with pytest.raises(CloudServiceError, match="Unable to resolve") as ei:
        _run(_res(), function_name="golden_fb_mod.missing", local_function=None)
    assert isinstance(ei.value.__cause__, AttributeError)
    with pytest.raises(CloudServiceError, match="Unable to resolve") as ei:
        _run(_res(), function_name="no_such_mod_xyz.fn", local_function=None)
    assert isinstance(ei.value.__cause__, ImportError)
