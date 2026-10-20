"""Captured LLM responses are attributed to examples by ownership (#2444).

Each example's evaluation runs in its own capture-ownership scope, so spend is
attributed exactly whatever the number of calls per example and whatever order
concurrent examples finish in -- never by counting or divisibility.
"""

import asyncio
from typing import Any

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import extract_llm_metrics
from traigent.utils.langchain_interceptor import capture_langchain_response

MODEL = "gpt-4o-mini"


class _Resp:
    def __init__(self, inp: int, out: int, *, model: str | None = MODEL) -> None:
        self.content = "ok"
        self.usage_metadata = {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
        }
        self.response_metadata = {"model_name": model} if model else {}


def _cost(resp: _Resp, model: str = MODEL) -> float:
    return float(extract_llm_metrics(response=resp, model_name=model).cost.total_cost)


def _dataset(n: int) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": str(i)},
                expected_output=None,
                metadata={"example_id": f"example_{i}"},
            )
            for i in range(n)
        ]
    )


def _evaluate(fn: Any, n: int, *, workers: int = 1, detailed: bool = True) -> Any:
    ev = LocalEvaluator(
        metrics=["accuracy"], timeout=10.0, max_workers=workers, detailed=detailed
    )
    return asyncio.run(ev.evaluate(fn, config={"model": MODEL}, dataset=_dataset(n)))


def _sizes(i: int, call: int) -> tuple[int, int]:
    # Distinct per (example, call) so any misattribution changes a cost.
    return 1000 * (i + 1) + 100 * call, 300 * (i + 1) + 50 * call


def _calls_fn(calls: list[int], seen: list[list[_Resp]]) -> Any:
    def fn(text: str, **_cfg: Any) -> str:
        i = int(text)
        made = []
        for c in range(calls[i]):
            r = _Resp(*_sizes(i, c))
            made.append(r)
            capture_langchain_response(r)
        seen[i] = made
        return "ok"

    return fn


@pytest.mark.parametrize("calls", [[1, 1, 2], [1, 3], [3, 1, 1], [2, 2]])
def test_uneven_and_even_calls_get_exact_per_example_costs(calls: list[int]) -> None:
    seen: list[list[_Resp]] = [[] for _ in calls]
    res = _evaluate(_calls_fn(calls, seen), len(calls))
    for i, er in enumerate(res.example_results):
        assert er.metrics["total_cost"] == pytest.approx(sum(_cost(r) for r in seen[i]))
    total = sum(_cost(r) for rs in seen for r in rs)
    m = res.aggregated_metrics
    assert not m.get("cost_unmeasured")
    assert m["cost"] == pytest.approx(total)


def test_concurrent_examples_completing_out_of_order_keep_their_own_spend() -> None:
    calls = [2, 1, 3]
    seen: list[list[_Resp]] = [[] for _ in calls]

    async def fn(text: str, **_cfg: Any) -> str:
        i = int(text)
        await asyncio.sleep(0.05 * (len(calls) - i))  # example 0 finishes last
        for c in range(calls[i]):
            r = _Resp(*_sizes(i, c))
            seen[i].append(r)
            capture_langchain_response(r)
            await asyncio.sleep(0)
        return "ok"

    res = _evaluate(fn, len(calls), workers=3)
    for i, er in enumerate(res.example_results):
        assert er.metrics["total_cost"] == pytest.approx(sum(_cost(r) for r in seen[i]))


def test_concurrent_sync_workers_keep_their_own_spend() -> None:
    import time

    calls = [1, 3, 2]
    seen: list[list[_Resp]] = [[] for _ in calls]

    def fn(text: str, **_cfg: Any) -> str:
        i = int(text)
        time.sleep(0.05 * (len(calls) - i))
        for c in range(calls[i]):
            r = _Resp(*_sizes(i, c))
            seen[i].append(r)
            capture_langchain_response(r)
        return "ok"

    res = _evaluate(fn, len(calls), workers=3)
    for i, er in enumerate(res.example_results):
        assert er.metrics["total_cost"] == pytest.approx(sum(_cost(r) for r in seen[i]))


def test_raw_response_dict_output_is_counted_exactly_once() -> None:
    calls = [1, 1, 2]
    seen: list[list[_Resp]] = [[] for _ in calls]

    def fn(text: str, **_cfg: Any) -> Any:
        i = int(text)
        for c in range(calls[i]):
            r = _Resp(*_sizes(i, c))
            seen[i].append(r)
            capture_langchain_response(r)
        if i == 0:
            return {"text": "ok", "raw_response": seen[0][0]}
        return "ok"

    res = _evaluate(fn, 3, detailed=False)
    total = sum(_cost(r) for rs in seen for r in rs)
    m = res.aggregated_metrics
    assert not m.get("cost_unmeasured")
    assert m["cost"] == pytest.approx(total)


def test_direct_response_object_output_is_counted_exactly_once() -> None:
    calls = [1, 1, 2]
    seen: list[list[_Resp]] = [[] for _ in calls]

    def fn(text: str, **_cfg: Any) -> Any:
        i = int(text)
        for c in range(calls[i]):
            r = _Resp(*_sizes(i, c))
            seen[i].append(r)
            capture_langchain_response(r)
        return seen[0][0] if i == 0 else "ok"

    res = _evaluate(fn, 3, detailed=False)
    total = sum(_cost(r) for rs in seen for r in rs)
    m = res.aggregated_metrics
    assert not m.get("cost_unmeasured")
    assert m["cost"] == pytest.approx(total)


def test_usage_without_response_model_priced_with_configured_model() -> None:
    calls = [1, 3]
    seen: list[list[_Resp]] = [[] for _ in calls]

    def fn(text: str, **_cfg: Any) -> str:
        i = int(text)
        for c in range(calls[i]):
            r = _Resp(*_sizes(i, c), model=None)
            seen[i].append(r)
            capture_langchain_response(r)
        return "ok"

    res = _evaluate(fn, 2)
    for i, er in enumerate(res.example_results):
        assert er.metrics["total_cost"] == pytest.approx(sum(_cost(r) for r in seen[i]))
        assert er.metrics["total_cost"] > 0


def test_unowned_response_is_unmeasured_with_exact_floor() -> None:
    from traigent.utils import langchain_interceptor as li

    owned: list[_Resp] = []
    stray: list[_Resp] = []

    def fn(text: str, **_cfg: Any) -> str:
        i = int(text)
        r = _Resp(*_sizes(i, 0))
        owned.append(r)
        capture_langchain_response(r)
        if i == 1:
            token = li._capture_owner.set(None)  # a call with no example owner
            try:
                s = _Resp(2000, 700)
                stray.append(s)
                capture_langchain_response(s)
            finally:
                li._capture_owner.reset(token)
        return "ok"

    res = _evaluate(fn, 2)
    m = res.aggregated_metrics
    total = sum(_cost(r) for r in owned + stray)
    assert m["cost_unmeasured"] == 1.0
    assert m.get("cost") is None
    assert m["cost_lower_bound"] == pytest.approx(total)
    for i, er in enumerate(res.example_results):
        assert er.metrics["total_cost"] == pytest.approx(_cost(owned[i]))
