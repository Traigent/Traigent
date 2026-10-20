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


def test_sole_modelless_call_priced_with_configured_model_siblings_unknown() -> None:
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
    # The example's only call is established as the agent call: configured model.
    assert res.example_results[0].metrics["total_cost"] == pytest.approx(
        _cost(seen[0][0])
    )
    # Three model-less calls: none is established as the agent's, so the agent
    # model is not assumed for them; the row's cost is unknown, never a $0.
    m = res.aggregated_metrics
    assert m["cost_unpriced"] == 1.0
    assert m["cost_unmeasured"] == 1.0
    assert m.get("cost") is None


def test_raw_response_consumes_only_the_matching_call_siblings_still_count() -> None:
    seen: list[_Resp] = []

    def fn(text: str, **_cfg: Any) -> Any:
        agent, judge = _Resp(1000, 500), _Resp(4000, 800, model="gpt-4o")
        seen.extend([agent, judge])
        capture_langchain_response(agent)
        capture_langchain_response(judge)
        return {"text": "ok", "raw_response": agent}

    res = _evaluate(fn, 1, detailed=False)
    total = _cost(seen[0]) + _cost(seen[1], "gpt-4o")
    assert res.aggregated_metrics["cost"] == pytest.approx(total)


def test_agent_and_judge_calls_with_different_models_priced_separately() -> None:
    agent, judge = _Resp(1000, 500), _Resp(4000, 800, model="gpt-4o")

    def fn(text: str, **_cfg: Any) -> str:
        capture_langchain_response(agent)
        capture_langchain_response(judge)
        return "ok"

    res = _evaluate(fn, 1)
    expected = _cost(agent) + _cost(judge, "gpt-4o")
    assert _cost(judge, "gpt-4o") != pytest.approx(_cost(judge))
    assert res.example_results[0].metrics["total_cost"] == pytest.approx(expected)


def test_call_without_usage_stays_unknown_not_merged_into_measured_total() -> None:
    class _NoUsage:
        content = "ok"
        response_metadata = {"model_name": MODEL}

    measured = _Resp(1000, 500)

    def fn(text: str, **_cfg: Any) -> str:
        capture_langchain_response(measured)
        capture_langchain_response(_NoUsage())
        return "ok"

    res = _evaluate(fn, 1)
    m = res.aggregated_metrics
    assert m["cost_unmeasured"] == 1.0
    assert m.get("cost") is None
    assert m["cost_lower_bound"] == pytest.approx(_cost(measured))


def test_unowned_judge_priced_by_its_own_model_not_the_agent_model() -> None:
    from traigent.utils import langchain_interceptor as li

    agent = _Resp(1000, 500)
    judge = _Resp(4000, 800, model="gpt-4o")

    def fn(text: str, **_cfg: Any) -> str:
        capture_langchain_response(agent)
        token = li._capture_owner.set(None)
        try:
            capture_langchain_response(judge)
        finally:
            li._capture_owner.reset(token)
        return "ok"

    res = _evaluate(fn, 1)
    m = res.aggregated_metrics
    assert m["cost_unmeasured"] == 1.0
    expected = _cost(agent) + _cost(judge, "gpt-4o")
    assert _cost(judge, "gpt-4o") != pytest.approx(_cost(judge))
    assert m["cost_lower_bound"] == pytest.approx(expected)


@pytest.mark.parametrize("via_raw_response", [False, True])
def test_unowned_capture_also_returned_is_counted_once(via_raw_response: bool) -> None:
    from traigent.utils import langchain_interceptor as li

    resp = _Resp(1000, 500)

    def fn(text: str, **_cfg: Any) -> Any:
        token = li._capture_owner.set(None)
        try:
            capture_langchain_response(resp)
        finally:
            li._capture_owner.reset(token)
        return {"text": "ok", "raw_response": resp} if via_raw_response else resp

    res = _evaluate(fn, 1, detailed=False)
    m = res.aggregated_metrics
    assert not m.get("cost_unmeasured")
    assert m["cost"] == pytest.approx(_cost(resp))


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
