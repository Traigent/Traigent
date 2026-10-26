"""Golden characterization of HybridAPIEvaluator._evaluate_outputs."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.evaluators.hybrid_api import HybridAPIEvaluator

GOLDEN = Path(__file__).parent / "golden" / "hybrid_evaluate_outputs.json"


def _exec(outputs, ops=None, reported=None, execution_id="x1"):
    return SimpleNamespace(
        outputs=outputs,
        execution_id=execution_id,
        operational_metrics={"latency_ms": 40.0} if ops is None else ops,
        get_reported_total_cost=lambda: reported,
    )


def _inputs(*ids):
    return [{"example_id": i, "data": {"q": i}} for i in ids]


def _batch(*expected):
    return [{"input": {"q": n}, "expected_output": e} for n, e in enumerate(expected)]


def _ev(eval_results=None, eval_exc=None):
    transport = MagicMock()
    ev = HybridAPIEvaluator(transport=transport, tunable_id="cap", keep_alive=False)
    if eval_exc is not None:
        transport.evaluate = AsyncMock(side_effect=eval_exc)
    else:
        transport.evaluate = AsyncMock(
            return_value=SimpleNamespace(
                results=eval_results, aggregate_metrics={"accuracy": 1.0}
            )
        )
    return ev, transport


def _summ(results):
    return [dataclasses.asdict(r) for r in results]


# name -> (outputs, eval_results, expected list, input ids, exec kwargs, eval_exc)
SCENARIOS = {
    "basic": (
        [
            {"example_id": "a", "output": "o1", "cost_usd": 0.5, "latency_ms": 11.0},
            {"example_id": "b", "output": "o2"},
        ],
        [
            {"example_id": "a", "metrics": {"accuracy": 1.0, "f": True, "s": "x"}},
            {
                "input_id": "b",
                "metrics": {"overall_accuracy": 0.5, "cost_unmeasured": 1},
            },
        ],
        ["ea", "eb"],
        ["a", "b"],
        dict(reported=0.2),
        None,
    ),
    "output_id_only_and_missing_output": (
        [{"input_id": "a", "output_id": "oid1"}],
        [{"example_id": "a", "metrics": {"accuracy": 0.3}}],
        ["ea", "eb"],
        ["a", "b"],
        dict(reported=None),
        None,
    ),
    "eval_error_and_expected_behavior": (
        [
            {"example_id": "a", "output": "o", "error": "exec-fail"},
            {"example_id": "b", "output": "o", "error": {"not": "str"}},
        ],
        [
            {
                "example_id": "a",
                "metrics": {"accuracy": 0.0},
                "error": "eval-fail",
                "expected_behavior": "should say hi",
            },
            {
                "example_id": "b",
                "metrics": {"accuracy": 1.0},
                "error": 5,
                "expected_behavior": "ignored when expected present",
            },
        ],
        [None, "given"],
        ["a", "b"],
        dict(reported=0.0),
        None,
    ),
    "exec_error_only": (
        [{"example_id": "a", "output": None, "error": "boom"}],
        [],
        ["ea"],
        ["a"],
        dict(reported=0.1),
        None,
    ),
    "metrics_none_and_missing_eval_row": (
        [{"example_id": "a", "output": "o"}, {"example_id": "b", "output": "o"}],
        [{"example_id": "a", "metrics": None}],
        ["ea", "eb"],
        ["a", "b"],
        dict(reported=0.1),
        None,
    ),
    "negative_item_cost": (
        [{"example_id": "a", "output": "o", "cost_usd": -1.0}],
        [{"example_id": "a", "metrics": {"accuracy": 1.0}}],
        ["ea"],
        ["a"],
        dict(reported=0.1),
        None,
    ),
    "no_matching_outputs_falls_back": (
        [{"example_id": "zzz", "output": "o"}],
        [],
        ["ea"],
        ["a"],
        dict(reported=0.1),
        None,
    ),
    "transport_error_falls_back": (
        [{"example_id": "a", "output": "o"}],
        None,
        ["ea"],
        ["a"],
        dict(reported=0.1),
        RuntimeError("down"),
    ),
    "malformed_eval_results_fall_back": (
        [{"example_id": "a", "output": "o"}],
        [7],
        ["ea"],
        ["a"],
        dict(reported=0.1),
        None,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_matches_golden(name):
    outputs, eval_results, expected, ids, exec_kw, exc = SCENARIOS[name]
    ev, transport = _ev(eval_results, exc)
    results = await ev._evaluate_outputs(
        transport, {}, _batch(*expected), _inputs(*ids), _exec(outputs, **exec_kw)
    )
    request = None
    if transport.evaluate.await_args is not None:
        request = transport.evaluate.await_args.args[0].evaluations
    got = json.loads(
        json.dumps({"results": _summ(results), "request": request}, default=str)
    )
    data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if os.environ.get("REGEN_GOLDEN") == "1":
        data[name] = got
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True))
        return
    assert got == data[name]


@pytest.mark.asyncio
async def test_cancelled_error_propagates():
    ev, transport = _ev(eval_exc=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await ev._evaluate_outputs(
            transport,
            {},
            _batch("e"),
            _inputs("a"),
            _exec([{"example_id": "a", "output": "o"}]),
        )


@pytest.mark.asyncio
async def test_odd_none_operational_metrics_raises_in_fallback():
    # Inside the try the merge fails (None.get), then the fallback fails the same way.
    ev, transport = _ev([])
    resp = _exec([{"example_id": "a", "output": "o"}])
    resp.operational_metrics = None
    with pytest.raises(AttributeError):
        await ev._evaluate_outputs(transport, {}, _batch("e"), _inputs("a"), resp)


@pytest.mark.asyncio
async def test_odd_none_results_list_falls_back_to_execute_only():
    ev, transport = _ev(None)
    results = await ev._evaluate_outputs(
        transport,
        {},
        _batch("e"),
        _inputs("a"),
        _exec([{"example_id": "a", "output": "o"}], reported=0.1),
    )
    assert [r.metadata["evaluation_mode"] for r in results] == ["evaluate_fallback"]
