"""#2904: the wrapper unwraps a single-key {"value": x} before the handler runs."""

from __future__ import annotations

import pytest

from traigent.wrapper.service import TraigentService


async def _run(items):
    svc = TraigentService()
    seen: list[tuple[object, object]] = []

    @svc.evaluate
    def score(output, target, config):
        seen.append((output, target))
        return {"accuracy": 1.0}

    resp = await svc.handle_evaluate({"config": {}, "evaluations": items})
    assert resp["status"] == "completed"
    return seen


@pytest.mark.asyncio
async def test_wrapped_scalars_reach_handler_unwrapped():
    seen = await _run(
        [{"example_id": "e", "output": {"value": 42}, "target": {"value": "pos"}}]
    )
    assert seen == [(42, "pos")]


@pytest.mark.asyncio
async def test_wrapped_null_target_unwraps_to_none():
    seen = await _run(
        [{"example_id": "e", "output": {"value": 1}, "target": {"value": None}}]
    )
    assert seen == [(1, None)]


@pytest.mark.asyncio
async def test_genuine_objects_pass_through_unchanged():
    obj = {"value": 1, "other": 2}
    seen = await _run([{"example_id": "e", "output": obj, "target": obj}])
    assert seen == [(obj, obj)]


@pytest.mark.asyncio
async def test_other_single_key_object_unchanged():
    seen = await _run(
        [{"example_id": "e", "output": {"label": 1}, "target": {"label": 1}}]
    )
    assert seen == [({"label": 1}, {"label": 1})]
