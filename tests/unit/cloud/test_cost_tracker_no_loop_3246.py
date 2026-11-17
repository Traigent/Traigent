"""CostTracker sync track_* methods must record once and not need a loop (#3246)."""

from __future__ import annotations

import asyncio
import json

import pytest

from traigent.cloud.billing import CostCategory, CostTracker, CostTrackingConfig

pytestmark = pytest.mark.backend_online


def _tracker(tmp_path) -> CostTracker:
    return CostTracker(
        CostTrackingConfig(
            cache_costs_locally=True,
            cost_cache_file=str(tmp_path / "cache.json"),
            enable_server_sync=False,
        )
    )


def _calls(tracker: CostTracker):
    return [
        lambda: tracker.track_optimization_cost("s", tokens_used=100),
        lambda: tracker.track_inference_cost("s", 10, 10, "m"),
        lambda: tracker.track_data_transfer_cost("s", 1.0),
        lambda: tracker.track_custom_cost("s", CostCategory.STORAGE, "x", 0.5),
    ]


@pytest.mark.parametrize("idx", range(4))
def test_track_outside_loop_records_exactly_once(tmp_path, idx: int) -> None:
    tracker = _tracker(tmp_path)
    item = _calls(tracker)[idx]()
    assert tracker._total_cost == pytest.approx(item.total_cost)
    assert len(tracker._cost_items) == 1
    cached = json.loads((tmp_path / "cache.json").read_text())
    assert [c["item_id"] for c in cached] == [item.item_id]


@pytest.mark.parametrize("idx", range(4))
def test_track_inside_loop_records_exactly_once(tmp_path, idx: int) -> None:
    tracker = _tracker(tmp_path)

    async def run():
        item = _calls(tracker)[idx]()
        await asyncio.gather(*list(tracker._background_tasks))
        return item

    item = asyncio.run(run())
    assert tracker._total_cost == pytest.approx(item.total_cost)
    assert len(tracker._cost_items) == 1


def test_cache_failure_does_not_raise_or_double_count(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    tracker = CostTracker(
        CostTrackingConfig(
            cost_cache_file=str(blocker / "sub" / "c.json"), enable_server_sync=False
        )
    )
    tracker.track_custom_cost("s", CostCategory.STORAGE, "x", 0.5)
    assert tracker._total_cost == pytest.approx(0.5)
    assert len(tracker._cost_items) == 1


def test_task_scheduling_failure_does_not_raise_or_double_count(tmp_path) -> None:
    tracker = _tracker(tmp_path)

    async def run():
        loop = asyncio.get_running_loop()

        def boom(_loop, _coro):
            _coro.close()
            raise RuntimeError("cannot schedule")

        loop.set_task_factory(boom)
        try:
            tracker.track_custom_cost("s", CostCategory.STORAGE, "x", 0.5)
            tracker.track_custom_cost("s", CostCategory.STORAGE, "x", 0.5)
        finally:
            loop.set_task_factory(None)

    asyncio.run(run())
    assert tracker._total_cost == pytest.approx(1.0)
    assert len(tracker._cost_items) == 2
