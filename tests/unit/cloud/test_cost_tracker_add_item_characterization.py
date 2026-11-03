"""Characterization of CostTracker._add_cost_item (bounded cost bookkeeping)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.cloud import billing as billing_module
from traigent.cloud.billing import (
    CostCategory,
    CostItem,
    CostTracker,
    CostTrackingConfig,
)


def _item(item_id, session=None, ts=0.0, total=1.0):
    return CostItem(
        item_id=item_id,
        category=CostCategory.COMPUTE,
        description="d",
        quantity=1.0,
        unit_cost=total,
        total_cost=total,
        timestamp=ts,
        session_id=session,
    )


class _Logger:
    def __init__(self, trace):
        self._trace = trace

    def debug(self, msg, *args, **kwargs):
        self._trace.append(("debug", msg))

    def info(self, msg, *args, **kwargs):
        self._trace.append(("info", msg))

    def warning(self, msg, *args, **kwargs):
        self._trace.append(("warning", msg))

    def error(self, msg, *args, **kwargs):
        self._trace.append(("error", msg))


@pytest.fixture
def env(monkeypatch):
    trace: list = []
    monkeypatch.setattr(billing_module, "logger", _Logger(trace))

    def make(**config):
        config.setdefault("enable_server_sync", False)
        config.setdefault("cache_costs_locally", False)
        config.setdefault("enable_real_time_alerts", False)
        tracker = CostTracker(CostTrackingConfig(**config))
        monkeypatch.setattr(
            tracker, "_check_cost_alerts", lambda: trace.append(("alerts",))
        )
        monkeypatch.setattr(
            tracker,
            "_cache_cost_item",
            lambda item: (
                trace.append(("cache", item.item_id)) or ("coro", item.item_id)
            ),
        )
        monkeypatch.setattr(
            tracker,
            "_register_background_task",
            lambda task: trace.append(("register", task)),
        )
        return tracker

    def fake_create_task(coro):
        trace.append(("create_task", coro))
        return ("task", coro)

    monkeypatch.setattr(billing_module.asyncio, "create_task", fake_create_task)
    return SimpleNamespace(make=make, trace=trace)


def _ids(items):
    return [i.item_id for i in items]


class TestGlobalBookkeeping:
    def test_item_totals_and_stats(self, env):
        tracker = env.make()
        tracker._add_cost_item(_item("a", total=1.5))
        tracker._add_cost_item(_item("b", total=2.25))
        assert _ids(tracker._cost_items) == ["a", "b"]
        assert tracker._total_cost == 3.75
        assert tracker._stats["total_items_tracked"] == 2
        assert tracker._session_costs == {}
        assert tracker._pending_sync_items == []
        assert env.trace == []

    def test_one_below_limit_does_not_prune(self, env):
        tracker = env.make()
        tracker._max_cost_items = 10
        for i in range(9):
            tracker._add_cost_item(_item(f"i{i}"))
        assert len(tracker._cost_items) == 9
        assert env.trace == []

    @pytest.mark.parametrize(
        ("limit", "removed"), [(1, 1), (5, 1), (10, 1), (19, 1), (20, 2), (100, 10)]
    )
    def test_full_list_drops_oldest_tenth_with_floor_of_one(self, env, limit, removed):
        tracker = env.make()
        tracker._max_cost_items = limit
        for i in range(limit):
            tracker._add_cost_item(_item(f"i{i}"))
        env.trace.clear()
        tracker._add_cost_item(_item("new"))
        expected = [f"i{i}" for i in range(removed, limit)] + ["new"]
        assert _ids(tracker._cost_items) == expected
        assert env.trace == [
            (
                "debug",
                f"Pruned {removed} old cost items to stay within memory limit",
            )
        ]

    def test_pruning_does_not_reduce_totals(self, env):
        tracker = env.make()
        tracker._max_cost_items = 2
        for i in range(5):
            tracker._add_cost_item(_item(f"i{i}", total=1.0))
        assert tracker._total_cost == 5.0
        assert tracker._stats["total_items_tracked"] == 5


class TestSessionBookkeeping:
    @pytest.mark.parametrize("session", [None, ""])
    def test_items_without_session_are_not_grouped(self, env, session):
        tracker = env.make()
        tracker._add_cost_item(_item("a", session=session))
        assert tracker._session_costs == {}

    def test_new_and_existing_sessions_collect_items(self, env):
        tracker = env.make()
        tracker._add_cost_item(_item("a", "s1"))
        tracker._add_cost_item(_item("b", "s2"))
        tracker._add_cost_item(_item("c", "s1"))
        assert {k: _ids(v) for k, v in tracker._session_costs.items()} == {
            "s1": ["a", "c"],
            "s2": ["b"],
        }

    def test_session_limit_boundary_allows_filling_exactly(self, env):
        tracker = env.make()
        tracker._max_sessions = 2
        tracker._add_cost_item(_item("a", "s1", ts=5.0))
        tracker._add_cost_item(_item("b", "s2", ts=1.0))
        assert sorted(tracker._session_costs) == ["s1", "s2"]
        assert env.trace == []

    def test_full_sessions_evict_the_one_with_oldest_first_item(self, env):
        tracker = env.make()
        tracker._max_sessions = 2
        tracker._add_cost_item(_item("a", "s1", ts=5.0))
        tracker._add_cost_item(_item("b", "s2", ts=1.0))
        tracker._add_cost_item(_item("c", "s3", ts=9.0))
        assert {k: _ids(v) for k, v in tracker._session_costs.items()} == {
            "s1": ["a"],
            "s3": ["c"],
        }
        assert env.trace == [
            ("debug", "Removed oldest session s2 to stay within session limit")
        ]
        # the evicted session's items remain in the global list
        assert _ids(tracker._cost_items) == ["a", "b", "c"]

    def test_oldest_is_judged_by_first_item_not_latest(self, env):
        tracker = env.make()
        tracker._max_sessions = 2
        tracker._add_cost_item(_item("a", "s1", ts=1.0))
        tracker._add_cost_item(_item("b", "s2", ts=2.0))
        tracker._add_cost_item(_item("c", "s2", ts=100.0))
        tracker._add_cost_item(_item("d", "s3", ts=50.0))
        assert sorted(tracker._session_costs) == ["s2", "s3"]

    def test_empty_session_list_never_counts_as_oldest(self, env):
        tracker = env.make()
        tracker._max_sessions = 2
        tracker._session_costs["empty"] = []
        tracker._add_cost_item(_item("a", "s1", ts=5.0))
        tracker._add_cost_item(_item("b", "s2", ts=6.0))
        assert sorted(tracker._session_costs) == ["empty", "s2"]

    def test_equal_timestamps_evict_the_first_inserted(self, env):
        tracker = env.make()
        tracker._max_sessions = 2
        tracker._add_cost_item(_item("a", "s1", ts=3.0))
        tracker._add_cost_item(_item("b", "s2", ts=3.0))
        tracker._add_cost_item(_item("c", "s3", ts=3.0))
        assert sorted(tracker._session_costs) == ["s2", "s3"]

    def test_existing_session_never_triggers_eviction(self, env):
        tracker = env.make()
        tracker._max_sessions = 2
        tracker._add_cost_item(_item("a", "s1", ts=1.0))
        tracker._add_cost_item(_item("b", "s2", ts=2.0))
        tracker._add_cost_item(_item("c", "s1", ts=3.0))
        assert sorted(tracker._session_costs) == ["s1", "s2"]
        assert env.trace == []

    @pytest.mark.parametrize(
        ("limit", "removed"), [(1, 1), (9, 1), (10, 1), (20, 2), (50, 5)]
    )
    def test_full_session_drops_its_oldest_tenth_with_floor_of_one(
        self, env, limit, removed
    ):
        tracker = env.make()
        tracker._max_items_per_session = limit
        for i in range(limit):
            tracker._add_cost_item(_item(f"i{i}", "s"))
        tracker._add_cost_item(_item("new", "s"))
        expected = [f"i{i}" for i in range(removed, limit)] + ["new"]
        assert _ids(tracker._session_costs["s"]) == expected
        assert len(tracker._cost_items) == limit + 1
        assert env.trace == []

    def test_session_one_below_limit_keeps_everything(self, env):
        tracker = env.make()
        tracker._max_items_per_session = 5
        for i in range(4):
            tracker._add_cost_item(_item(f"i{i}", "s"))
        assert _ids(tracker._session_costs["s"]) == ["i0", "i1", "i2", "i3"]

    def test_session_pruning_runs_after_session_creation_and_before_append(self, env):
        tracker = env.make()
        tracker._max_items_per_session = 1
        tracker._add_cost_item(_item("a", "s"))
        tracker._add_cost_item(_item("b", "s"))
        assert _ids(tracker._session_costs["s"]) == ["b"]


class TestOptionalSteps:
    def test_sync_queue_only_when_enabled(self, env):
        off = env.make(enable_server_sync=False)
        off._add_cost_item(_item("a"))
        assert off._pending_sync_items == []
        on = env.make(enable_server_sync=True)
        item = _item("b")
        on._add_cost_item(item)
        assert on._pending_sync_items == [item]

    def test_cache_task_is_created_and_registered(self, env):
        tracker = env.make(cache_costs_locally=True)
        tracker._add_cost_item(_item("a"))
        assert env.trace == [
            ("cache", "a"),
            ("create_task", ("coro", "a")),
            ("register", ("task", ("coro", "a"))),
        ]

    def test_no_cache_task_when_disabled(self, env):
        tracker = env.make(cache_costs_locally=False)
        tracker._add_cost_item(_item("a"))
        assert env.trace == []

    def test_alerts_checked_only_when_enabled(self, env):
        env.make(enable_real_time_alerts=False)._add_cost_item(_item("a"))
        assert env.trace == []
        env.make(enable_real_time_alerts=True)._add_cost_item(_item("a"))
        assert env.trace == [("alerts",)]

    def test_full_order_of_steps(self, env):
        tracker = env.make(
            enable_server_sync=True,
            cache_costs_locally=True,
            enable_real_time_alerts=True,
        )
        tracker._max_cost_items = 1
        tracker._add_cost_item(_item("a", "s"))
        env.trace.clear()
        tracker._add_cost_item(_item("b", "s"))
        assert [e[0] for e in env.trace] == [
            "debug",
            "cache",
            "create_task",
            "register",
            "alerts",
        ]
        assert len(tracker._pending_sync_items) == 2


class TestFailurePaths:
    def test_zero_session_limit_fails_after_global_bookkeeping(self, env):
        tracker = env.make(enable_server_sync=True)
        tracker._max_sessions = 0
        with pytest.raises(ValueError):
            tracker._add_cost_item(_item("a", "s", total=2.0))
        assert _ids(tracker._cost_items) == ["a"]
        assert tracker._total_cost == 2.0
        assert tracker._stats["total_items_tracked"] == 1
        assert tracker._session_costs == {}
        assert tracker._pending_sync_items == []

    def test_alert_error_propagates_after_everything_else(self, env):
        tracker = env.make(enable_real_time_alerts=True, enable_server_sync=True)

        def boom():
            raise RuntimeError("alerts")

        tracker._check_cost_alerts = boom
        with pytest.raises(RuntimeError, match="alerts"):
            tracker._add_cost_item(_item("a", "s"))
        assert _ids(tracker._session_costs["s"]) == ["a"]
        assert len(tracker._pending_sync_items) == 1

    def test_task_creation_error_propagates_before_alerts(self, env, monkeypatch):
        tracker = env.make(cache_costs_locally=True, enable_real_time_alerts=True)

        def boom(coro):
            raise RuntimeError("no loop")

        monkeypatch.setattr(billing_module.asyncio, "create_task", boom)
        with pytest.raises(RuntimeError, match="no loop"):
            tracker._add_cost_item(_item("a"))
        assert ("alerts",) not in env.trace
        assert _ids(tracker._cost_items) == ["a"]


class _FlakySessionItem(CostItem):
    """CostItem whose ``session_id`` accessor serves two reads, then raises."""

    reads = 0

    @property
    def session_id(self):
        type(self).reads += 1
        if type(self).reads > 2:
            raise RuntimeError("session_id read")
        return "s"

    @session_id.setter
    def session_id(self, value):
        pass


class TestAccessorReads:
    def test_third_session_id_read_error_propagates_before_grouping(self, env):
        _FlakySessionItem.reads = 0
        tracker = env.make(enable_server_sync=True)
        item = _FlakySessionItem(
            item_id="a",
            category=CostCategory.COMPUTE,
            description="d",
            quantity=1.0,
            unit_cost=2.0,
            total_cost=2.0,
        )
        _FlakySessionItem.reads = 0
        with pytest.raises(RuntimeError, match="session_id read"):
            tracker._add_cost_item(item)
        assert _FlakySessionItem.reads == 3
        assert _ids(tracker._cost_items) == ["a"]
        assert tracker._total_cost == 2.0
        assert tracker._stats["total_items_tracked"] == 1
        assert tracker._session_costs == {}
        assert tracker._pending_sync_items == []


class TestUnorderedSessionLimit:
    def test_nan_session_limit_never_evicts_and_accepts_empty_storage(self, env):
        tracker = env.make()
        tracker._max_sessions = float("nan")
        tracker._add_cost_item(_item("a", "s1", ts=1.0))
        assert {k: _ids(v) for k, v in tracker._session_costs.items()} == {"s1": ["a"]}
        assert env.trace == []

    def test_nan_session_limit_with_existing_sessions_does_not_evict(self, env):
        tracker = env.make()
        tracker._max_sessions = float("nan")
        tracker._add_cost_item(_item("a", "s1", ts=1.0))
        tracker._add_cost_item(_item("b", "s2", ts=2.0))
        assert sorted(tracker._session_costs) == ["s1", "s2"]
        assert env.trace == []
