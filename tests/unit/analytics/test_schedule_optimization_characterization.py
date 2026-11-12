"""Golden characterization for ``SmartScheduler.schedule_optimization``.

Pins request conversion, defaults, packing, conflicts, duration and the
original-order cost tail so internal restructuring stays behaviour-preserving.
"""

from __future__ import annotations

import copy
from datetime import datetime
from types import SimpleNamespace

import pytest

from traigent.analytics import scheduling
from traigent.analytics.scheduling import SmartScheduler

_RESULT_KEYS = [
    "schedule",
    "total_duration",
    "total_duration_hours",
    "conflicts",
    "utilization",
    "optimization_window",
    "scheduled_start_time",
    "estimated_duration",
    "estimated_cost",
    "priority_score",
    "resource_allocation",
]
_SCHEDULED_KEYS = [
    "task_id",
    "name",
    "start_time",
    "end_time",
    "duration_minutes",
    "priority",
    "status",
]
_WIDE = {"max_concurrent": 100}


def _window(start=0, days=7):
    return {"start": start, "duration_days": days}


def _run(tasks, window=None, constraints=None, **kwargs):
    return SmartScheduler().schedule_optimization(
        tasks=tasks,
        optimization_window=window if window is not None else _window(),
        constraints=constraints,
        **kwargs,
    )


def _task(task_id, duration, priority, **extra):
    return {
        "id": task_id,
        "estimated_duration": duration,
        "priority": priority,
        **extra,
    }


def _install_clock(monkeypatch, value=5000.0):
    calls = []

    def fake_time():
        calls.append(1)
        return value

    monkeypatch.setattr(scheduling, "time", SimpleNamespace(time=fake_time))
    return calls


# --------------------------------------------------------------------------
# Empty input, clock, request conversion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"tasks": None},
        {"tasks": []},
        {"tasks": [], "optimization_request": {}},
        {"tasks": None, "optimization_request": None},
    ],
)
def test_empty_input_returns_exact_dict_without_clock(monkeypatch, kwargs):
    calls = _install_clock(monkeypatch)
    out = SmartScheduler().schedule_optimization(**kwargs)
    assert out == {"schedule": [], "total_duration": 0, "conflicts": []}
    assert list(out) == ["schedule", "total_duration", "conflicts"]
    assert calls == []


def test_clock_called_once_only_when_window_missing(monkeypatch):
    calls = _install_clock(monkeypatch, 5000.0)
    tasks = [_task("a", 10, 1), _task("b", 10, 1), _task("c", 10, 1)]
    out = SmartScheduler().schedule_optimization(tasks=tasks)
    assert len(calls) == 1
    assert out["optimization_window"] == {"start": 5000.0, "duration_days": 7}
    assert [t["start_time"] for t in out["schedule"]] == [5000.0, 5000.0, 5600.0]


def test_clock_not_called_when_window_given(monkeypatch):
    calls = _install_clock(monkeypatch)
    _run([_task("a", 10, 1)])
    assert calls == []


def test_clock_called_once_even_when_later_step_raises(monkeypatch):
    calls = _install_clock(monkeypatch)
    with pytest.raises(KeyError) as exc:
        SmartScheduler().schedule_optimization(
            tasks=[_task("a", 10, 1)], constraints={}
        )
    assert exc.value.args == ("max_concurrent",)
    assert len(calls) == 1


def test_falsey_empty_window_is_not_replaced_by_default(monkeypatch):
    calls = _install_clock(monkeypatch)
    with pytest.raises(KeyError) as exc:
        SmartScheduler().schedule_optimization(
            tasks=[_task("a", 10, 1)], optimization_window={}
        )
    assert exc.value.args == ("start",)
    assert calls == []


def test_zero_start_window_is_used_as_is():
    out = _run([_task("a", 10, 1)], window=_window(0, 7))
    assert out["schedule"][0]["start_time"] == 0
    assert out["total_duration"] == 600


def test_request_conversion_full_fields():
    request = {
        "function_name": "fn",
        "estimated_duration": 7260,
        "priority": "high",
        "estimated_cost": 100.0,
    }
    out = SmartScheduler().schedule_optimization(
        optimization_request=request,
        optimization_window=_window(1000, 7),
    )
    assert out["schedule"] == [
        {
            "task_id": "optimization_task",
            "name": "fn",
            "start_time": 1000,
            "end_time": 1000 + 121 * 60,
            "duration_minutes": 121,
            "priority": 3,
            "status": "scheduled",
        }
    ]
    assert out["estimated_cost"] == 100.0 * 0.95
    assert out["priority_score"] == 3.0
    assert out["total_duration"] == 121 * 60


@pytest.mark.parametrize(
    ("priority", "expected"),
    [("high", 3), ("medium", 2), ("low", 1), ("urgent", 2), (None, 2)],
)
def test_request_priority_mapping(priority, expected):
    request = {"function_name": "x", "priority": priority}
    out = SmartScheduler().schedule_optimization(
        optimization_request=request, optimization_window=_window()
    )
    assert out["schedule"][0]["priority"] == expected


def test_request_defaults_when_keys_missing():
    out = SmartScheduler().schedule_optimization(
        optimization_request={"unrelated": 1}, optimization_window=_window(10, 7)
    )
    assert out["schedule"] == [
        {
            "task_id": "optimization_task",
            "name": "optimization",
            "start_time": 10,
            "end_time": 10 + 60 * 60,
            "duration_minutes": 60,
            "priority": 2,
            "status": "scheduled",
        }
    ]
    assert out["estimated_cost"] == 0.0 * 0.95
    assert out["total_duration"] == 3600


@pytest.mark.parametrize(
    ("seconds", "minutes"), [(59, 0), (60, 1), (90, 1), (3600.0, 60.0)]
)
def test_request_duration_floor_divided_to_minutes(seconds, minutes):
    request = {"function_name": "x", "estimated_duration": seconds}
    out = SmartScheduler().schedule_optimization(
        optimization_request=request, optimization_window=_window()
    )
    assert out["schedule"][0]["duration_minutes"] == minutes
    assert out["schedule"][0]["end_time"] == minutes * 60


def test_request_is_ignored_when_tasks_supplied():
    out = SmartScheduler().schedule_optimization(
        tasks=[_task("mine", 5, 1)],
        optimization_request={"function_name": "ignored"},
        optimization_window=_window(),
    )
    assert [t["task_id"] for t in out["schedule"]] == ["mine"]


def test_empty_tasks_list_falls_back_to_request():
    out = SmartScheduler().schedule_optimization(
        tasks=[],
        optimization_request={"function_name": "q"},
        optimization_window=_window(),
    )
    assert [t["name"] for t in out["schedule"]] == ["q"]


def test_request_cost_tail_uses_request_cost_and_preferences():
    request = {"function_name": "x", "estimated_cost": 20.0}
    high = SmartScheduler().schedule_optimization(
        optimization_request=request,
        cost_preferences={"cost_priority": "high"},
        optimization_window=_window(),
    )
    low = SmartScheduler().schedule_optimization(
        optimization_request=request,
        cost_preferences={"cost_priority": "low"},
        optimization_window=_window(),
    )
    assert high["estimated_cost"] == 20.0 * 0.85
    assert low["estimated_cost"] == 20.0 * 0.95


def test_request_get_call_order():
    log = []

    class Request(dict):
        def get(self, key, default=None):
            log.append((key, default))
            return super().get(key, default)

    SmartScheduler().schedule_optimization(
        optimization_request=Request({"function_name": "x"}),
        optimization_window=_window(),
    )
    assert log == [
        ("function_name", "optimization"),
        ("estimated_duration", 3600),
        ("priority", "medium"),
        ("estimated_cost", 0.0),
    ]


# --------------------------------------------------------------------------
# Ordering, defaults, fields
# --------------------------------------------------------------------------


def test_sort_by_priority_desc_then_duration_asc_with_stable_ties():
    tasks = [
        _task("a", 30, 1),
        _task("b", 50, 2),
        _task("c", 20, 2),
        _task("d", 20, 2),
        _task("e", 30, 1),
        _task("f", 90, 3),
    ]
    snapshot = copy.deepcopy(tasks)
    original_order = list(tasks)
    out = _run(tasks, constraints=_WIDE)
    assert [t["task_id"] for t in out["schedule"]] == ["f", "c", "d", "b", "a", "e"]
    assert tasks == snapshot
    assert all(x is y for x, y in zip(tasks, original_order, strict=True))


def test_missing_priority_and_duration_use_sort_defaults():
    tasks = [
        {"id": "dflt"},
        _task("p2", 5, 2),
        _task("short", 59, 1),
        _task("long", 61, 1),
    ]
    out = _run(tasks, constraints=_WIDE)
    assert [t["task_id"] for t in out["schedule"]] == ["p2", "short", "dflt", "long"]
    dflt = out["schedule"][2]
    assert dflt["duration_minutes"] == 60
    assert dflt["priority"] == 1
    assert dflt["end_time"] == 3600


def test_missing_id_and_name_defaults_follow_scheduled_position():
    tasks = [{"priority": 1, "name": "x"}, {"id": "named", "priority": 2}]
    out = _run(tasks, constraints=_WIDE)
    assert [t["task_id"] for t in out["schedule"]] == ["named", "task_1"]
    assert [t["name"] for t in out["schedule"]] == ["Unnamed task", "x"]


def test_falsey_present_ids_are_kept():
    tasks = [
        {"id": None, "priority": 3, "estimated_duration": 1},
        {"id": 0, "priority": 2, "estimated_duration": 1},
        {"id": "", "priority": 1, "estimated_duration": 1},
    ]
    out = _run(tasks, constraints=_WIDE)
    assert [t["task_id"] for t in out["schedule"]] == [None, 0, ""]


def test_scheduled_entry_exact_shape_and_values():
    out = _run(
        [_task("a", 1.5, 2, name="alpha")], window=_window(1000, 7), constraints=_WIDE
    )
    entry = out["schedule"][0]
    assert list(entry) == _SCHEDULED_KEYS
    assert entry == {
        "task_id": "a",
        "name": "alpha",
        "start_time": 1000,
        "end_time": 1000 + 1.5 * 60,
        "duration_minutes": 1.5,
        "priority": 2,
        "status": "scheduled",
    }


def test_result_key_order_and_derived_fields():
    tasks = [_task("a", 10, 3), _task("b", 20, 1), _task("c", 30, 2)]
    window = _window(1000, 7)
    out = _run(tasks, window=window, constraints=_WIDE)
    assert list(out) == _RESULT_KEYS
    assert out["schedule"][0]["start_time"] == 1000
    assert out["total_duration"] == 30 * 60
    assert out["total_duration_hours"] == (30 * 60) / 3600
    assert out["estimated_duration"] == out["total_duration"]
    assert out["conflicts"] == []
    assert out["utilization"] == 1.0
    assert out["optimization_window"] is window
    assert out["scheduled_start_time"] == datetime.fromtimestamp(1000).isoformat()
    assert out["priority_score"] == (3 + 1 + 2) / 3
    assert out["resource_allocation"] == {
        "cpu_cores": 2,
        "memory_gb": 4,
        "storage_gb": 100,
        "estimated_instances": 3,
    }


def test_extra_arguments_are_ignored():
    tasks = [_task("a", 10, 1), _task("b", 10, 1)]
    plain = _run(copy.deepcopy(tasks))
    extra = _run(
        copy.deepcopy(tasks),
        time_constraints={"deadline": "x"},
        unexpected=object(),
    )
    assert plain == extra


def test_fromtimestamp_called_once_with_first_scheduled_start(monkeypatch):
    seen = []

    class _Stamp:
        def isoformat(self):
            return "stamp"

    def fake_fromtimestamp(value):
        seen.append(value)
        return _Stamp()

    def fail_now(*_a, **_k):
        raise AssertionError("datetime.now must not be used")

    monkeypatch.setattr(
        scheduling,
        "datetime",
        SimpleNamespace(fromtimestamp=fake_fromtimestamp, now=fail_now),
    )
    out = _run([_task("a", 10, 1), _task("b", 10, 2)], window=_window(777, 7))
    assert seen == [777]
    assert out["scheduled_start_time"] == "stamp"


# --------------------------------------------------------------------------
# Cost tail keeps original-order semantics
# --------------------------------------------------------------------------


def test_cost_uses_first_input_task_not_first_sorted_task():
    tasks = [
        _task("cheap-low", 60, 1, estimated_cost=10.0),
        _task("pricey-high", 10, 3, estimated_cost=1000.0),
    ]
    out = _run(tasks)
    assert out["schedule"][0]["task_id"] == "pricey-high"
    assert out["estimated_cost"] == 10.0 * 0.95


def test_cost_preference_variants():
    tasks = [_task("a", 10, 1, estimated_cost=10.0)]
    high = _run(copy.deepcopy(tasks), cost_preferences={"cost_priority": "high"})
    low = _run(copy.deepcopy(tasks), cost_preferences={"cost_priority": "low"})
    empty = _run(copy.deepcopy(tasks), cost_preferences={})
    none = _run(copy.deepcopy(tasks), cost_preferences=None)
    assert high["estimated_cost"] == 10.0 * 0.85
    assert low["estimated_cost"] == 10.0 * 0.95
    assert empty["estimated_cost"] == 10.0 * 0.95
    assert none["estimated_cost"] == 10.0 * 0.95


def test_missing_first_task_cost_defaults_to_zero():
    out = _run([_task("a", 10, 1), _task("b", 5, 1, estimated_cost=50.0)])
    assert out["estimated_cost"] == 0.0 * 0.95


def test_priority_score_uses_default_for_missing_priority():
    out = _run([{"id": "a", "estimated_duration": 1}, _task("b", 1, 3)])
    assert out["priority_score"] == (1 + 3) / 2


# --------------------------------------------------------------------------
# Concurrency packing
# --------------------------------------------------------------------------


def _five_equal():
    return [_task(f"t{i}", 10, 1) for i in range(5)]


def _starts(out):
    return [t["start_time"] for t in out["schedule"]]


def test_default_constraints_allow_two_concurrent():
    out = _run(_five_equal())
    assert _starts(out) == [0, 0, 600, 600, 1200]
    assert out["total_duration"] == 1800


def test_max_concurrent_one_serializes():
    out = _run(_five_equal(), constraints={"max_concurrent": 1})
    assert _starts(out) == [0, 600, 1200, 1800, 2400]
    assert out["total_duration"] == 3000


def test_max_concurrent_two_explicit_matches_default():
    out = _run(_five_equal(), constraints={"max_concurrent": 2})
    assert _starts(out) == [0, 0, 600, 600, 1200]


def test_max_concurrent_three():
    out = _run(_five_equal(), constraints={"max_concurrent": 3})
    assert _starts(out) == [0, 0, 0, 600, 600]


def test_earliest_end_is_used_when_slots_are_full():
    tasks = [_task("a", 10, 1), _task("b", 20, 1), _task("c", 30, 1)]
    out = _run(tasks, constraints={"max_concurrent": 2})
    assert _starts(out) == [0, 0, 600]
    assert [t["end_time"] for t in out["schedule"]] == [600, 1200, 2400]
    assert out["total_duration"] == 2400


def test_zero_duration_tasks_do_not_occupy_a_slot():
    tasks = [_task("z1", 0, 1), _task("z2", 0, 1), _task("n", 10, 1)]
    out = _run(tasks, constraints={"max_concurrent": 1})
    assert _starts(out) == [0, 0, 0]
    assert out["total_duration"] == 600


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_max_concurrent_raises_min_error(limit):
    with pytest.raises(ValueError, match="empty"):
        _run([_task("a", 10, 1)], constraints={"max_concurrent": limit})


@pytest.mark.parametrize("constraints", [{}, {"business_hours_only": True}])
def test_missing_max_concurrent_raises_key_error(constraints):
    with pytest.raises(KeyError) as exc:
        _run([_task("a", 10, 1)], constraints=constraints)
    assert exc.value.args == ("max_concurrent",)


def test_other_constraint_keys_are_ignored():
    constraints = {
        "max_concurrent": 1,
        "business_hours_only": True,
        "maintenance_windows": [{"start": 0, "end": 99999}],
    }
    snapshot = copy.deepcopy(constraints)
    out = _run(_five_equal(), constraints=constraints)
    assert _starts(out) == [0, 600, 1200, 1800, 2400]
    assert constraints == snapshot


def test_max_concurrent_read_once_per_task():
    reads = []

    class Constraints(dict):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    _run(_five_equal()[:3], constraints=Constraints(_WIDE))
    assert reads == ["max_concurrent"] * 3


def test_invalid_max_concurrent_read_once_before_error():
    reads = []

    class Constraints(dict):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    with pytest.raises(ValueError, match="empty"):
        _run(_five_equal(), constraints=Constraints({"max_concurrent": 0}))
    assert reads == ["max_concurrent"]


def test_inputs_are_not_mutated():
    tasks = [_task("a", 10, 2), _task("b", 20, 1)]
    constraints = {"max_concurrent": 1}
    window = _window(100, 3)
    prefs = {"cost_priority": "high"}
    snapshots = [copy.deepcopy(x) for x in (tasks, constraints, window, prefs)]
    _run(tasks, window=window, constraints=constraints, cost_preferences=prefs)
    assert [tasks, constraints, window, prefs] == snapshots


# --------------------------------------------------------------------------
# Window conflicts
# --------------------------------------------------------------------------

_CONFLICT_DETAILS = "Task extends beyond the optimization window"


def test_window_end_exactly_on_bound_is_not_a_conflict():
    out = _run([_task("edge", 1440, 1)], window=_window(1000, 1))
    assert out["schedule"][0]["end_time"] == 1000 + 86400
    assert out["conflicts"] == []


def test_window_one_minute_over_is_a_conflict_with_exact_dict():
    out = _run([_task("over", 1441, 1)], window=_window(1000, 1))
    assert out["conflicts"] == [
        {
            "task_id": "over",
            "reason": "exceeds_optimization_window",
            "details": _CONFLICT_DETAILS,
        }
    ]
    assert list(out["conflicts"][0]) == ["task_id", "reason", "details"]


def test_conflicts_follow_scheduled_order_and_only_offenders():
    tasks = [_task("exact", 1440, 2), _task("over", 1441, 1), _task("big", 5000, 1)]
    out = _run(tasks, window=_window(1000, 1), constraints=_WIDE)
    assert [c["task_id"] for c in out["conflicts"]] == ["over", "big"]


def test_fractional_window_days():
    window = _window(0, 0.5)
    ok = _run([_task("ok", 720, 1)], window=window)
    over = _run([_task("over", 721, 1)], window=_window(0, 0.5))
    assert ok["conflicts"] == []
    assert [c["task_id"] for c in over["conflicts"]] == ["over"]


def test_zero_day_window():
    tasks = [_task("zero", 0, 1), _task("pos", 1, 1)]
    out = _run(tasks, window=_window(5, 0), constraints=_WIDE)
    assert [c["task_id"] for c in out["conflicts"]] == ["pos"]


def test_delay_from_concurrency_can_cause_conflict():
    tasks = [_task("A", 1000, 2), _task("B", 1000, 1)]
    out = _run(tasks, window=_window(0, 1), constraints={"max_concurrent": 1})
    assert _starts(out) == [0, 60000]
    assert [c["task_id"] for c in out["conflicts"]] == ["B"]


def test_conflict_uses_default_id_for_unnamed_task():
    out = _run([{"estimated_duration": 99999}], window=_window(0, 1))
    assert out["conflicts"][0]["task_id"] == "task_0"


# --------------------------------------------------------------------------
# Total duration
# --------------------------------------------------------------------------


def test_total_duration_is_max_end_minus_window_start():
    tasks = [_task("long", 100, 2), _task("short", 10, 1)]
    out = _run(tasks, window=_window(500, 7), constraints=_WIDE)
    assert out["total_duration"] == 6000
    assert out["total_duration_hours"] == 6000 / 3600
    assert out["estimated_duration"] == 6000


def test_total_duration_zero_for_zero_length_task():
    out = _run([_task("z", 0, 1)])
    assert out["total_duration"] == 0
    assert out["total_duration_hours"] == 0.0
    assert out["utilization"] == 1.0


def test_negative_duration_gives_negative_total():
    out = _run([_task("neg", -10, 1)], window=_window(1000, 7))
    assert out["schedule"][0]["end_time"] == 1000 - 600
    assert out["total_duration"] == -600
    assert out["conflicts"] == []


# --------------------------------------------------------------------------
# Errors, probes and access order
# --------------------------------------------------------------------------


def test_missing_window_start_raises_before_sorting_error():
    tasks = [{"priority": None}, {"priority": 1}]
    with pytest.raises(KeyError) as exc:
        SmartScheduler().schedule_optimization(
            tasks=tasks, optimization_window={"duration_days": 7}
        )
    assert exc.value.args == ("start",)


def test_unsortable_priorities_raise_type_error_with_valid_window():
    tasks = [{"priority": None}, {"priority": 1}]
    with pytest.raises(TypeError):
        _run(tasks)


def test_task_without_get_raises_attribute_error():
    with pytest.raises(AttributeError):
        _run([object()])


def test_missing_duration_days_raises_after_first_task_is_scheduled():
    with pytest.raises(KeyError) as exc:
        SmartScheduler().schedule_optimization(
            tasks=[_task("a", 10, 1)], optimization_window={"start": 0}
        )
    assert exc.value.args == ("duration_days",)


def test_generator_tasks_fail_on_cost_tail_indexing():
    with pytest.raises(TypeError):
        _run(_task(f"t{i}", 10, 1) for i in range(2))


def test_tuple_tasks_work():
    out = _run((_task("a", 10, 1), _task("b", 10, 1)))
    assert [t["task_id"] for t in out["schedule"]] == ["a", "b"]


def test_access_order_single_task():
    log = []

    class Task(dict):
        def get(self, key, default=None):
            log.append(("t", key, default))
            return super().get(key, default)

    class Window(dict):
        def __getitem__(self, key):
            log.append(("w", key))
            return super().__getitem__(key)

    class Constraints(dict):
        def __getitem__(self, key):
            log.append(("c", key))
            return super().__getitem__(key)

    out = SmartScheduler().schedule_optimization(
        tasks=[Task()],
        constraints=Constraints(_WIDE),
        optimization_window=Window(_window()),
    )
    assert len(out["schedule"]) == 1
    assert log == [
        ("w", "start"),
        ("t", "priority", 1),
        ("t", "estimated_duration", 60),
        ("t", "estimated_duration", 60),
        ("t", "priority", 1),
        ("c", "max_concurrent"),
        ("t", "id", "task_0"),
        ("t", "name", "Unnamed task"),
        ("w", "start"),
        ("w", "duration_days"),
        ("w", "start"),
        ("t", "estimated_cost", 0.0),
        ("t", "priority", 1),
    ]


def test_access_order_two_tasks_sort_then_schedule_then_tail():
    log = []

    class Task(dict):
        def get(self, key, default=None):
            log.append((self["id"], key))
            return super().get(key, default)

    first = Task(id="A", priority=1, estimated_duration=10)
    second = Task(id="B", priority=2, estimated_duration=10)
    SmartScheduler().schedule_optimization(
        tasks=[first, second],
        constraints=_WIDE,
        optimization_window=_window(),
    )
    assert log == [
        ("A", "priority"),
        ("A", "estimated_duration"),
        ("B", "priority"),
        ("B", "estimated_duration"),
        ("B", "estimated_duration"),
        ("B", "priority"),
        ("B", "id"),
        ("B", "name"),
        ("A", "estimated_duration"),
        ("A", "priority"),
        ("A", "id"),
        ("A", "name"),
        ("A", "estimated_cost"),
        ("A", "priority"),
        ("B", "priority"),
    ]


def test_window_read_counts_for_two_tasks():
    reads = []

    class Window(dict):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    SmartScheduler().schedule_optimization(
        tasks=[_task("a", 10, 1), _task("b", 10, 1)],
        constraints=_WIDE,
        optimization_window=Window(_window()),
    )
    assert reads == [
        "start",
        "start",
        "duration_days",
        "start",
        "duration_days",
        "start",
    ]
