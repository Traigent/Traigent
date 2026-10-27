"""Characterization of TraigentSpanProcessor.on_end (span intake decisions)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.observability.otel import processor as processor_module
from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.processor import TraigentSpanProcessor
from traigent.utils import optimization_only_egress as oo_egress


class _NullTransport:
    def post(self, body, *, timeout):  # pragma: no cover - never reached
        raise AssertionError("no export expected")


def _ctx(trace_id=1, span_id=2, sampled=True):
    return SimpleNamespace(
        trace_id=trace_id, span_id=span_id, trace_flags=SimpleNamespace(sampled=sampled)
    )


class _Span:
    """Span stand-in; every read of ``context`` is recorded in ``trace``."""

    def __init__(self, trace, ctx="default", name="abc", attrs=2, hook=None):
        self._trace = trace
        self._ctx = _ctx() if ctx == "default" else ctx
        self.name = name
        self.attributes = {f"k{i}": i for i in range(attrs)}
        self._hook = hook

    @property
    def context(self):
        self._trace.append("context")
        if self._hook is not None:
            return self._hook()
        return self._ctx


@pytest.fixture
def env(monkeypatch):
    trace: list = []
    state = {"offline": False, "oo": False}

    def fake_offline():
        trace.append("backend_offline")
        return state["offline"]

    def fake_oo(explicit=None, scope=None):
        trace.append(("oo_active", scope))
        return state["oo"]

    monkeypatch.setattr(processor_module, "is_backend_offline", fake_offline)
    monkeypatch.setattr(oo_egress, "is_active", fake_oo)

    def make(**kw):
        drops: list = []
        kw.setdefault("on_drop", lambda reason, count: drops.append((reason, count)))
        proc = TraigentSpanProcessor(
            TraigentOTLPExporter(_NullTransport(), export_timeout=1.0),
            exit_flush=False,
            **kw,
        )
        monkeypatch.setattr(
            proc,
            "_ensure_worker_locked",
            lambda: trace.append(("worker", len(proc._queue))),
        )
        return proc, drops

    return SimpleNamespace(trace=trace, state=state, make=make)


def _stats(proc):
    stats = proc.stats()
    stats.pop("queue_depth")
    stats.pop("in_flight")
    return {k: v for k, v in stats.items() if v}


class TestShutdown:
    def test_shutdown_returns_before_reading_anything(self, env):
        proc, drops = env.make()
        proc._shutdown = True
        proc.on_end(_Span(env.trace))
        assert env.trace == []
        assert proc.stats()["queued"] == 0
        assert drops == []


class TestOfflineDrops:
    def test_overflow_latch_drops_without_other_checks(self, env):
        proc, _ = env.make()
        proc._offline_overflow = True
        proc.on_end(_Span(env.trace))
        assert env.trace == ["context"]
        assert _stats(proc) == {"dropped_offline": 1}
        assert len(proc._queue) == 0

    def test_tracked_start_drops_once_and_is_forgotten(self, env):
        proc, _ = env.make()
        proc._started_offline.add((1, 2))
        proc.on_end(_Span(env.trace))
        assert env.trace == ["context"]
        assert proc._started_offline == set()
        assert _stats(proc) == {"dropped_offline": 1}
        env.trace.clear()
        proc.on_end(_Span(env.trace))
        assert _stats(proc) == {"dropped_offline": 1, "queued": 1}

    def test_only_the_matching_key_is_forgotten(self, env):
        proc, _ = env.make()
        proc._started_offline.update({(1, 2), (1, 3), (9, 2)})
        proc.on_end(_Span(env.trace, ctx=_ctx(1, 2)))
        assert proc._started_offline == {(1, 3), (9, 2)}

    def test_untracked_key_is_not_dropped(self, env):
        proc, _ = env.make()
        proc._started_offline.add((1, 3))
        proc.on_end(_Span(env.trace))
        assert proc._started_offline == {(1, 3)}
        assert _stats(proc) == {"queued": 1}

    def test_backend_offline_drops(self, env):
        proc, _ = env.make()
        env.state["offline"] = True
        proc.on_end(_Span(env.trace))
        assert env.trace == ["context", "backend_offline"]
        assert _stats(proc) == {"dropped_offline": 1}
        assert len(proc._queue) == 0

    def test_optimization_only_drops_and_receives_scope(self, env):
        proc, _ = env.make()
        proc.oo_scope = "scope-a"
        env.state["oo"] = True
        proc.on_end(_Span(env.trace))
        assert env.trace == ["context", "backend_offline", ("oo_active", "scope-a")]
        assert _stats(proc) == {"dropped_offline": 1}
        assert len(proc._queue) == 0

    def test_missing_scope_attribute_passes_none(self, env):
        proc, _ = env.make()
        assert not hasattr(proc, "oo_scope")
        proc.on_end(_Span(env.trace))
        assert ("oo_active", None) in env.trace

    def test_offline_drop_does_not_read_context_again(self, env):
        proc, drops = env.make()
        env.state["offline"] = True
        proc.on_end(_Span(env.trace))
        assert env.trace.count("context") == 1
        assert drops == []

    def test_overflow_flag_is_read_before_context(self, env):
        proc, _ = env.make()

        def hook():
            proc._offline_overflow = True
            return _ctx()

        proc.on_end(_Span(env.trace, hook=hook))
        assert _stats(proc) == {"queued": 1}

    def test_missing_first_context_still_checks_offline_sources(self, env):
        proc, _ = env.make()
        env.state["offline"] = True
        proc.on_end(_Span(env.trace, ctx=None))
        assert _stats(proc) == {"dropped_offline": 1}
        assert env.trace == ["context", "backend_offline"]


class TestSampling:
    def test_missing_context_is_sampled_out(self, env):
        proc, _ = env.make()
        proc.on_end(_Span(env.trace, ctx=None))
        assert _stats(proc) == {"sampled_out": 1}
        assert env.trace == [
            "context",
            "backend_offline",
            ("oo_active", None),
            "context",
        ]
        assert len(proc._queue) == 0

    def test_unsampled_context_is_sampled_out(self, env):
        proc, _ = env.make()
        proc.on_end(_Span(env.trace, ctx=_ctx(sampled=False)))
        assert _stats(proc) == {"sampled_out": 1}
        assert len(proc._queue) == 0

    def test_second_context_read_is_the_one_that_is_sampled(self, env):
        proc, _ = env.make()
        reads = iter([_ctx(), None])
        proc.on_end(_Span(env.trace, hook=lambda: next(reads)))
        assert _stats(proc) == {"sampled_out": 1}

    def test_first_read_error_propagates_before_any_check(self, env):
        proc, _ = env.make()

        def boom():
            raise RuntimeError("first read")

        with pytest.raises(RuntimeError, match="first read"):
            proc.on_end(_Span(env.trace, hook=boom))
        assert env.trace == ["context"]
        assert proc.stats()["dropped_offline"] == 0

    def test_second_read_error_propagates_after_offline_checks(self, env):
        proc, _ = env.make()
        reads = [0]

        def hook():
            reads[0] += 1
            if reads[0] == 2:
                raise RuntimeError("second read")
            return _ctx()

        with pytest.raises(RuntimeError, match="second read"):
            proc.on_end(_Span(env.trace, hook=hook))
        assert env.trace == [
            "context",
            "backend_offline",
            ("oo_active", None),
            "context",
        ]
        assert len(proc._queue) == 0
        assert _stats(proc) == {}


class TestQueueing:
    def test_queued_span_updates_state(self, env):
        times = iter([100.0, 200.0])
        proc, drops = env.make(clock=lambda: next(times))
        first = _Span(env.trace, name="abc", attrs=2)
        second = _Span(env.trace, name="abcd", attrs=0)
        proc.on_end(first)
        assert proc._first_queued_at == 100.0
        proc.on_end(second)
        assert proc._first_queued_at == 100.0
        assert list(proc._queue) == [first, second]
        assert proc._queued_bytes == (200 + 3 + 96) + (200 + 4)
        assert _stats(proc) == {"queued": 2}
        assert drops == []

    def test_first_queued_time_resets_when_queue_was_empty(self, env):
        times = iter([100.0, 200.0])
        proc, _ = env.make(clock=lambda: next(times))
        proc.on_end(_Span(env.trace))
        proc._queue.clear()
        proc.on_end(_Span(env.trace))
        assert proc._first_queued_at == 200.0

    def test_worker_is_ensured_after_append_for_each_span(self, env):
        proc, _ = env.make()
        proc.on_end(_Span(env.trace))
        proc.on_end(_Span(env.trace))
        assert [e for e in env.trace if isinstance(e, tuple) and e[0] == "worker"] == [
            ("worker", 1),
            ("worker", 2),
        ]

    def test_queue_limit_boundary(self, env):
        proc, drops = env.make(max_queue_spans=2)
        proc.on_end(_Span(env.trace))
        assert len(proc._queue) == 1
        proc.on_end(_Span(env.trace))
        assert len(proc._queue) == 2
        assert drops == []
        proc.on_end(_Span(env.trace))
        assert len(proc._queue) == 2
        assert _stats(proc) == {"queued": 2, "dropped_queue_full": 1}
        assert drops == [("queue_full", 1)]

    def test_full_queue_does_not_touch_worker_or_bytes(self, env):
        proc, _ = env.make(max_queue_spans=1)
        proc.on_end(_Span(env.trace))
        before = (proc._queued_bytes, proc._first_queued_at)
        env.trace.clear()
        proc.on_end(_Span(env.trace))
        assert (proc._queued_bytes, proc._first_queued_at) == before
        assert not [e for e in env.trace if isinstance(e, tuple) and e[0] == "worker"]

    def test_drop_callback_runs_after_the_lock_is_released(self, env):
        seen = []
        proc, _ = env.make(max_queue_spans=1)

        def probe(reason, count):
            got = proc._cond.acquire(blocking=False)
            if got:
                proc._cond.release()
            seen.append((reason, count, got))

        proc._on_drop = probe
        proc.on_end(_Span(env.trace))
        proc.on_end(_Span(env.trace))
        assert seen == [("queue_full", 1, True)]

    def test_drop_callback_error_is_contained(self, env):
        proc, _ = env.make(max_queue_spans=1)

        def boom(reason, count):
            raise RuntimeError("callback")

        proc._on_drop = boom
        proc.on_end(_Span(env.trace))
        proc.on_end(_Span(env.trace))
        assert _stats(proc) == {"queued": 1, "dropped_queue_full": 1}

    def test_size_estimate_error_propagates_after_append(self, env):
        proc, _ = env.make()

        class BadSpan(_Span):
            @property
            def name(self):
                raise RuntimeError("name read")

            @name.setter
            def name(self, value):
                pass

        with pytest.raises(RuntimeError, match="name read"):
            proc.on_end(BadSpan(env.trace))
        assert len(proc._queue) == 1
        assert proc.stats()["queued"] == 0
        assert proc._cond.acquire(blocking=False)
        proc._cond.release()


class TestWakeUp:
    def _counting(self, env, proc):
        calls = []
        real = proc._cond.notify_all

        def notify():
            env.trace.append("notify")
            calls.append(len(proc._queue))
            real()

        proc._cond.notify_all = notify
        return calls

    def test_batch_count_boundary(self, env):
        proc, _ = env.make(max_batch_spans=3)
        calls = self._counting(env, proc)
        proc.on_end(_Span(env.trace))
        proc.on_end(_Span(env.trace))
        assert calls == []
        proc.on_end(_Span(env.trace))
        assert calls == [3]

    def test_batch_bytes_boundary(self, env):
        # One span estimates to 200 + 3 + 2 * 48 = 299 bytes.
        proc, _ = env.make(max_batch_bytes=299)
        calls = self._counting(env, proc)
        proc.on_end(_Span(env.trace))
        assert calls == [1]
        other, _ = env.make(max_batch_bytes=300)
        calls = self._counting(env, other)
        other.on_end(_Span(env.trace))
        assert calls == []

    def test_wake_follows_worker_check(self, env):
        proc, _ = env.make(max_batch_spans=1)
        self._counting(env, proc)
        env.trace.clear()
        proc.on_end(_Span(env.trace))
        assert env.trace[-2:] == [("worker", 1), "notify"]

    def test_dropped_span_never_wakes(self, env):
        proc, _ = env.make(max_queue_spans=1, max_batch_spans=1)
        proc.on_end(_Span(env.trace))
        calls = self._counting(env, proc)
        proc.on_end(_Span(env.trace))
        assert calls == []
