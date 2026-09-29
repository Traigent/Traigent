"""Head sampling: shared vectors, ratio accuracy, parent-based inheritance."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from traigent.observability.otel import sampling as S

VECTORS = json.loads(
    (
        Path(__file__).parents[3] / "fixtures/observability/sampling_vectors_v1.json"
    ).read_text()
)["vectors"]


@pytest.mark.parametrize("vec", VECTORS, ids=lambda v: f"{v['rate']}-{v['trace_id'][-16:]}")
def test_shared_vectors(vec):
    assert S.trace_id_sampled(int(vec["trace_id"], 16), float(vec["rate"])) is vec["sampled"]


def test_vectors_cover_boundaries():
    assert any(v["rate"] == "1" and v["sampled"] for v in VECTORS)
    assert any(v["rate"] == "0" and not v["sampled"] for v in VECTORS)
    assert {True, False} <= {v["sampled"] for v in VECTORS if v["rate"] == "0.5"}


def test_ratio_within_one_and_half_points():
    rng = random.Random(7)
    n = 40_000
    for rate in (0.1, 0.5, 0.9):
        hits = sum(S.trace_id_sampled(rng.getrandbits(128), rate) for _ in range(n))
        assert abs(hits / n - rate) < 0.015


def test_negative_control_mutant_rules_fail_the_vectors():
    """Wrong rules (high bits, <=) must disagree with at least one shared vector."""

    def mutants():
        yield lambda t, r: (t >> 64) < S.rate_threshold(r)
        yield lambda t, r: (t & ((1 << 64) - 1)) <= S.rate_threshold(r)
        yield lambda t, r: (t & ((1 << 32) - 1)) < S.rate_threshold(r) >> 32

    for mutant in mutants():
        assert any(
            mutant(int(v["trace_id"], 16), float(v["rate"])) is not v["sampled"]
            for v in VECTORS
        )


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_rates_rejected(bad):
    with pytest.raises(ValueError):
        S.validate_rate(bad)


def _provider(rate):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        sampler=S.parent_based_ratio_sampler(rate), shutdown_on_exit=False
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_rate_zero_drops_everything_and_rate_one_keeps_everything():
    p0, e0 = _provider(0.0)
    p1, e1 = _provider(1.0)
    for _ in range(20):
        with p0.get_tracer("t").start_as_current_span("s"):
            pass
        with p1.get_tracer("t").start_as_current_span("s"):
            pass
    assert len(e0.get_finished_spans()) == 0
    assert len(e1.get_finished_spans()) == 20


def test_children_follow_the_root_decision_never_split():
    provider, exporter = _provider(0.5)
    tracer = provider.get_tracer("t")
    for _ in range(200):
        with tracer.start_as_current_span("root"):
            with tracer.start_as_current_span("child"):
                pass
    by_trace: dict[int, list[str]] = {}
    for s in exporter.get_finished_spans():
        by_trace.setdefault(s.context.trace_id, []).append(s.name)
    assert by_trace  # some sampled
    assert all(sorted(v) == ["child", "root"] for v in by_trace.values())
    assert len(by_trace) < 200  # and some dropped


@pytest.mark.parametrize("flag,expected", [(TraceFlags.SAMPLED, 1), (TraceFlags.DEFAULT, 0)])
def test_inherited_remote_parent_decision_wins_over_ratio(flag, expected):
    """Inherited-unsampled stays unsampled even at rate 1.0; sampled stays sampled at 0.0."""
    rate = 1.0 if expected == 0 else 0.0
    provider, exporter = _provider(rate)
    remote = SpanContext(
        trace_id=random.Random(3).getrandbits(127) + 1,
        span_id=99,
        is_remote=True,
        trace_flags=TraceFlags(flag),
    )
    ctx = trace.set_span_in_context(NonRecordingSpan(remote))
    with provider.get_tracer("t").start_as_current_span("child", context=ctx):
        pass
    assert len(exporter.get_finished_spans()) == expected
