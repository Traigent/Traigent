"""Head sampling for Traigent-created tracer providers.

Decision rule (identical in every Traigent SDK; shared vectors in
``tests/fixtures/observability/sampling_vectors_v1.json``)::

    micro     = round(rate * 1_000_000)          # rate resolution 1e-6
    threshold = (micro * 2**64) // 1_000_000
    sampled   = low64(trace_id) < threshold      # low64 = trace_id & (2**64-1)

``rate`` 1.0 always samples, 0.0 never.  The sampler is parent-based: a sampled
or unsampled parent (local or remote) decides for its children, so a trace is
never split.  There is intentionally NO error-rescue / tail behaviour: a span
that was not sampled at the head is dropped.

An existing provider's sampler is authoritative; Traigent never replaces it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import (
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)
from opentelemetry.trace import Link, SpanKind, TraceState
from opentelemetry.util.types import Attributes

_MASK64 = (1 << 64) - 1
_MICRO = 1_000_000


def validate_rate(rate: float) -> float:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError("sample_rate must be a number between 0 and 1")
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError("sample_rate must be between 0 and 1")
    return float(rate)


def rate_threshold(rate: float) -> int:
    micro = round(validate_rate(rate) * _MICRO)
    return (micro * (1 << 64)) // _MICRO


def trace_id_sampled(trace_id: int, rate: float) -> bool:
    return (trace_id & _MASK64) < rate_threshold(rate)


class TraceIdRatioRoot(Sampler):
    """Root-span sampler using the shared low-64-bit rule."""

    def __init__(self, rate: float) -> None:
        self._rate = validate_rate(rate)
        self._threshold = rate_threshold(self._rate)

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        sampled = (trace_id & _MASK64) < self._threshold
        return SamplingResult(
            Decision.RECORD_AND_SAMPLE if sampled else Decision.DROP,
            attributes=None,
            trace_state=trace_state,
        )

    def get_description(self) -> str:
        return f"TraigentTraceIdRatio{{{self._rate}}}"


def parent_based_ratio_sampler(rate: float) -> Sampler:
    return ParentBased(root=TraceIdRatioRoot(rate))
