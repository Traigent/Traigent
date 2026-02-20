"""Data types for PydanticAI integration metrics.

This module defines dataclasses for capturing and aggregating metrics
from PydanticAI agent runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRunMetrics:
    """Metrics from a single PydanticAI agent run.

    Captures token usage, cost, latency, and tool call counts
    from a single invocation of ``agent.run()`` or its variants.
    """

    model: str | None = None
    start_time: float = 0.0
    end_time: float | None = None
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    tool_calls: int = 0

    @property
    def latency_ms(self) -> float:
        """Wall-clock latency in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


@dataclass
class PydanticAIHandlerMetrics:
    """Aggregated metrics across multiple PydanticAI agent runs.

    Provides summary statistics and a ``to_measures_dict()`` method
    compatible with Traigent's ``MeasuresDict`` format.
    """

    runs: list[AgentRunMetrics] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.runs)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.runs)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.runs)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.runs)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.runs)

    @property
    def request_count(self) -> int:
        return sum(r.request_count for r in self.runs)

    @property
    def tool_call_count(self) -> int:
        return sum(r.tool_calls for r in self.runs)

    @property
    def run_count(self) -> int:
        return len(self.runs)

    def to_measures_dict(self, prefix: str = "") -> dict[str, float | int]:
        """Convert to a flat dict suitable for ``MeasuresDict``.

        Args:
            prefix: Optional prefix for all keys (e.g. ``"pydantic_ai_"``).

        Returns:
            Dict with numeric metric values.
        """
        return {
            f"{prefix}total_cost": self.total_cost,
            f"{prefix}total_latency_ms": self.total_latency_ms,
            f"{prefix}total_input_tokens": self.total_input_tokens,
            f"{prefix}total_output_tokens": self.total_output_tokens,
            f"{prefix}total_tokens": self.total_tokens,
            f"{prefix}request_count": self.request_count,
            f"{prefix}tool_call_count": self.tool_call_count,
            f"{prefix}run_count": self.run_count,
        }
