"""Metric registry defining aggregation strategies."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

AggregatorType = Literal["mean", "sum", "last"]


@dataclass(slots=True)
class MetricSpec:
    """Specification for aggregating a metric across trials."""

    name: str
    aggregator: AggregatorType
    mandatory: bool = False
    description: str | None = None


class MetricRegistry:
    """Registry mapping metric names to aggregation strategies."""

    def __init__(self, *, default_aggregator: AggregatorType = "mean") -> None:
        self._specs: dict[str, MetricSpec] = {}
        self._default_aggregator: AggregatorType = default_aggregator

    @classmethod
    def default(cls) -> MetricRegistry:
        registry = cls()
        registry.register_many(
            MetricSpec(name="total_cost", aggregator="sum", mandatory=True),
            MetricSpec(name="total_tokens", aggregator="sum", mandatory=True),
            MetricSpec(name="total_duration", aggregator="sum", mandatory=True),
            MetricSpec(
                name="examples_attempted_total", aggregator="sum", mandatory=True
            ),
        )
        return registry

    def register(self, spec: MetricSpec) -> None:
        self._specs[spec.name] = spec

    def register_many(self, *specs: MetricSpec) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> MetricSpec | None:
        return self._specs.get(name)

    def aggregator_for(self, name: str) -> AggregatorType:
        spec = self.get(name)
        if spec is None:
            return self._default_aggregator
        return spec.aggregator

    def is_mandatory(self, name: str) -> bool:
        spec = self.get(name)
        return bool(spec and spec.mandatory)

    def specs(self) -> tuple[MetricSpec, ...]:
        return tuple(self._specs.values())

    def clone(self) -> MetricRegistry:
        cloned = MetricRegistry(default_aggregator=self._default_aggregator)
        if self._specs:
            cloned.register_many(*(replace(spec) for spec in self._specs.values()))
        return cloned

    @property
    def mandatory_metric_names(self) -> Iterable[str]:
        return (name for name, spec in self._specs.items() if spec.mandatory)

    @property
    def default_aggregator(self) -> AggregatorType:
        return self._default_aggregator


__all__ = ["MetricSpec", "MetricRegistry", "AggregatorType"]
