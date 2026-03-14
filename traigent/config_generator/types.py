"""Output types for the auto-optimization config generator.

All types are frozen dataclasses for immutability and thread safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TVarSpec:
    """A fully-resolved tuned variable specification with its ParameterRange."""

    name: str
    range_type: str  # "Range", "IntRange", "LogRange", "Choices"
    range_kwargs: dict[str, Any] = field(default_factory=dict)
    source: str = "preset"  # "preset" | "heuristic" | "llm" | "detection"
    confidence: float = 1.0
    reasoning: str = ""

    def to_range_code(self) -> str:
        """Generate Python code like ``Range(low=0.0, high=2.0)``."""
        parts = [f"{k}={v!r}" for k, v in self.range_kwargs.items()]
        return f"{self.range_type}({', '.join(parts)})"


@dataclass(frozen=True)
class ObjectiveSpec:
    """A generated objective definition."""

    name: str
    orientation: str = "maximize"  # "maximize" | "minimize"
    weight: float = 1.0
    source: str = "default"  # "default" | "heuristic" | "llm"
    reasoning: str = ""


@dataclass(frozen=True)
class BenchmarkSpec:
    """A benchmark suggestion with format template."""

    name: str
    description: str = ""
    example_schema: dict[str, Any] = field(default_factory=dict)
    source: str = "catalog"  # "catalog" | "llm"
    sample_examples: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SafetySpec:
    """A generated safety constraint."""

    metric_name: str
    operator: str  # ">=" | "<="
    threshold: float
    agent_type: str = ""
    source: str = "preset"  # "preset" | "llm"
    reasoning: str = ""


@dataclass(frozen=True)
class StructuralConstraintSpec:
    """A generated structural constraint between TVars."""

    description: str
    constraint_code: str  # Python code string
    requires_tvars: tuple[str, ...] = ()
    source: str = "template"  # "template" | "llm"
    reasoning: str = ""


@dataclass(frozen=True)
class TVarRecommendation:
    """A recommended additional TVAR the user may not have considered."""

    name: str
    range_type: str
    range_kwargs: dict[str, Any] = field(default_factory=dict)
    category: str = ""  # "prompting" | "retrieval" | "model" | "evaluation"
    reasoning: str = ""
    impact_estimate: str = "medium"  # "high" | "medium" | "low"

    def to_range_code(self) -> str:
        """Generate Python code for the recommended range."""
        parts = [f"{k}={v!r}" for k, v in self.range_kwargs.items()]
        return f"{self.range_type}({', '.join(parts)})"


@dataclass(frozen=True)
class AutoConfigResult:
    """Complete output from the auto-config generator."""

    tvars: tuple[TVarSpec, ...] = ()
    objectives: tuple[ObjectiveSpec, ...] = ()
    benchmarks: tuple[BenchmarkSpec, ...] = ()
    safety_constraints: tuple[SafetySpec, ...] = ()
    structural_constraints: tuple[StructuralConstraintSpec, ...] = ()
    recommendations: tuple[TVarRecommendation, ...] = ()
    agent_type: str | None = None
    warnings: tuple[str, ...] = ()
    llm_calls_made: int = 0
    llm_cost_usd: float = 0.0

    def to_decorator_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs suitable for ``@traigent.optimize(...)``.

        Returns actual ``ParameterRange`` and ``SafetyConstraint`` objects
        that can be passed directly to the decorator.
        """
        kwargs: dict[str, Any] = {}

        if self.tvars:
            config_space: dict[str, Any] = {}
            for tvar in self.tvars:
                config_space[tvar.name] = _reconstruct_range(tvar)
            kwargs["configuration_space"] = config_space

        if self.objectives:
            kwargs["objectives"] = [obj.name for obj in self.objectives]

        if self.safety_constraints:
            kwargs["safety_constraints"] = [
                _reconstruct_safety(sc) for sc in self.safety_constraints
            ]

        return kwargs

    def to_dict_kwargs(self) -> dict[str, Any]:
        """Convert to serializable kwargs dict (for JSON/TVL export).

        Same shape as ``to_decorator_kwargs`` but with plain dicts
        instead of live objects.
        """
        kwargs: dict[str, Any] = {}

        if self.tvars:
            config_space: dict[str, Any] = {}
            for tvar in self.tvars:
                config_space[tvar.name] = {
                    "type": tvar.range_type,
                    **tvar.range_kwargs,
                }
            kwargs["configuration_space"] = config_space

        if self.objectives:
            kwargs["objectives"] = [obj.name for obj in self.objectives]

        return kwargs

    def to_tvl_spec(self, module_name: str = "auto_generated") -> dict[str, Any]:
        """Convert to a TVL 0.9 spec dict (YAML-serializable).

        Leverages ``ConfigSpace.to_tvl_spec()`` for tvar/constraint
        conversion and appends objectives + safety sections.
        """
        # Lazy import to avoid circular dependency
        from traigent.api.config_space import ConfigSpace

        # Build a ConfigSpace from live ParameterRange objects
        decorator_kwargs = self.to_decorator_kwargs()
        config_space = ConfigSpace.from_decorator_args(
            configuration_space=decorator_kwargs.get("configuration_space"),
        )

        spec: dict[str, Any] = config_space.to_tvl_spec()
        spec["header"] = {"module": module_name}

        # Append structural constraints from the pipeline (stored as code
        # strings, not live Constraint objects, so we add them directly).
        if self.structural_constraints:
            structural = [
                {
                    "description": sc.description,
                    "code": sc.constraint_code,
                    "requires": list(sc.requires_tvars),
                }
                for sc in self.structural_constraints
            ]
            spec.setdefault("constraints", {})["structural"] = structural

        if self.objectives:
            spec["objectives"] = [
                {
                    "name": obj.name,
                    "direction": obj.orientation,
                    "weight": obj.weight,
                }
                for obj in self.objectives
            ]

        if self.safety_constraints:
            spec["safety"] = [
                {
                    "metric": sc.metric_name,
                    "operator": sc.operator,
                    "threshold": sc.threshold,
                }
                for sc in self.safety_constraints
            ]

        return spec

    def to_python_code(self) -> str:
        """Generate copy-pasteable ``@traigent.optimize(...)`` Python code."""
        lines: list[str] = []
        lines.append("@traigent.optimize(")

        # configuration_space
        if self.tvars:
            lines.append("    configuration_space={")
            for tvar in self.tvars:
                lines.append(f"        {tvar.name!r}: {tvar.to_range_code()},")
            lines.append("    },")

        # objectives
        if self.objectives:
            obj_list = ", ".join(f"{o.name!r}" for o in self.objectives)
            lines.append(f"    objectives=[{obj_list}],")

        # safety_constraints
        if self.safety_constraints:
            lines.append("    safety_constraints=[")
            for sc in self.safety_constraints:
                metric_expr = _metric_expression(sc.metric_name)
                method = "above" if sc.operator == ">=" else "below"
                lines.append(f"        {metric_expr}.{method}({sc.threshold}),")
            lines.append("    ],")

        # recommendations as comments
        if self.recommendations:
            lines.append("    # Recommended additional TVars:")
            for rec in self.recommendations:
                lines.append(f"    #   {rec.name!r}: {rec.to_range_code()},")

        lines.append(")")
        return "\n".join(lines)


# Metrics in traigent.api.safety that are factory functions (need "()")
# vs module-level instances (used as-is).
_FACTORY_METRICS = frozenset(
    {
        "hallucination_rate",
        "toxicity_score",
        "bias_score",
        "safety_score",
        "custom_safety",
    }
)


def _metric_expression(metric_name: str) -> str:
    """Return a valid Python expression for a safety metric.

    Factory-function metrics need ``()``, module-level instances don't.
    """
    if metric_name in _FACTORY_METRICS:
        return f"{metric_name}()"
    return metric_name


def _reconstruct_range(tvar: TVarSpec) -> Any:
    """Reconstruct a live ``ParameterRange`` from a ``TVarSpec``.

    Uses lazy imports to avoid circular dependency with ``traigent.api``.
    """
    from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range

    constructors = {
        "Range": Range,
        "IntRange": IntRange,
        "LogRange": LogRange,
        "Choices": Choices,
    }
    cls = constructors.get(tvar.range_type)
    if cls is None:
        raise ValueError(f"Unknown range type: {tvar.range_type!r}")
    return cls(**tvar.range_kwargs, name=tvar.name)


def _reconstruct_safety(sc: SafetySpec) -> Any:
    """Reconstruct a live ``SafetyConstraint`` from a ``SafetySpec``.

    Uses lazy imports to avoid circular dependency with ``traigent.api``.
    """
    from traigent.api.safety import (
        bias_score,
        custom_safety,
        faithfulness,
        hallucination_rate,
        safety_score,
        toxicity_score,
    )

    metrics: dict[str, Any] = {
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
        "toxicity_score": toxicity_score,
        "bias_score": bias_score,
        "safety_score": safety_score,
        "custom_safety": custom_safety,
    }
    metric_or_factory = metrics.get(sc.metric_name)
    if metric_or_factory is None:
        raise ValueError(f"Unknown safety metric: {sc.metric_name!r}")

    # Factory metrics need to be called to get an instance
    metric = (
        metric_or_factory() if sc.metric_name in _FACTORY_METRICS else metric_or_factory
    )
    if sc.operator == ">=":
        return metric.above(sc.threshold)  # type: ignore[union-attr]
    return metric.below(sc.threshold)  # type: ignore[union-attr]
