"""Output types for the auto-optimization config generator.

All types are frozen dataclasses for immutability and thread safety.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


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


def tvar_spec_problem(spec: TVarSpec) -> str | None:
    """Return why ``spec`` is not a usable search-space entry, or ``None``.

    Rejects reversed or empty numeric ranges (``low >= high``), a
    non-positive ``LogRange`` lower bound, and ``Choices`` with fewer than two
    distinct-able values (a one-value choice tunes nothing).
    """
    kwargs = spec.range_kwargs
    if not isinstance(kwargs, dict):
        return "range kwargs are not a mapping"
    if spec.range_type in ("Range", "IntRange", "LogRange"):
        bounds = (kwargs.get("low"), kwargs.get("high"))
        for bound in bounds:
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                return f"{spec.range_type} bounds must be numbers"
            if not math.isfinite(bound):
                return f"{spec.range_type} bounds must be finite"
        low, high = bounds
        assert low is not None and high is not None  # nosec B101 - checked above
        if low >= high:
            return f"empty or reversed range (low={low!r}, high={high!r})"
        if spec.range_type == "LogRange" and low <= 0:
            return f"LogRange low must be positive (low={low!r})"
        return None
    if spec.range_type == "Choices":
        values = kwargs.get("values")
        if not isinstance(values, (list, tuple)):
            return "Choices values must be a list"
        if len(values) < 2:
            return f"Choices needs at least two values (got {len(values)})"
        return None
    return f"unknown range type {spec.range_type!r}"


@dataclass(frozen=True)
class ObjectiveSpec:
    """A generated objective definition."""

    name: str
    orientation: Literal["maximize", "minimize"]
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
class EvidenceRef:
    """Public-safe reference to measured evidence supporting a recommendation.

    Carries only publishable provenance (scope/metric/n/model/baseline/candidate/
    delta/limitations). Internal artifact paths and run IDs are intentionally
    excluded so this can ship in the public SDK; richer internal provenance, if
    any, lives backend-side.
    """

    scope: str
    metric: str
    n: int
    model: str
    baseline: Any
    candidate: Any
    delta: float | None = None
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TVarRecommendation:
    """A recommended additional TVAR the user may not have considered."""

    name: str
    range_type: str
    range_kwargs: dict[str, Any] = field(default_factory=dict)
    category: str = ""  # "prompting" | "retrieval" | "model" | "evaluation"
    reasoning: str = ""
    impact_estimate: str = "medium"  # "high" | "medium" | "low"
    entry_id: str = ""
    catalog_entry_id: str = ""
    kind: str = ""  # "value" | "cardinality" | "topology" | "policy"
    effectuation_status: str = ""  # "executable" | "manual_guidance" | "advisory"
    effectuation_strategy: str = ""
    evidence_refs: tuple[EvidenceRef, ...] = ()
    apply_guidance: str = ""
    recommended_values: tuple[Any, ...] = ()

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
            from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

            kwargs["objectives"] = ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition(
                        name=obj.name,
                        orientation=obj.orientation,
                        weight=obj.weight,
                    )
                    for obj in self.objectives
                ]
            )

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
            kwargs["objectives"] = [
                {
                    "name": obj.name,
                    "orientation": obj.orientation,
                    "weight": obj.weight,
                }
                for obj in self.objectives
            ]

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
        # The loader reads the module header from the ``tvl`` section; a
        # ``header`` section is rejected as an unknown top-level key (#2647).
        spec["tvl"] = {"module": module_name}

        # Structural constraints are stored as code strings; translate the known
        # shapes to loader-accepted expr / when-then entries and skip (with a
        # value-free warning) any shape the loader grammar cannot express (#2647).
        if self.structural_constraints:
            from traigent.config_generator.tvl_constraints import (
                structural_constraints_to_tvl,
            )

            structural = structural_constraints_to_tvl(self.structural_constraints)
            if structural:
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
            spec["safety_constraints"] = [
                {
                    "metric": sc.metric_name,
                    "operator": sc.operator,
                    "threshold": sc.threshold,
                }
                for sc in self.safety_constraints
            ]

        return spec

    def to_python_code(self) -> str:
        """Generate copy-pasteable ``@traigent.optimize(...)`` Python code.

        Raises
        ------
        ValueError
            If ``structural_constraints`` is non-empty. The Python decorator
            surface does not currently support structural constraints, and
            silently dropping them would let invalid sampled configurations
            slip through. Export to TVL (``-o tvl`` / ``to_tvl_spec()``) to
            preserve them.
        """
        if self.structural_constraints:
            raise ValueError(_structural_constraint_error(self.structural_constraints))

        lines: list[str] = ["@traigent.optimize("]
        if self.tvars:
            lines.extend(_configuration_space_lines(self.tvars))
        if self.objectives:
            lines.extend(_objective_lines(self.objectives))
        if self.safety_constraints:
            lines.extend(_safety_constraint_lines(self.safety_constraints))
        if self.recommendations:
            lines.extend(_recommendation_comment_lines(self.recommendations))

        lines.append(")")
        return "\n".join(lines)


def _configuration_space_lines(tvars: tuple[TVarSpec, ...]) -> list[str]:
    lines = ["    configuration_space={"]
    for tvar in tvars:
        lines.append(f"        {tvar.name!r}: {tvar.to_range_code()},")
    lines.append("    },")
    return lines


def _objective_lines(objectives: tuple[ObjectiveSpec, ...]) -> list[str]:
    lines = ["    objectives=ObjectiveSchema.from_objectives(["]
    for objective in objectives:
        lines.append(
            "        ObjectiveDefinition("
            f"name={objective.name!r}, orientation={objective.orientation!r}, "
            f"weight={objective.weight!r}),"
        )
    lines.append("    ]),")
    return lines


def _safety_constraint_lines(constraints: tuple[SafetySpec, ...]) -> list[str]:
    lines = ["    safety_constraints=["]
    for sc in constraints:
        metric_expr = _metric_expression(sc.metric_name)
        method = "above" if sc.operator == ">=" else "below"
        lines.append(f"        {metric_expr}.{method}({sc.threshold}),")
    lines.append("    ],")
    return lines


def _recommendation_comment_lines(
    recommendations: tuple[TVarRecommendation, ...],
) -> list[str]:
    lines = ["    # Recommended additional TVars:"]
    for rec in recommendations:
        lines.append(f"    #   {rec.name!r}: {rec.to_range_code()},")
    return lines


def _structural_constraint_error(
    constraints: tuple[StructuralConstraintSpec, ...],
) -> str:
    """Build the rejection message for structural constraints in Python output."""
    count = len(constraints)
    sample = "; ".join(sc.description or sc.constraint_code for sc in constraints[:3])
    suffix = "" if count <= 3 else f" (+{count - 3} more)"
    return (
        f"Cannot emit @traigent.optimize(...) Python code: {count} generated "
        f"structural constraint(s) would be silently dropped because the "
        f"decorator surface does not accept them. To preserve them, export "
        f"with `-o tvl` (not combinable with --apply) or call "
        f"AutoConfigResult.to_tvl_spec(). "
        f"Constraints: {sample}{suffix}"
    )


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
