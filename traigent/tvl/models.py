"""TVL 0.9 data models for the Traigent SDK.

This module provides typed data models aligned with the TVL 0.9 specification,
including TVAR declarations, domain specifications, banded objectives,
and promotion policies.

Concept: CONC-TVLSpec
Implements: FUNC-TVLSPEC
Sync: SYNC-OptimizationFlow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Type aliases for TVL type system
TVarType = Literal["bool", "int", "float", "enum", "tuple", "callable"]
DomainKind = Literal["enum", "range", "set", "registry"]
AdjustMethod = Literal["none", "BH"]
TieBreaker = Literal["min_abs_deviation", "custom"]


@dataclass(slots=True)
class DomainSpec:
    """Domain specification for a TVAR.

    Supports four domain kinds from TVL 0.9:
    - enum: Finite set of literal values
    - range: Numeric interval with optional resolution
    - set: Explicit set wrapper
    - registry: External registry reference

    Attributes:
        kind: The type of domain specification.
        values: List of allowed values (for enum/set kinds).
        range: Tuple of (min, max) bounds (for range kind).
        resolution: Optional discretization step for ranges.
        registry: Registry identifier (for registry kind).
        filter: Optional filter expression for registry lookups.
        version: Optional version constraint for registry lookups.
    """

    kind: DomainKind
    values: list[Any] | None = None
    range: tuple[int, int] | tuple[float, float] | None = None
    resolution: float | None = None
    registry: str | None = None
    filter: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        """Validate domain specification."""
        if self.kind in ("enum", "set") and not self.values:
            raise ValueError(f"Domain kind '{self.kind}' requires non-empty values")
        if self.kind == "range" and self.range is None:
            raise ValueError("Domain kind 'range' requires a range tuple")
        if self.kind == "registry" and not self.registry:
            raise ValueError("Domain kind 'registry' requires a registry identifier")

    def to_configuration_space_entry(
        self,
    ) -> list[Any] | tuple[int, int] | tuple[float, float]:
        """Convert to Traigent configuration space format.

        Returns:
            List of values for categorical types, or (min, max) tuple for numeric.
        """
        if self.kind in ("enum", "set"):
            return self.values or []
        if self.kind == "range":
            if self.range is None:
                raise ValueError("Range domain has no range defined")
            return self.range
        if self.kind == "registry":
            raise ValueError(
                "Registry domains must be resolved into concrete values via a "
                "RegistryResolver during spec loading."
            )
        return []


@dataclass(slots=True)
class TVarDecl:
    """TVAR (Tuned Variable) declaration from TVL 0.9.

    Attributes:
        name: Unique identifier for the TVAR.
        type: Normalized type (bool, int, float, enum, tuple, callable).
        raw_type: Original type string from spec (e.g., "enum[str]").
        domain: Domain specification defining valid values.
        default: Optional default value.
        unit: Optional unit of measurement.
    """

    name: str
    type: TVarType
    raw_type: str
    domain: DomainSpec
    default: Any | None = None
    unit: str | None = None

    def to_configuration_space_entry(
        self,
    ) -> tuple[str, list[Any] | tuple[int, int] | tuple[float, float]]:
        """Convert to Traigent configuration space format.

        Returns:
            Tuple of (name, domain_spec) for configuration space.
        """
        return (self.name, self.domain.to_configuration_space_entry())


@dataclass(slots=True)
class BandTarget:
    """Target band for banded objectives.

    Can be specified either as:
    - An interval [low, high]
    - A center with tolerance (center +/- tol)

    Attributes:
        low: Lower bound of the band.
        high: Upper bound of the band.
        center: Center of the band (alternative to low/high).
        tol: Tolerance around the center (alternative to low/high).
    """

    low: float | None = None
    high: float | None = None
    center: float | None = None
    tol: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize band target."""
        has_interval = self.low is not None and self.high is not None
        has_center_tol = self.center is not None and self.tol is not None

        if not has_interval and not has_center_tol:
            raise ValueError("BandTarget requires either (low, high) or (center, tol)")

        # Convert center/tol to low/high if needed
        if has_center_tol and not has_interval:
            if self.center is None or self.tol is None:
                raise ValueError(
                    "center and tol must be set when using center/tol mode"
                )
            self.low = self.center - self.tol
            self.high = self.center + self.tol

        # Validate bounds
        if self.low is not None and self.high is not None:
            if self.low >= self.high:
                raise ValueError(
                    f"Band low ({self.low}) must be less than high ({self.high})"
                )

    def contains(self, value: float) -> bool:
        """Check if a value falls within the band.

        Args:
            value: The value to check.

        Returns:
            True if value is within [low, high], False otherwise.
        """
        if self.low is None or self.high is None:
            return False
        return self.low <= value <= self.high

    def deviation(self, value: float) -> float:
        """Calculate the deviation from the band.

        Args:
            value: The value to measure deviation from.

        Returns:
            0 if inside band, positive distance to nearest bound otherwise.
        """
        if self.low is None or self.high is None:
            return float("inf")

        if value < self.low:
            return self.low - value
        if value > self.high:
            return value - self.high
        return 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[float]) -> BandTarget:
        """Create BandTarget from dict or list representation.

        Args:
            data: Either [low, high] list or {center, tol}/{low, high} dict.

        Returns:
            BandTarget instance.
        """
        if isinstance(data, list):
            if len(data) != 2:
                raise ValueError("Band target list must have exactly 2 elements")
            return cls(low=float(data[0]), high=float(data[1]))

        if "center" in data and "tol" in data:
            return cls(center=float(data["center"]), tol=float(data["tol"]))

        if "low" in data and "high" in data:
            return cls(low=float(data["low"]), high=float(data["high"]))

        raise ValueError(
            "Band target dict must have either (center, tol) or (low, high)"
        )


@dataclass(slots=True)
class ChanceConstraint:
    """Chance constraint for promotion policy.

    Enforces probabilistic constraints on metrics during promotion.

    Attributes:
        name: Metric identifier to constrain.
        threshold: Target value the metric must satisfy.
        confidence: Confidence level for the constraint (0 < confidence <= 1).
    """

    name: str
    threshold: float
    confidence: float

    def __post_init__(self) -> None:
        """Validate chance constraint."""
        if not 0 < self.confidence <= 1:
            raise ValueError(f"Confidence must be in (0, 1], got {self.confidence}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChanceConstraint:
        """Create ChanceConstraint from dict representation."""
        return cls(
            name=data["name"],
            threshold=float(data["threshold"]),
            confidence=float(data["confidence"]),
        )


@dataclass(slots=True)
class PromotionPolicy:
    """Promotion policy configuration for epsilon-Pareto dominance.

    Controls how candidates are promoted based on statistical testing
    with configurable error rates and effect sizes.

    Attributes:
        dominance: Dominance relation (TVL 0.9 mandates "epsilon_pareto").
        alpha: Family-wise error rate budget (0 < alpha < 1, default 0.05).
        min_effect: Per-objective epsilon tolerances for dominance comparison.
        adjust: Multiple testing adjustment ("none" or "BH" for Benjamini-Hochberg).
        chance_constraints: Hard constraints with confidence thresholds.
        tie_breakers: Secondary ordering rules for ties.
    """

    dominance: Literal["epsilon_pareto"] = "epsilon_pareto"
    alpha: float = 0.05
    min_effect: dict[str, float] = field(default_factory=dict)
    adjust: AdjustMethod = "none"
    chance_constraints: list[ChanceConstraint] = field(default_factory=list)
    tie_breakers: dict[str, TieBreaker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate promotion policy."""
        if not 0 < self.alpha < 1:
            raise ValueError(f"Alpha must be in (0, 1), got {self.alpha}")

        for name, epsilon in self.min_effect.items():
            if epsilon < 0:
                raise ValueError(
                    f"min_effect for '{name}' must be non-negative, got {epsilon}"
                )

    def get_epsilon(self, objective_name: str, default: float = 0.0) -> float:
        """Get the epsilon tolerance for an objective.

        Args:
            objective_name: Name of the objective.
            default: Default epsilon if not specified.

        Returns:
            The epsilon tolerance for the objective.
        """
        return self.min_effect.get(objective_name, default)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromotionPolicy:
        """Create PromotionPolicy from dict representation."""
        chance_constraints = [
            ChanceConstraint.from_dict(cc) for cc in data.get("chance_constraints", [])
        ]

        return cls(
            dominance=data.get("dominance", "epsilon_pareto"),
            alpha=float(data.get("alpha", 0.05)),
            min_effect=data.get("min_effect", {}),
            adjust=data.get("adjust", "none"),
            chance_constraints=chance_constraints,
            tie_breakers=data.get("tie_breakers", {}),
        )


@dataclass(slots=True)
class StructuralConstraint:
    """Structural constraint from TVL 0.9.

    Represents either a standalone expression or a when/then implication.

    Attributes:
        expr: Standalone constraint expression.
        when: Guard/antecedent for conditional constraint.
        then: Consequent for conditional constraint.
        index: Position in the constraints array (for diagnostics).
    """

    expr: str | None = None
    when: str | None = None
    then: str | None = None
    index: int = 0

    def __post_init__(self) -> None:
        """Validate structural constraint."""
        has_expr = self.expr is not None
        has_implication = self.when is not None and self.then is not None

        if not has_expr and not has_implication:
            raise ValueError(
                "StructuralConstraint requires either 'expr' or both 'when' and 'then'"
            )

        if has_expr and has_implication:
            raise ValueError(
                "StructuralConstraint cannot have both 'expr' and 'when/then'"
            )

    def to_rule_expression(self) -> str:
        """Convert to a single rule expression.

        Implications are converted to: not(when) or then

        Returns:
            The constraint as a single expression string.
        """
        if self.expr is not None:
            return self.expr

        # Convert when/then implication to expression
        # when -> then is equivalent to: not(when) or then
        if self.when is None or self.then is None:
            raise ValueError("when and then must be set for implication constraints")
        return f"not ({self.when}) or ({self.then})"

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> StructuralConstraint:
        """Create StructuralConstraint from dict representation."""
        return cls(
            expr=data.get("expr"),
            when=data.get("when"),
            then=data.get("then"),
            index=index,
        )


@dataclass(slots=True)
class DerivedConstraint:
    """Derived constraint from TVL 0.9.

    Derived constraints are linear arithmetic expressions over environment
    symbols (e.g., prices, quotas) that are evaluated at runtime when the
    environment snapshot is known. They compile to decidable SMT fragments
    (QF_LIA/LRA).

    Note: These constraints are stored as data only in Traigent. Actual
    SMT compilation and solving is handled by TVL tools, not the SDK.

    Attributes:
        require: The linear arithmetic expression to satisfy.
        index: Position in the constraints array (for diagnostics).
        description: Optional human-readable description.
    """

    require: str
    index: int = 0
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate derived constraint."""
        if not self.require or not self.require.strip():
            raise ValueError("DerivedConstraint 'require' expression cannot be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> DerivedConstraint:
        """Create DerivedConstraint from dict representation."""
        require = data.get("require")
        if not isinstance(require, str):
            raise ValueError(
                f"Derived constraint at index {index} requires a 'require' string"
            )
        return cls(
            require=require,
            index=index,
            description=data.get("description"),
        )


@dataclass(slots=True)
class TVLHeader:
    """TVL module header containing metadata.

    Attributes:
        module: Fully-qualified module identifier (e.g., 'corp.product.spec').
        validation: Optional validation configuration for tooling.
    """

    module: str
    skip_budget_checks: bool = False
    skip_cost_estimation: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TVLHeader:
        """Create TVLHeader from dict representation."""
        module = data.get("module")
        if not isinstance(module, str):
            raise ValueError("TVL header requires a 'module' string")

        validation = data.get("validation", {})
        return cls(
            module=module,
            skip_budget_checks=validation.get("skip_budget_checks", False),
            skip_cost_estimation=validation.get("skip_cost_estimation", False),
        )


@dataclass(slots=True)
class EnvironmentSnapshot:
    """Environment snapshot from TVL 0.9.

    The environment snapshot (E_τ) provides symbols used to specialise domains
    and evaluate derived constraints at a specific point in time.

    Attributes:
        snapshot_id: RFC3339 timestamp labeling the environment snapshot.
        components: Implementation-defined map of environment components.
    """

    snapshot_id: str
    components: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentSnapshot:
        """Create EnvironmentSnapshot from dict representation."""
        snapshot_id = data.get("snapshot_id")
        if not isinstance(snapshot_id, str):
            raise ValueError("Environment requires a 'snapshot_id' string")

        return cls(
            snapshot_id=snapshot_id,
            components=data.get("components", {}),
        )


@dataclass(slots=True)
class EvaluationSet:
    """Evaluation set definition from TVL 0.9.

    Anchors the dataset and optional randomness seed used in validation runs.

    Attributes:
        dataset: Canonical identifier or URI for the evaluation dataset.
        seed: Optional deterministic seed for stochastic evaluation sets.
    """

    dataset: str
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationSet:
        """Create EvaluationSet from dict representation."""
        dataset = data.get("dataset")
        if not isinstance(dataset, str):
            raise ValueError("EvaluationSet requires a 'dataset' string")

        seed = data.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("EvaluationSet 'seed' must be an integer")

        return cls(dataset=dataset, seed=seed)


@dataclass(slots=True)
class ConvergenceCriteria:
    """Convergence criteria for exploration from TVL 0.9.

    Attributes:
        metric: Convergence signal monitored (hypervolume_improvement or none).
        window: Sliding window size for convergence metric aggregation.
        threshold: Minimum improvement required to continue exploration.
    """

    metric: str = "hypervolume_improvement"
    window: int = 5
    threshold: float = 0.01

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceCriteria:
        """Create ConvergenceCriteria from dict representation."""
        return cls(
            metric=data.get("metric", "hypervolume_improvement"),
            window=int(data.get("window", 5)),
            threshold=float(data.get("threshold", 0.01)),
        )


@dataclass(slots=True)
class ExplorationBudgets:
    """Hard limits on exploration from TVL 0.9.

    Attributes:
        max_trials: Maximum number of trials permitted.
        max_spend_usd: Budget cap on spend in USD.
        max_wallclock_s: Maximum wall-clock duration in seconds.
    """

    max_trials: int | None = None
    max_spend_usd: float | None = None
    max_wallclock_s: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplorationBudgets:
        """Create ExplorationBudgets from dict representation."""
        max_trials = data.get("max_trials")
        max_spend = data.get("max_spend_usd")
        max_wallclock = data.get("max_wallclock_s")

        return cls(
            max_trials=int(max_trials) if max_trials is not None else None,
            max_spend_usd=float(max_spend) if max_spend is not None else None,
            max_wallclock_s=int(max_wallclock) if max_wallclock is not None else None,
        )


# Type alias for objectives that can be either standard or banded
ObjectiveType = Literal["standard", "banded"]


def normalize_tvar_type(raw_type: str) -> TVarType | None:
    """Normalize a raw type string to a TVarType.

    Args:
        raw_type: Original type string (e.g., "enum[str]", "float").

    Returns:
        Normalized TVarType, or None if unsupported.
    """
    lowered = raw_type.strip().lower()

    if lowered.startswith("enum"):
        return "enum"
    if lowered.startswith("tuple"):
        return "tuple"
    if lowered.startswith("callable"):
        return "callable"
    if lowered in ("bool", "boolean"):
        return "bool"
    if lowered in ("int", "integer"):
        return "int"
    if lowered in ("float", "number", "continuous"):
        return "float"

    return None


def parse_domain_spec(
    tvar_name: str,
    tvar_type: TVarType,
    domain_data: Any,
) -> DomainSpec:
    """Parse domain specification from TVL spec data.

    Args:
        tvar_name: Name of the TVAR (for error messages).
        tvar_type: Normalized type of the TVAR.
        domain_data: Raw domain specification from YAML.

    Returns:
        DomainSpec instance.

    Raises:
        ValueError: If domain specification is invalid.
    """
    # Boolean type has implicit domain
    if tvar_type == "bool":
        if domain_data is None:
            return DomainSpec(kind="enum", values=[True, False])
        # Allow explicit boolean domain
        if isinstance(domain_data, list):
            values = [
                v if isinstance(v, bool) else str(v).lower() == "true"
                for v in domain_data
            ]
            return DomainSpec(kind="enum", values=values)

    # Handle list/array domains (enum values)
    if isinstance(domain_data, list):
        return DomainSpec(kind="enum", values=domain_data)

    # Handle dict domains
    if isinstance(domain_data, dict):
        # Range specification
        if "range" in domain_data:
            range_val = domain_data["range"]
            if isinstance(range_val, list) and len(range_val) == 2:
                range_tuple: tuple[int, int] | tuple[float, float]
                # Preserve integer type for int TVARs; reject lossy truncation.
                if tvar_type == "int":
                    low_raw, high_raw = range_val

                    def _coerce_int_bound(value: Any, which: str) -> int:
                        if isinstance(value, bool) or not isinstance(
                            value, (int, float)
                        ):
                            raise ValueError(
                                f"TVAR '{tvar_name}' int range bounds must be integers, "
                                f"got {which}={value!r}"
                            )
                        if isinstance(value, int):
                            return value
                        if not isinstance(value, float):
                            raise TypeError(
                                f"Expected int or float for {which}, got {type(value).__name__}"
                            )
                        if value.is_integer():
                            return int(value)
                        raise ValueError(
                            f"TVAR '{tvar_name}' int range bounds must be integers "
                            f"(no truncation), got {which}={value!r}"
                        )

                    range_tuple = (
                        _coerce_int_bound(low_raw, "min"),
                        _coerce_int_bound(high_raw, "max"),
                    )
                else:
                    range_tuple = (
                        float(range_val[0]),
                        float(range_val[1]),
                    )
                return DomainSpec(
                    kind="range",
                    range=range_tuple,
                    resolution=domain_data.get("resolution"),
                )
            raise ValueError(
                f"TVAR '{tvar_name}' range must be [min, max], got {range_val}"
            )

        # Set specification
        if "set" in domain_data:
            return DomainSpec(kind="set", values=domain_data["set"])

        # Registry specification
        if "registry" in domain_data:
            return DomainSpec(
                kind="registry",
                registry=domain_data["registry"],
                filter=domain_data.get("filter"),
                version=domain_data.get("version"),
            )

    raise ValueError(
        f"TVAR '{tvar_name}' has invalid domain specification: {domain_data}"
    )


@runtime_checkable
class RegistryResolver(Protocol):
    """Protocol for resolving registry domain values.

    Registry domains in TVL 0.9 allow TVARs to reference external registries
    (e.g., model catalogs, scorer registries). This protocol defines the
    interface for resolving those references at runtime.

    Implementers should provide concrete resolution logic for their specific
    registry systems. The resolver is called during spec loading when a
    TVAR has a registry domain.

    Example implementation:
        ```python
        class ModelCatalogResolver:
            def __init__(self, catalog: ModelCatalog):
                self.catalog = catalog

            def resolve(
                self,
                registry_id: str,
                filter_expr: str | None = None,
                version: str | None = None,
            ) -> list[Any]:
                models = self.catalog.list_models()
                if filter_expr:
                    models = [m for m in models if self._matches(m, filter_expr)]
                if version:
                    models = [m for m in models if m.version == version]
                return [m.id for m in models]

            def _matches(self, model, filter_expr: str) -> bool:
                # Parse and evaluate filter expression
                ...
        ```

    Note: Registry resolution is opt-in. If a TVL spec includes a registry
    domain and no resolver is provided to the spec loader, the SDK fails fast.
    """

    def resolve(
        self,
        registry_id: str,
        filter_expr: str | None = None,
        version: str | None = None,
    ) -> list[Any]:
        """Resolve a registry reference to concrete values.

        Args:
            registry_id: Identifier of the registry to query (e.g., "scorers",
                "models", "embeddings").
            filter_expr: Optional filter expression in the TVL filter syntax
                (e.g., "version >= 2", "provider = 'openai'").
            version: Optional version constraint for the registry lookup.

        Returns:
            List of resolved values that can be used as the domain for a TVAR.
            Returns an empty list if the registry is not found or no values
            match the filter.

        Raises:
            ValueError: If the registry_id is invalid or the filter_expr
                contains unsupported syntax.
        """
        ...
