"""Objective definitions and validation for multi-objective optimization.

This module provides data structures and validation for objectives with
explicit orientation (maximize/minimize), weight normalization, and
banded objectives (TVL 0.9 support).

Concept: CONC-TVLSpec
Implements: FUNC-TVLSPEC
Sync: SYNC-OptimizationFlow
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from traigent.tvl.models import BandTarget


class AggregationMode(Enum):
    """Aggregation mode for combining multiple objective scores.

    Different modes provide different trade-off characteristics:
    - WEIGHTED_SUM: Simple weighted sum (default). Linear trade-offs.
    - HARMONIC: Harmonic mean of weighted scores. Penalizes extreme imbalances.
    - CHEBYSHEV: Minimax approach. Optimizes worst-case objective performance.
    """

    WEIGHTED_SUM = "sum"
    HARMONIC = "harmonic"
    CHEBYSHEV = "chebyshev"


@dataclass
class ObjectiveDefinition:
    """Definition of a single optimization objective.

    Supports both standard objectives (maximize/minimize) and banded objectives
    (TVL 0.9) where the goal is to fall within a target band.

    Attributes:
        name: Name of the objective (e.g., "accuracy", "cost")
        orientation: Whether to maximize or minimize this objective.
            For banded objectives, this is set to "band".
        weight: Positive weight for this objective
        normalization: Normalization strategy (default: "min_max")
        bounds: Optional bounds for the objective values
        unit: Optional unit of measurement (e.g., "USD", "percentage")
        band: Optional band target for banded objectives (TVL 0.9).
            When set, orientation should be "band".
        band_test: Statistical test for banded objectives ("TOST").
        band_alpha: Significance level for banded objective test.
    """

    name: str
    orientation: Literal["maximize", "minimize", "band"]
    weight: float
    normalization: str = "min_max"
    bounds: tuple[float, float] | None = None
    unit: str | None = None
    # Banded objective fields (TVL 0.9)
    band: BandTarget | None = None
    band_test: Literal["TOST"] = "TOST"
    band_alpha: float = 0.05

    def __post_init__(self) -> None:
        """Validate objective definition after initialization."""
        # Validate weight
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(
                f"Weight must be a finite positive number, "
                f"got {self.weight} for objective '{self.name}'"
            )

        # Validate orientation
        if self.orientation not in ["maximize", "minimize", "band"]:
            raise ValueError(
                f"Orientation must be 'maximize', 'minimize', or 'band', "
                f"got '{self.orientation}' for objective '{self.name}'"
            )

        # Validate banded objective configuration
        if self.orientation == "band":
            if self.band is None:
                raise ValueError(
                    f"Banded objective '{self.name}' requires a band target"
                )
        elif self.band is not None:
            # If band is provided but orientation is not "band", auto-set it
            object.__setattr__(self, "orientation", "band")

        # Validate bounds if provided
        if self.bounds is not None:
            if len(self.bounds) != 2:
                raise ValueError(
                    f"Bounds must be a tuple of (min, max), got {self.bounds}"
                )
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(
                    f"Invalid bounds: min ({self.bounds[0]}) must be less than max ({self.bounds[1]})"
                )

        # Validate normalization
        valid_normalizations = ["min_max", "z_score", "robust"]
        if self.normalization not in valid_normalizations:
            raise ValueError(
                f"Normalization must be one of {valid_normalizations}, "
                f"got '{self.normalization}' for objective '{self.name}'"
            )

        # Validate band_alpha
        if not 0 < self.band_alpha < 1:
            raise ValueError(f"band_alpha must be in (0, 1), got {self.band_alpha}")

    @property
    def is_banded(self) -> bool:
        """Check if this is a banded objective."""
        return self.band is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {
            "name": self.name,
            "orientation": self.orientation,
            "weight": self.weight,
            "normalization": self.normalization,
            "bounds": list(self.bounds) if self.bounds else None,
            "unit": self.unit,
        }

        # Add banded objective fields if present
        if self.band is not None:
            band_dict: dict[str, Any] = {}
            if self.band.low is not None and self.band.high is not None:
                band_dict["target"] = [self.band.low, self.band.high]
            if self.band.center is not None and self.band.tol is not None:
                band_dict["center"] = self.band.center
                band_dict["tol"] = self.band.tol
            band_dict["test"] = self.band_test
            band_dict["alpha"] = self.band_alpha
            result["band"] = band_dict

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveDefinition:
        """Create from dictionary representation."""
        bounds = tuple(data["bounds"]) if data.get("bounds") else None

        # Handle banded objective
        band = None
        band_test: Literal["TOST"] = "TOST"
        band_alpha = 0.05

        if "band" in data:
            # Import here to avoid circular imports
            from traigent.tvl.models import BandTarget

            band_data = data["band"]
            if isinstance(band_data, dict):
                band = BandTarget.from_dict(band_data.get("target", band_data))
                band_test = band_data.get("test", "TOST")
                band_alpha = float(band_data.get("alpha", 0.05))

        # Determine orientation
        orientation = data.get("orientation", "maximize")
        if band is not None and orientation not in ["band"]:
            orientation = "band"

        return cls(
            name=data["name"],
            orientation=orientation,
            weight=float(data["weight"]) if data.get("weight") is not None else 1.0,
            normalization=data.get("normalization", "min_max"),
            bounds=bounds,
            unit=data.get("unit"),
            band=band,
            band_test=band_test,
            band_alpha=band_alpha,
        )


@dataclass
class ObjectiveSchema:
    """Complete schema for multi-objective optimization.

    Attributes:
        objectives: List of objective definitions
        weights_sum: Sum of all objective weights
        weights_normalized: Normalized weights (sum to 1.0)
        schema_version: Version of the objective schema
    """

    objectives: list[ObjectiveDefinition]
    weights_sum: float
    weights_normalized: dict[str, float]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate schema after initialization."""
        if not self.objectives:
            raise ValueError("At least one objective must be defined")

        # Check for duplicate objective names
        names = [obj.name for obj in self.objectives]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate objective names found: {set(duplicates)}")

        # Validate weights_sum matches actual sum
        actual_sum = sum(obj.weight for obj in self.objectives)
        if abs(self.weights_sum - actual_sum) > 1e-10:
            raise ValueError(
                f"weights_sum ({self.weights_sum}) doesn't match "
                f"actual sum of weights ({actual_sum})"
            )

        # Validate normalized weights
        if self.weights_sum > 0:
            for obj in self.objectives:
                expected_normalized = obj.weight / self.weights_sum
                actual_normalized = self.weights_normalized.get(obj.name, 0)
                if abs(expected_normalized - actual_normalized) > 1e-10:
                    raise ValueError(
                        f"Normalized weight for '{obj.name}' is incorrect. "
                        f"Expected {expected_normalized}, got {actual_normalized}"
                    )

    @classmethod
    def from_objectives(
        cls, objectives: list[ObjectiveDefinition], schema_version: str = "1.0.0"
    ) -> ObjectiveSchema:
        """Create schema from list of objectives with automatic weight normalization.

        Args:
            objectives: List of objective definitions
            schema_version: Version of the schema

        Returns:
            ObjectiveSchema with normalized weights
        """
        if not objectives:
            raise ValueError("At least one objective must be provided")

        # Calculate weights sum
        weights_sum = sum(obj.weight for obj in objectives)

        if weights_sum <= 0:
            raise ValueError(f"Sum of weights must be positive, got {weights_sum}")

        # Calculate normalized weights
        weights_normalized = {obj.name: obj.weight / weights_sum for obj in objectives}

        # Multi-objective guard: no single objective may hold 100%
        if len(objectives) > 1:
            for obj in objectives:
                nw = weights_normalized[obj.name]
                if nw > 1.0 - 1e-9:
                    raise ValueError(
                        f"In a multi-objective schema, no single "
                        f"objective can have 100% of the weight. "
                        f"Objective '{obj.name}' has normalized "
                        f"weight {nw:.6f}. Adjust weights so all "
                        f"objectives contribute."
                    )

        # Log when weights are re-scaled
        if abs(weights_sum - 1.0) > 1e-9:
            orig = ", ".join(f"{o.name}={o.weight}" for o in objectives)
            normed = ", ".join(
                f"{o.name}={o.weight / weights_sum:.4f}" for o in objectives
            )
            logger.info(
                "Objective weights [%s] normalized to [%s] (sum=%.4f -> 1.0)",
                orig,
                normed,
                weights_sum,
            )

        return cls(
            objectives=objectives,
            weights_sum=weights_sum,
            weights_normalized=weights_normalized,
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for JSON serialization."""
        return {
            "objectives": [obj.to_dict() for obj in self.objectives],
            "weights_sum": self.weights_sum,
            "weights_normalized": self.weights_normalized,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveSchema:
        """Create from dictionary representation."""
        objectives = [
            ObjectiveDefinition.from_dict(obj_data) for obj_data in data["objectives"]
        ]
        return cls(
            objectives=objectives,
            weights_sum=data["weights_sum"],
            weights_normalized=data["weights_normalized"],
            schema_version=data.get("schema_version", "1.0.0"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ObjectiveSchema:
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_objective(self, name: str) -> ObjectiveDefinition | None:
        """Get objective by name.

        Args:
            name: Name of the objective

        Returns:
            ObjectiveDefinition if found, None otherwise
        """
        for obj in self.objectives:
            if obj.name == name:
                return obj
        return None

    def get_orientation(self, name: str) -> str | None:
        """Get orientation for an objective.

        Args:
            name: Name of the objective

        Returns:
            "maximize" or "minimize" if found, None otherwise
        """
        obj = self.get_objective(name)
        return obj.orientation if obj else None

    def get_normalized_weight(self, name: str) -> float:
        """Get normalized weight for an objective.

        Args:
            name: Name of the objective

        Returns:
            Normalized weight (0.0 if not found)
        """
        return self.weights_normalized.get(name, 0.0)

    def normalize_value(
        self,
        objective_name: str,
        value: float,
        min_val: float | None = None,
        max_val: float | None = None,
        epsilon: float = 1e-10,
    ) -> float:
        """Normalize a value based on objective orientation.

        Args:
            objective_name: Name of the objective
            value: Value to normalize
            min_val: Minimum value (if None, uses bounds if available)
            max_val: Maximum value (if None, uses bounds if available)
            epsilon: Small value to handle zero-range cases

        Returns:
            Normalized value in [0, 1] range
        """
        obj = self.get_objective(objective_name)
        if obj is None:
            raise ValueError(f"Objective '{objective_name}' not found")

        # Use provided bounds or objective's bounds
        if min_val is None or max_val is None:
            if obj.bounds:
                min_val = min_val or obj.bounds[0]
                max_val = max_val or obj.bounds[1]
            else:
                raise ValueError(
                    f"No bounds provided for normalization of '{objective_name}'"
                )

        # Handle zero-range case
        if abs(max_val - min_val) < epsilon:
            return 0.5  # Return middle value for constant objectives

        # Normalize based on orientation
        if obj.orientation == "maximize":
            # For maximize: 0 = worst (min), 1 = best (max)
            normalized = (value - min_val) / (max_val - min_val)
        else:  # minimize
            # For minimize: 0 = worst (max), 1 = best (min)
            normalized = (max_val - value) / (max_val - min_val)

        # Clip to [0, 1] range to handle values outside bounds
        return max(0.0, min(1.0, normalized))

    def normalize_metrics(
        self,
        metrics: dict[str, float],
        ranges: dict[str, tuple[float, float]] | None = None,
        epsilon: float = 1e-10,
    ) -> dict[str, float]:
        """Normalize all metrics based on their orientations.

        Args:
            metrics: Dictionary of metric values
            ranges: Optional dictionary of (min, max) ranges for each metric
            epsilon: Small value to handle zero-range cases

        Returns:
            Dictionary of normalized values
        """
        normalized = {}

        for obj in self.objectives:
            if obj.name not in metrics:
                continue

            value = metrics[obj.name]
            if value is None:
                continue

            # Get range for this objective
            if ranges and obj.name in ranges:
                min_val, max_val = ranges[obj.name]
            elif obj.bounds:
                min_val, max_val = obj.bounds
            else:
                # Skip if no range available
                continue

            normalized[obj.name] = self.normalize_value(
                obj.name, value, min_val, max_val, epsilon
            )

        return normalized

    def validate_metrics(self, metrics: dict[str, float]) -> list[str]:
        """Validate that metrics match defined objectives.

        Args:
            metrics: Dictionary of metric values

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for missing objectives
        objective_names = {obj.name for obj in self.objectives}
        for name in objective_names:
            if name not in metrics:
                errors.append(f"Missing metric for objective '{name}'")

        # Check for extra metrics
        for name in metrics:
            if name not in objective_names:
                errors.append(f"Metric '{name}' is not a defined objective")

        # Check bounds if defined
        for obj in self.objectives:
            if obj.name in metrics and obj.bounds:
                value = metrics[obj.name]
                min_val, max_val = obj.bounds
                if value < min_val or value > max_val:
                    errors.append(
                        f"Metric '{obj.name}' value {value} is outside bounds [{min_val}, {max_val}]"
                    )

        return errors

    def compute_weighted_score(self, metrics: dict[str, float]) -> float | None:
        """Compute a single weighted score from multiple objectives.

        For maximize objectives, higher values contribute positively.
        For minimize objectives, lower values contribute positively (inverted).

        Args:
            metrics: Dictionary of metric values

        Returns:
            Weighted score (higher is better), or None if required metrics missing
        """
        return self.compute_aggregated_score(metrics, AggregationMode.WEIGHTED_SUM)

    def compute_aggregated_score(
        self,
        metrics: dict[str, float],
        mode: AggregationMode = AggregationMode.WEIGHTED_SUM,
        reference_point: dict[str, float] | None = None,
    ) -> float | None:
        """Compute an aggregated score using the specified aggregation mode.

        This method supports multiple aggregation strategies for multi-objective
        optimization, matching capabilities found in frameworks like NeMo Optimizer.

        Args:
            metrics: Dictionary of metric values
            mode: Aggregation mode (WEIGHTED_SUM, HARMONIC, or CHEBYSHEV)
            reference_point: Reference/ideal point for Chebyshev (optional).
                            If not provided, uses 1.0 for maximize, 0.0 for minimize.

        Returns:
            Aggregated score (higher is better), or None if required metrics missing

        Examples::

            schema = ObjectiveSchema.from_objectives([...])
            metrics = {"accuracy": 0.9, "cost": 0.05}

            # Weighted sum (default, linear trade-offs)
            schema.compute_aggregated_score(metrics, AggregationMode.WEIGHTED_SUM)

            # Harmonic mean (penalizes imbalanced solutions)
            schema.compute_aggregated_score(metrics, AggregationMode.HARMONIC)

            # Chebyshev/minimax (optimizes worst-case performance)
            schema.compute_aggregated_score(metrics, AggregationMode.CHEBYSHEV)
        """
        if not metrics:
            return None

        # Collect normalized values for each objective
        # (name, normalized_value (higher is better), weight)
        normalized_values: list[tuple[str, float, float]] = []

        for obj in self.objectives:
            if obj.name not in metrics:
                continue

            value = metrics[obj.name]
            if value is None or not math.isfinite(value):
                continue

            weight = self.weights_normalized.get(obj.name, 0.0)
            if weight <= 0:
                continue

            normalized = self._normalize_value_for_aggregation(
                obj, value, reference_point
            )
            if normalized is None:
                continue

            normalized_values.append((obj.name, normalized, weight))

        if not normalized_values:
            return None

        if mode == AggregationMode.WEIGHTED_SUM:
            return self._compute_weighted_sum(normalized_values)
        elif mode == AggregationMode.HARMONIC:
            return self._compute_harmonic(normalized_values)
        elif mode == AggregationMode.CHEBYSHEV:
            return self._compute_chebyshev(normalized_values, metrics, reference_point)
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

    def _compute_weighted_sum(
        self, normalized_values: list[tuple[str, float, float]]
    ) -> float:
        """Compute weighted sum aggregation.

        Simple linear combination of weighted objectives.

        Args:
            normalized_values: List of (name, normalized_value, weight) tuples

        Returns:
            Weighted sum score
        """
        return sum(value * weight for _, value, weight in normalized_values)

    def _compute_harmonic(
        self, normalized_values: list[tuple[str, float, float]]
    ) -> float:
        """Compute weighted harmonic mean aggregation.

        Harmonic mean penalizes solutions where one objective is very poor.
        This encourages balanced trade-offs between objectives.

        Formula: n / sum(1/x_i) for unweighted, or sum(w_i) / sum(w_i/x_i) for weighted

        Args:
            normalized_values: List of (name, normalized_value, weight) tuples

        Returns:
            Harmonic mean score (returns 0.0 if any value is zero or negative)
        """
        positive_values = []
        for name, value, weight in normalized_values:
            if value <= 0:
                return 0.0  # Harmonic mean undefined for non-positive values
            positive_values.append((name, value, weight))

        weight_sum = sum(w for _, _, w in positive_values)
        if weight_sum <= 0:
            return 0.0

        weighted_reciprocal_sum = sum(
            weight / value for _, value, weight in positive_values
        )

        if weighted_reciprocal_sum <= 0:
            return 0.0

        return weight_sum / weighted_reciprocal_sum

    def _compute_chebyshev(
        self,
        normalized_values: list[tuple[str, float, float]],
        metrics: dict[str, float],
        reference_point: dict[str, float] | None = None,
    ) -> float:
        """Compute Chebyshev (minimax) aggregation.

        Minimizes the maximum weighted distance from the reference point.
        This optimizes the worst-performing objective.

        Formula: -max(w_i * |x_i - ref_i|) (negated so higher is better)

        Args:
            normalized_values: List of (name, normalized_value, weight) tuples
            metrics: Original metrics (for reference point calculation)
            reference_point: Ideal point for each objective. If not provided,
                            uses 1.0 for maximize objectives, 0.0 for minimize.

        Returns:
            Negative max weighted distance (higher/less negative is better)
        """
        max_weighted_distance = 0.0

        for name, normalized_value, weight in normalized_values:
            obj = self.get_objective(name)
            if obj is None:
                continue

            # Determine reference point for this objective
            if reference_point and name in reference_point:
                raw_ref = reference_point[name]
                normalized_ref = self._normalize_value_for_aggregation(
                    obj, raw_ref, reference_point
                )
                if normalized_ref is None:
                    ref = 1.0
                else:
                    ref = normalized_ref
            else:
                # Default ideal in normalized space: 1.0 represents "best"
                ref = 1.0

            # Compute weighted distance from reference
            distance = abs(normalized_value - ref)
            weighted_distance = weight * distance

            max_weighted_distance = max(max_weighted_distance, weighted_distance)

        # Return negative so that smaller distances (better) give higher scores
        return -max_weighted_distance

    def _normalize_value_for_aggregation(
        self,
        obj: ObjectiveDefinition,
        value: float,
        reference_point: dict[str, float] | None,
    ) -> float | None:
        """Normalize raw metric value so higher is always better and stays positive."""
        if not math.isfinite(value):
            return None

        if obj.orientation == "maximize":
            return max(value, 0.0)

        # Minimize objectives -> map smaller values to higher normalized scores
        ref_value = None
        if reference_point and obj.name in reference_point:
            ref_value = reference_point[obj.name]

        baseline = None
        if ref_value is not None and math.isfinite(ref_value) and ref_value > 0:
            baseline = max(ref_value, 1e-9)

        if baseline is None:
            # Default: inverse scaling to keep values in (0, 1]
            return 1.0 / (1.0 + max(value, 0.0))

        return baseline / (baseline + max(value, 0.0))


def create_default_objectives(
    objective_names: list[str],
    orientations: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
) -> ObjectiveSchema:
    """Create an objective schema with defaults.

    Args:
        objective_names: List of objective names
        orientations: Optional dict of orientations (defaults to maximize)
        weights: Optional dict of weights (defaults to equal weights)

    Returns:
        ObjectiveSchema with the specified objectives
    """
    if not objective_names:
        raise ValueError("At least one objective name must be provided")

    # Default orientations (maximize for common metrics)
    default_orientations = {
        "accuracy": "maximize",
        "precision": "maximize",
        "recall": "maximize",
        "f1": "maximize",
        "cost": "minimize",
        "latency": "minimize",
        "error": "minimize",
        "loss": "minimize",
        "time": "minimize",
        "memory": "minimize",
    }

    # Build objectives
    objectives = []
    for name in objective_names:
        # Get orientation
        orientation: Literal["maximize", "minimize"]
        if orientations and name in orientations:
            orientation = orientations[name]  # type: ignore[assignment]
        elif name in default_orientations:
            orientation = default_orientations[name]  # type: ignore[assignment]
        else:
            orientation = "maximize"  # Default to maximize

        # Get weight
        if weights and name in weights:
            weight = weights[name]
        else:
            weight = 1.0  # Equal weights by default

        objectives.append(
            ObjectiveDefinition(name=name, orientation=orientation, weight=weight)
        )

    return ObjectiveSchema.from_objectives(objectives)


def normalize_objectives(
    objectives: list[str] | ObjectiveSchema | Sequence[str] | None,
) -> ObjectiveSchema | None:
    """Convert objectives input to an ObjectiveSchema.

    This function normalizes various objective input formats into a consistent
    ObjectiveSchema for use throughout the optimization process.

    Args:
        objectives: Objectives in one of the following formats:
            - list[str]: List of objective names (creates default schema)
            - ObjectiveSchema: Returns as-is
            - None: Returns None

    Returns:
        ObjectiveSchema if objectives were provided, None otherwise
    """
    if objectives is None:
        return None

    if isinstance(objectives, ObjectiveSchema):
        return objectives

    if isinstance(objectives, list):
        if not objectives:
            return None
        # Convert list of strings to ObjectiveSchema with defaults
        return create_default_objectives(objectives)

    raise TypeError(
        f"objectives must be list[str], ObjectiveSchema, or None, "
        f"got {type(objectives).__name__}"
    )


def schema_to_objective_names(schema: ObjectiveSchema | None) -> list[str]:
    """Extract objective names from an ObjectiveSchema.

    Args:
        schema: ObjectiveSchema to extract names from, or None

    Returns:
        List of objective names (empty list if schema is None)
    """
    if schema is None:
        return []

    return [obj.name for obj in schema.objectives]
