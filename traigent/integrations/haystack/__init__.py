"""Haystack Pipeline Integration for Traigent.

This module provides pipeline introspection and optimization capabilities
for Haystack AI pipelines, using TVL (Tuning Variable Language) terminology.

Terminology (aligned with TVL):
- TVAR (Tuned Variable): A parameter that can be optimized
- DiscoveredTVAR: A TVAR discovered through pipeline introspection
- TVARScope: A namespace/container for TVARs (maps to Haystack Component)
- PipelineSpec: The complete discovered pipeline structure with all scopes
- Connection: A data flow edge between TVARScopes
- ExplorationSpace: Optimization search space with TVARs and constraints
- CategoricalConstraint: Constraint for discrete choice parameters
- NumericalConstraint: Constraint for numerical parameters with bounds

Example usage:
    from traigent.integrations.haystack import from_pipeline, PipelineSpec

    pipeline = Pipeline()
    # ... add components ...

    # For discovery (pipeline structure):
    pipeline_spec = from_pipeline(pipeline)
    print(pipeline_spec.scopes)  # List of discovered TVARScopes
    print(pipeline_spec.connections)  # List of data flow connections

    # For optimization (search space):
    space = from_pipeline(pipeline, as_exploration_space=True)
    print(space.tunable_tvars)  # Dict of tunable TVARs with constraints

Backwards Compatibility:
    The old names (ConfigSpace, Component, Edge, Parameter) are still available
    as aliases for the new names to support existing code during migration.
"""

from __future__ import annotations

from .configspace import (
    TVAR,
    CategoricalConstraint,
    ConditionalConstraint,
    Configuration,
    ExplorationSpace,
    NumericalConstraint,
    TVARConstraint,
)
from .introspection import (
    PARAMETER_SEMANTICS,
    TVAR_SEMANTICS,
    from_pipeline,
)
from .models import (  # New TVL-aligned types (preferred); Backwards-compatible aliases (deprecated)
    Component,
    ConfigSpace,
    Connection,
    DiscoveredTVAR,
    Edge,
    Parameter,
    PipelineSpec,
    TVARScope,
)

__all__ = [
    # Core function
    "from_pipeline",
    # New TVL-aligned types (preferred)
    "PipelineSpec",
    "TVARScope",
    "DiscoveredTVAR",
    "Connection",
    "TVAR_SEMANTICS",
    # Optimization types (Epic 2) - aligned with TVL Glossary v2.0
    "ExplorationSpace",  # 𝒳 - feasible configuration set
    "TVAR",  # tᵢ - tuned variable
    "TVARConstraint",  # Domain constraint (Dᵢ)
    "CategoricalConstraint",  # C^str for discrete choices
    "NumericalConstraint",  # C^str for numerical ranges
    "ConditionalConstraint",  # C^str for inter-TVAR dependencies
    "Configuration",  # θ - assignment to all TVARs
    # Backwards-compatible aliases (deprecated, for migration)
    "ConfigSpace",
    "Component",
    "Edge",
    "Parameter",
    "PARAMETER_SEMANTICS",
]
