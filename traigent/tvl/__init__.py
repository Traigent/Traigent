"""TVL integration helpers for the Traigent SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-TVLSPEC REQ-TVLSPEC-012 SYNC-OptimizationFlow

from __future__ import annotations

from .models import (
    BandTarget,
    ChanceConstraint,
    DerivedConstraint,
    DomainSpec,
    PromotionPolicy,
    RegistryResolver,
    StructuralConstraint,
    TVarDecl,
    normalize_tvar_type,
    parse_domain_spec,
)
from .objectives import (
    BandedComparisonResult,
    BandedObjectiveSpec,
    TOSTResult,
    band_deviation,
    compare_banded_objectives,
    compare_banded_with_tost,
    is_in_band,
    tost_equivalence_test,
)
from .options import TVLOptions
from .promotion_gate import (
    ChanceConstraintResult,
    ObjectiveResult,
    ObjectiveSpec,
    PromotionDecision,
    PromotionGate,
)
from .registry import (
    DictRegistryResolver,
    FileRegistryResolver,
)
from .spec_loader import (
    TVLBudget,
    TVLSpecArtifact,
    compile_constraint_expression,
    load_tvl_spec,
)
from .spec_validator import (
    DriftSeverity,
    SpecDriftIssue,
    SpecDriftReport,
    validate_spec_code_alignment,
    validate_tvar_types_match,
)
from .statistics import (
    PairedComparisonResult,
    benjamini_hochberg_adjust,
    clopper_pearson_lower_bound,
    hypervolume_improvement,
    paired_comparison_test,
)

__all__ = [
    # Core spec loading
    "TVLBudget",
    "TVLSpecArtifact",
    "TVLOptions",
    "compile_constraint_expression",
    "load_tvl_spec",
    # TVL 0.9 models
    "BandTarget",
    "ChanceConstraint",
    "DerivedConstraint",
    "DomainSpec",
    "PromotionPolicy",
    "RegistryResolver",
    "StructuralConstraint",
    "TVarDecl",
    "normalize_tvar_type",
    "parse_domain_spec",
    # Banded objectives & TOST
    "BandedComparisonResult",
    "BandedObjectiveSpec",
    "TOSTResult",
    "band_deviation",
    "compare_banded_objectives",
    "compare_banded_with_tost",
    "is_in_band",
    "tost_equivalence_test",
    # Statistics
    "PairedComparisonResult",
    "benjamini_hochberg_adjust",
    "clopper_pearson_lower_bound",
    "hypervolume_improvement",
    "paired_comparison_test",
    # Promotion gate
    "ChanceConstraintResult",
    "ObjectiveResult",
    "ObjectiveSpec",
    "PromotionDecision",
    "PromotionGate",
    # Registry resolvers
    "DictRegistryResolver",
    "FileRegistryResolver",
    # Spec validation
    "DriftSeverity",
    "SpecDriftIssue",
    "SpecDriftReport",
    "validate_spec_code_alignment",
    "validate_tvar_types_match",
]
