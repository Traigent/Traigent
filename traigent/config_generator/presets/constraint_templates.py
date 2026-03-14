"""Structural constraint templates triggered by TVAR combinations.

Each template defines a set of required TVARs and a builder function
that produces a human-readable description + Python constraint code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from traigent.config_generator.types import StructuralConstraintSpec


@dataclass(frozen=True)
class ConstraintTemplate:
    """A template for a structural constraint between TVars."""

    name: str
    description: str
    requires_tvars: frozenset[str]

    def matches(self, tvar_names: set[str]) -> bool:
        """Check if this template's required TVars are all present."""
        return self.requires_tvars.issubset(tvar_names)


def get_matching_constraints(tvar_names: set[str]) -> list[StructuralConstraintSpec]:
    """Return all structural constraints whose required TVars are present."""
    results: list[StructuralConstraintSpec] = []
    for template in _CONSTRAINT_TEMPLATES:
        if template.matches(tvar_names):
            spec = _BUILDERS[template.name](template, tvar_names)
            if spec is not None:
                results.append(spec)
    return results


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

_CONSTRAINT_TEMPLATES: list[ConstraintTemplate] = [
    ConstraintTemplate(
        name="temperature_top_p_stability",
        description="Avoid extreme values of both temperature and top_p simultaneously",
        requires_tvars=frozenset({"temperature", "top_p"}),
    ),
    ConstraintTemplate(
        name="model_temperature_conservative",
        description="Conservative temperature for larger/more expensive models",
        requires_tvars=frozenset({"model", "temperature"}),
    ),
    ConstraintTemplate(
        name="model_max_tokens_scaling",
        description="Scale max_tokens appropriately with model capability",
        requires_tvars=frozenset({"model", "max_tokens"}),
    ),
    ConstraintTemplate(
        name="chunk_size_overlap_ratio",
        description="Chunk overlap should not exceed chunk size",
        requires_tvars=frozenset({"chunk_size", "chunk_overlap"}),
    ),
]


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def _build_temp_top_p(
    template: ConstraintTemplate, tvar_names: set[str]
) -> StructuralConstraintSpec:
    return StructuralConstraintSpec(
        description=template.description,
        constraint_code=(
            "# Avoid numerically unstable extreme combinations\n"
            "require(~(temperature.gte(1.5) & top_p.lte(0.2)))"
        ),
        requires_tvars=tuple(sorted(template.requires_tvars)),
        source="template",
        reasoning=(
            "Very high temperature with very low top_p produces "
            "unpredictable, low-quality outputs"
        ),
    )


def _build_model_temperature(
    template: ConstraintTemplate, tvar_names: set[str]
) -> StructuralConstraintSpec:
    return StructuralConstraintSpec(
        description=template.description,
        constraint_code=(
            "# Use conservative temperature for expensive models\n"
            "implies(model.is_in(['gpt-4o', 'gpt-4.1']), temperature.lte(1.0))"
        ),
        requires_tvars=tuple(sorted(template.requires_tvars)),
        source="template",
        reasoning=(
            "Expensive models are often used for precision tasks; "
            "high temperature wastes budget on noisy outputs"
        ),
    )


def _build_model_max_tokens(
    template: ConstraintTemplate, tvar_names: set[str]
) -> StructuralConstraintSpec:
    return StructuralConstraintSpec(
        description=template.description,
        constraint_code=(
            "# Ensure mini models aren't given too many tokens\n"
            "implies(model.is_in(['gpt-4o-mini', 'gpt-4.1-mini']), max_tokens.lte(2048))"
        ),
        requires_tvars=tuple(sorted(template.requires_tvars)),
        source="template",
        reasoning=(
            "Smaller models have lower context windows and may produce "
            "lower quality output with excessive token budgets"
        ),
    )


def _build_chunk_overlap(
    template: ConstraintTemplate, tvar_names: set[str]
) -> StructuralConstraintSpec:
    return StructuralConstraintSpec(
        description=template.description,
        constraint_code=(
            "# Overlap must be less than chunk size\n"
            "lambda config: config['chunk_overlap'] < config['chunk_size']"
        ),
        requires_tvars=tuple(sorted(template.requires_tvars)),
        source="template",
        reasoning="Chunk overlap exceeding chunk size creates redundant/empty chunks",
    )


_BUILDERS: dict[str, Any] = {
    "temperature_top_p_stability": _build_temp_top_p,
    "model_temperature_conservative": _build_model_temperature,
    "model_max_tokens_scaling": _build_model_max_tokens,
    "chunk_size_overlap_ratio": _build_chunk_overlap,
}
