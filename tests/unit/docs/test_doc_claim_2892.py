"""Doc claim matches code (#2892)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_constraint_dsl_guide_does_not_limit_set_membership_to_choices_2892() -> None:
    from traigent.api.constraint_builders import (
        CategoricalConstraintBuilderMixin,
        NumericConstraintBuilderMixin,
    )

    for mixin in (NumericConstraintBuilderMixin, CategoricalConstraintBuilderMixin):
        assert hasattr(mixin, "is_in") and hasattr(mixin, "not_in")
    for line in _read("docs/features/constraint-dsl.md").splitlines():
        if line.startswith(("| `.is_in(", "| `.not_in(")):
            assert "Choices only" not in line, line
