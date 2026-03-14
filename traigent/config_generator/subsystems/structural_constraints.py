"""Subsystem 5: Generate structural constraints between TVars.

Uses template matching based on which TVars are present, plus optional
LLM enrichment for domain-specific constraints.
"""

from __future__ import annotations

from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM
from traigent.config_generator.presets.constraint_templates import (
    get_matching_constraints,
)
from traigent.config_generator.types import StructuralConstraintSpec, TVarSpec


def generate_structural_constraints(
    tvars: list[TVarSpec],
    *,
    llm: ConfigGenLLM | None = None,
    source_code: str = "",
) -> list[StructuralConstraintSpec]:
    """Generate structural constraints between TVars.

    Parameters
    ----------
    tvars:
        Resolved TVarSpecs from subsystem 1.
    llm:
        Optional LLM for enrichment.
    source_code:
        Original source code for LLM context.
    """
    tvar_names = {t.name for t in tvars}

    # Template-based constraints
    constraints = get_matching_constraints(tvar_names)

    # LLM enrichment
    if llm is not None and len(tvars) >= 2:
        llm_constraints = _llm_suggest_constraints(tvars, llm, source_code)
        constraints.extend(llm_constraints)

    return constraints


def _extract_json_text(response: str) -> str:
    """Extract JSON array text from a possibly markdown-wrapped response."""
    text = response.strip()
    if "```" not in text:
        return text
    for part in text.split("```"):
        stripped = part.strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.startswith("["):
            return stripped
    return text


def _parse_constraint_item(item: dict) -> StructuralConstraintSpec | None:
    """Parse a single LLM-suggested constraint, or *None* if invalid."""
    desc = item.get("description", "")
    code = item.get("constraint_code", "")
    if not desc or not code:
        return None
    return StructuralConstraintSpec(
        description=desc,
        constraint_code=code,
        source="llm",
        reasoning=item.get("reasoning", ""),
    )


def _llm_suggest_constraints(
    tvars: list[TVarSpec],
    llm: ConfigGenLLM,
    source_code: str,
) -> list[StructuralConstraintSpec]:
    """Ask LLM to suggest structural constraints."""
    import json

    tvar_summary = "\n".join(f"  - {t.name}: {t.to_range_code()}" for t in tvars)
    prompt = (
        "You are configuring structural constraints between tunable variables "
        "in an LLM agent optimization.\n\n"
        f"Tunable variables:\n{tvar_summary}\n\n"
        "Source code:\n"
        f"```python\n{source_code[:2000]}\n```\n\n"
        "Suggest 0-2 structural constraints (relationships or dependencies) "
        "between these variables. Only suggest constraints that are clearly "
        "meaningful for this specific code.\n\n"
        "Reply with ONLY a JSON array:\n"
        '[{"description": "...", "constraint_code": "...", "reasoning": "..."}]\n'
        "constraint_code should be a Python expression using implies(), require(), "
        "or a lambda function.\n"
        "Return an empty array [] if no constraints are needed.\n"
    )

    try:
        response = llm.complete(prompt, max_tokens=512)
    except BudgetExhausted:
        return []

    text = _extract_json_text(response)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [c for item in data if (c := _parse_constraint_item(item)) is not None]
