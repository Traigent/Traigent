"""Subsystem 6: TVAR recommendations.

Suggests additional tunable variables the user may not have considered,
based on agent type and what TVARs are already detected.
"""

from __future__ import annotations

from typing import Any

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.catalog import catalog_entries, entry_to_recommendation
from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM
from traigent.config_generator.presets.range_presets import get_preset_range
from traigent.config_generator.types import TVarRecommendation, TVarSpec
from traigent.utils.llm_response_parsing import extract_json_array_text


def generate_recommendations(
    tvars: list[TVarSpec],
    *,
    llm: ConfigGenLLM | None = None,
    source_code: str = "",
    classification: ClassificationResult | None = None,
) -> list[TVarRecommendation]:
    """Generate TVAR recommendations.

    Parameters
    ----------
    tvars:
        Already-detected TVarSpecs.
    llm:
        Optional LLM for enrichment.
    source_code:
        Original source code.
    classification:
        Pre-computed agent classification.
    """
    existing_names = {t.name for t in tvars}
    agent_type = classification.agent_type if classification else "general_llm"

    # Preset recommendations
    recs = _preset_recommendations(agent_type, existing_names)

    # LLM enrichment
    if llm is not None:
        llm_recs = _llm_recommend(tvars, agent_type, source_code, llm)
        # Deduplicate by name
        existing_rec_names = {r.name for r in recs}
        for rec in llm_recs:
            if rec.name not in existing_rec_names and rec.name not in existing_names:
                recs.append(rec)
                existing_rec_names.add(rec.name)

    return recs


# ---------------------------------------------------------------------------
# Preset recommendation catalog
# ---------------------------------------------------------------------------

_LEGACY_RECOMMENDATION_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "rag": [
        {
            "name": "prompting_strategy",
            "category": "prompting",
            "reasoning": "RAG systems benefit from different prompting strategies (direct, chain_of_thought, react)",
            "impact": "high",
        },
        {
            "name": "retriever",
            "category": "retrieval",
            "reasoning": "Different retrieval methods (similarity, MMR, BM25, hybrid) affect answer quality",
            "impact": "high",
        },
        {
            "name": "chunk_size",
            "category": "retrieval",
            "reasoning": "Document chunk size significantly impacts retrieval quality",
            "impact": "medium",
        },
        {
            "name": "reranker",
            "category": "retrieval",
            "reasoning": "Reranking retrieved documents can improve answer quality",
            "impact": "medium",
        },
        {
            "name": "context_format",
            "category": "prompting",
            "reasoning": "How retrieved context is formatted affects LLM comprehension",
            "impact": "medium",
        },
    ],
    "chat": [
        {
            "name": "prompting_strategy",
            "category": "prompting",
            "reasoning": "Different prompting strategies can dramatically change response quality",
            "impact": "high",
        },
        {
            "name": "context_format",
            "category": "prompting",
            "reasoning": "How context is formatted affects LLM comprehension",
            "impact": "medium",
        },
    ],
    "code_gen": [
        {
            "name": "prompting_strategy",
            "category": "prompting",
            "reasoning": "Chain-of-thought and self-consistency improve code generation accuracy",
            "impact": "high",
        },
    ],
    "summarization": [
        {
            "name": "prompting_strategy",
            "category": "prompting",
            "reasoning": "Different prompting strategies produce different summary styles",
            "impact": "medium",
        },
        {
            "name": "context_format",
            "category": "prompting",
            "reasoning": "Document formatting affects summarization quality",
            "impact": "low",
        },
    ],
    "classification": [
        {
            "name": "few_shot_count",
            "category": "prompting",
            "reasoning": "Number of few-shot examples significantly impacts classification accuracy",
            "impact": "high",
        },
        {
            "name": "prompting_strategy",
            "category": "prompting",
            "reasoning": "Chain-of-thought can improve complex classification tasks",
            "impact": "medium",
        },
    ],
    "general_llm": [
        {
            "name": "prompting_strategy",
            "category": "prompting",
            "reasoning": "Different prompting strategies can improve output quality",
            "impact": "medium",
        },
    ],
}

_RECOMMENDATION_ORDER: dict[str, tuple[str, ...]] = {
    "rag": (
        "prompting_strategy",
        "retriever",
        "retrieval_k",
        "chunk_size",
        "reranker",
        "context_format",
    ),
    "chat": ("prompting_strategy", "context_format"),
    "code_gen": (
        "prompting_strategy",
        "schema_context",
        "evidence_usage",
        "fewshot_selector",
        "generation_path",
        "fewshot_k",
        "candidate_count",
        "repair_policy",
    ),
    "summarization": ("prompting_strategy", "context_format"),
    "classification": ("few_shot_count", "prompting_strategy"),
    "general_llm": ("prompting_strategy",),
}


def _catalog_recommendation_templates(agent_type: str) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for entry in catalog_entries(agent_type):
        rec = entry_to_recommendation(entry)
        templates.append(
            {
                "name": rec.name,
                "category": rec.category,
                "reasoning": rec.reasoning,
                "impact": rec.impact_estimate,
                "entry_id": rec.entry_id,
                "evidence_refs": rec.evidence_refs,
                "apply_guidance": rec.apply_guidance,
            }
        )
    return templates


def _recommendation_templates(agent_type: str) -> list[dict[str, Any]]:
    templates = [
        dict(template)
        for template in _LEGACY_RECOMMENDATION_TEMPLATES.get(agent_type, ())
    ]
    templates.extend(_catalog_recommendation_templates(agent_type))

    order = _RECOMMENDATION_ORDER.get(agent_type, ())
    if not order:
        return templates

    order_index = {name: index for index, name in enumerate(order)}
    return sorted(
        templates,
        key=lambda template: order_index.get(str(template.get("name", "")), len(order)),
    )


_RECOMMENDATIONS: dict[str, list[dict[str, Any]]] = {
    agent_type: _recommendation_templates(agent_type)
    for agent_type in _RECOMMENDATION_ORDER
}


def _preset_recommendations(
    agent_type: str,
    existing_names: set[str],
) -> list[TVarRecommendation]:
    """Generate recommendations from presets."""
    templates = _recommendation_templates(agent_type)
    recs: list[TVarRecommendation] = []

    for tmpl in templates:
        name = tmpl["name"]
        if name in existing_names:
            continue

        preset = get_preset_range(name)
        if preset is None:
            continue

        recs.append(
            TVarRecommendation(
                name=name,
                range_type=preset["range_type"],
                range_kwargs=dict(preset["kwargs"]),
                category=tmpl["category"],
                reasoning=tmpl["reasoning"],
                impact_estimate=tmpl["impact"],
                entry_id=tmpl.get("entry_id", ""),
                evidence_refs=tuple(tmpl.get("evidence_refs", ())),
                apply_guidance=tmpl.get("apply_guidance", ""),
            )
        )

    return recs


_VALID_RANGE_TYPES = frozenset({"Range", "IntRange", "LogRange", "Choices"})


def _parse_recommendation_item(item: dict) -> TVarRecommendation | None:
    """Parse a single LLM-suggested recommendation, or *None* if invalid."""
    name = item.get("name", "")
    range_type = item.get("range_type", "")
    kwargs = item.get("kwargs", {})
    if not name or not range_type or not isinstance(kwargs, dict):
        return None
    if range_type not in _VALID_RANGE_TYPES:
        return None
    return TVarRecommendation(
        name=name,
        range_type=range_type,
        range_kwargs=kwargs,
        category=item.get("category", ""),
        reasoning=item.get("reasoning", "LLM-suggested"),
        impact_estimate=item.get("impact", "medium"),
    )


def _llm_recommend(
    tvars: list[TVarSpec],
    agent_type: str,
    source_code: str,
    llm: ConfigGenLLM,
) -> list[TVarRecommendation]:
    """Ask LLM for additional TVAR recommendations."""
    import json

    tvar_summary = ", ".join(t.name for t in tvars)
    prompt = (
        "You are configuring an optimization search space for an LLM agent.\n\n"
        f"Agent type: {agent_type}\n"
        f"Current tunable variables: [{tvar_summary}]\n\n"
        "Source code:\n"
        f"```python\n{source_code[:2000]}\n```\n\n"
        "Suggest 0-3 additional parameters that could be tuned to improve this "
        "agent's performance. Consider: prompting strategies, retrieval techniques, "
        "model selection, and domain-specific parameters.\n\n"
        "Reply with ONLY a JSON array:\n"
        '[{"name": "...", "range_type": "Range"|"IntRange"|"Choices", '
        '"kwargs": {...}, "category": "...", "reasoning": "...", "impact": "high"|"medium"|"low"}]\n'
        "Return an empty array [] if no recommendations.\n"
    )

    try:
        response = llm.complete(prompt, max_tokens=512)
    except BudgetExhausted:
        return []

    text = extract_json_array_text(response)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [r for item in data if (r := _parse_recommendation_item(item)) is not None]
