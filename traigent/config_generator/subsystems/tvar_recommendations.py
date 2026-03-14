"""Subsystem 6: TVAR recommendations.

Suggests additional tunable variables the user may not have considered,
based on agent type and what TVARs are already detected.
"""

from __future__ import annotations

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM
from traigent.config_generator.presets.range_presets import get_preset_range
from traigent.config_generator.types import TVarRecommendation, TVarSpec


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

_RECOMMENDATIONS: dict[str, list[dict[str, str]]] = {
    "rag": [
        {
            "name": "prompting_strategy",
            "category": "prompting",
            "reasoning": "RAG systems benefit from different prompting strategies (direct, chain_of_thought, react)",
            "impact": "high",
        },
        {
            "name": "retriever_type",
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
            "name": "reranker_model",
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


def _preset_recommendations(
    agent_type: str,
    existing_names: set[str],
) -> list[TVarRecommendation]:
    """Generate recommendations from presets."""
    templates = _RECOMMENDATIONS.get(agent_type, [])
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
            )
        )

    return recs


_VALID_RANGE_TYPES = frozenset({"Range", "IntRange", "LogRange", "Choices"})


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

    text = _extract_json_text(response)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [r for item in data if (r := _parse_recommendation_item(item)) is not None]
