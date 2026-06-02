"""Subsystem 6: TVAR recommendations.

Suggests additional tunable variables the user may not have considered,
based on agent type and what TVARs are already detected.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.catalog import catalog_entries, entry_to_recommendation
from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM
from traigent.config_generator.presets.range_presets import get_preset_range
from traigent.config_generator.types import TVarRecommendation, TVarSpec
from traigent.utils.llm_response_parsing import extract_json_array_text

if TYPE_CHECKING:
    from traigent.cloud.client import PriorsBundle

_MIN_PRIOR_SUPPORT_N = 30
_MIN_PRIOR_CONFIDENCE = 0.60
_PRIORS_FETCH_TIMEOUT_SECONDS = 2.0
_RECOMMENDATION_PRIORS_CACHE: dict[tuple[str, tuple[str, ...]], PriorsBundle] = {}


def generate_recommendations(
    tvars: list[TVarSpec],
    *,
    llm: ConfigGenLLM | None = None,
    source_code: str = "",
    classification: ClassificationResult | None = None,
    priors_bundle: PriorsBundle | None = None,
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
    priors_bundle:
        Optional learned priors bundle. ``None`` means try the backend when
        configured; an empty bundle keeps catalog output unchanged.
    """
    existing_names = {t.name for t in tvars}
    agent_type = classification.agent_type if classification else "general_llm"

    # Preset recommendations
    recs = _preset_recommendations(agent_type, existing_names)
    recs = _augment_recommendations_with_priors(
        recs,
        (
            priors_bundle
            if priors_bundle is not None
            else _fetch_recommendation_priors(agent_type, [rec.name for rec in recs])
        ),
    )

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


def _augment_recommendations_with_priors(
    recs: list[TVarRecommendation],
    priors_bundle: PriorsBundle,
) -> list[TVarRecommendation]:
    if not recs or not getattr(priors_bundle, "value_priors", ()):
        return recs

    priors_by_name = _gated_value_priors_by_tvar(priors_bundle.value_priors)
    if not priors_by_name:
        return recs

    augmented: list[TVarRecommendation] = []
    for rec in recs:
        hints = [
            hint
            for hint in priors_by_name.get(rec.name, ())
            if _prior_value_allowed(rec, hint[0])
        ]
        if not hints:
            augmented.append(rec)
            continue

        updated_kwargs = _range_kwargs_with_prior_order(rec, hints)
        recommended_values = tuple(value for value, _score in hints)
        augmented.append(
            replace(
                rec,
                range_kwargs=updated_kwargs,
                recommended_values=recommended_values,
            )
        )

    return augmented


def _gated_value_priors_by_tvar(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[tuple[Any, float], ...]]:
    priors_by_name: dict[str, tuple[tuple[Any, float], ...]] = {}
    for row in rows:
        if not _passes_prior_gate(row):
            continue

        tvar_name = str(row.get("tvar_name") or "").strip()
        value_priors = row.get("value_priors")
        if not tvar_name or not isinstance(value_priors, Sequence):
            continue

        gated: list[tuple[Any, float]] = []
        for item in value_priors:
            if not isinstance(item, Mapping) or not _passes_prior_gate(item):
                continue
            if "value" not in item:
                continue
            try:
                score = float(item["score"])
            except (TypeError, ValueError):
                continue
            gated.append((item["value"], score))

        ordered = _dedupe_prior_values(
            sorted(gated, key=lambda item: item[1], reverse=True)
        )
        if ordered:
            priors_by_name[tvar_name] = tuple(ordered)

    return priors_by_name


def _passes_prior_gate(item: Mapping[str, Any]) -> bool:
    try:
        support_n = int(item.get("support_n", 0))
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    return support_n >= _MIN_PRIOR_SUPPORT_N and confidence >= _MIN_PRIOR_CONFIDENCE


def _dedupe_prior_values(
    hints: Sequence[tuple[Any, float]],
) -> tuple[tuple[Any, float], ...]:
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[Any, float]] = []
    for value, score in hints:
        key = (type(value).__name__, repr(value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((value, score))
    return tuple(deduped)


def _range_kwargs_with_prior_order(
    rec: TVarRecommendation,
    hints: Sequence[tuple[Any, float]],
) -> dict[str, Any]:
    if rec.range_type != "Choices":
        return dict(rec.range_kwargs)

    current_values = rec.range_kwargs.get("values")
    if not isinstance(current_values, Sequence) or isinstance(current_values, str):
        return dict(rec.range_kwargs)

    hinted_values = [value for value, _score in hints if value in current_values]
    if not hinted_values:
        return dict(rec.range_kwargs)

    remaining_values = [value for value in current_values if value not in hinted_values]
    return {**rec.range_kwargs, "values": [*hinted_values, *remaining_values]}


def _prior_value_allowed(rec: TVarRecommendation, value: Any) -> bool:
    if rec.range_type == "Choices":
        values = rec.range_kwargs.get("values")
        return (
            isinstance(values, Sequence)
            and not isinstance(values, str)
            and value in values
        )

    if rec.range_type == "IntRange":
        return type(value) is int and _within_numeric_range(rec.range_kwargs, value)

    if rec.range_type in {"Range", "LogRange"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return _within_numeric_range(rec.range_kwargs, float(value))

    return False


def _within_numeric_range(range_kwargs: Mapping[str, Any], value: float | int) -> bool:
    low = range_kwargs.get("low")
    high = range_kwargs.get("high")
    if isinstance(low, (int, float)) and value < low:
        return False
    if isinstance(high, (int, float)) and value > high:
        return False
    return True


def _fetch_recommendation_priors(
    agent_type: str,
    tvar_names: Sequence[str],
) -> PriorsBundle:
    from traigent.cloud.client import PriorsBundle, TraigentCloudClient

    names = tuple(name for name in tvar_names if name)
    if not names or not _should_fetch_recommendation_priors():
        return PriorsBundle.empty()

    cache_key = (agent_type, names)
    cached = _RECOMMENDATION_PRIORS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        return PriorsBundle.empty()

    async def _fetch() -> PriorsBundle:
        client = TraigentCloudClient(timeout=_PRIORS_FETCH_TIMEOUT_SECONDS)
        try:
            return await client.fetch_tvar_priors(
                agent_type=agent_type,
                tvar_names=names,
            )
        finally:
            await client.close()

    try:
        bundle = asyncio.run(_fetch())
    except Exception:
        bundle = PriorsBundle.empty()

    _RECOMMENDATION_PRIORS_CACHE[cache_key] = bundle
    return bundle


def _should_fetch_recommendation_priors() -> bool:
    from traigent.config.backend_config import BackendConfig
    from traigent.utils.env_config import is_backend_offline

    if is_backend_offline() or _edge_analytics_env_enabled():
        return False

    try:
        if not BackendConfig.get_configured_backend_url():
            return False
        return BackendConfig.has_auth_credentials()
    except Exception:
        return False


def _edge_analytics_env_enabled() -> bool:
    values = {
        os.getenv("TRAIGENT_EDGE_ANALYTICS_MODE", "").strip().lower(),
        os.getenv("TRAIGENT_EXECUTION_MODE", "").strip().lower(),
    }
    return bool(values & {"true", "1", "yes", "edge_analytics"})


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
