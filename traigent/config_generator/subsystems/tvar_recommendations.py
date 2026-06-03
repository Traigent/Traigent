"""Subsystem 6: TVAR recommendations.

Suggests additional tunable variables the user may not have considered,
based on agent type and what TVARs are already detected.
"""

from __future__ import annotations

from typing import Any

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM
from traigent.config_generator.presets.range_presets import get_preset_range
from traigent.config_generator.types import EvidenceRef, TVarRecommendation, TVarSpec
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

_DEMO_MODEL = "bedrock/us.anthropic.claude-haiku-4-5"
_SINGLE_SLICE_LIMITATIONS = ("single_slice", "not_sota")
_LOW_JOINT_LIMITATIONS = ("low_or_zero_in_isolation", "gains_are_joint")
_OBSERVATIONAL_LIMITATIONS = ("observational_not_causal",)


_RECOMMENDATIONS: dict[str, list[dict[str, Any]]] = {
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
            "name": "retrieval_k",
            "category": "retrieval",
            "reasoning": "Measured HotpotQA demo isolation showed answer EM improved when retrieving more context, but the slice is small.",
            "impact": "medium",
            "evidence_refs": (
                EvidenceRef(
                    scope="isolation",
                    metric="answer_em",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline=1,
                    candidate=5,
                    delta=0.10,
                    limitations=_SINGLE_SLICE_LIMITATIONS,
                ),
            ),
            "apply_guidance": (
                "Manual wiring: after `cfg = traigent.get_config()`, read "
                "cfg['retrieval_k'] and pass it into your retriever's top-k "
                "or limit argument for each query. Keep the existing baseline "
                "value covered and add eval cases across the 1..5 range. "
                "NOTE: apply_config() only inserts the decorator + imports; "
                "it does not change retriever calls."
            ),
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
        {
            "name": "schema_context",
            "category": "structural",
            "reasoning": "Measured BIRD and Spider demo isolation runs showed schema context can materially improve SQL execution accuracy.",
            "impact": "high",
            "evidence_refs": (
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline="none",
                    candidate="linked_top6",
                    delta=0.40,
                    limitations=_SINGLE_SLICE_LIMITATIONS,
                ),
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline="none",
                    candidate="full_ddl_fk",
                    delta=0.30,
                    limitations=_SINGLE_SLICE_LIMITATIONS,
                ),
            ),
            "apply_guidance": (
                "Manual wiring: after `cfg = traigent.get_config()`, branch "
                "on cfg['schema_context']: 'none' sends no schema; "
                "'full_ddl_fk' injects full DDL+FK text; "
                "'linked_top6'/'linked_top10' inject the top-k schema-linked "
                "lines. Add eval coverage for the baseline and each branch. "
                "NOTE: apply_config() only inserts the @traigent.optimize "
                "decorator + imports; it does NOT wire this runtime branch - "
                "you do."
            ),
        },
        {
            "name": "evidence_usage",
            "category": "structural",
            "reasoning": "Measured BIRD demo isolation showed appended evidence hints can improve execution accuracy on a small slice.",
            "impact": "medium",
            "evidence_refs": (
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline="off",
                    candidate="hint_appended",
                    delta=0.10,
                    limitations=_SINGLE_SLICE_LIMITATIONS,
                ),
            ),
            "apply_guidance": (
                "Manual wiring: after `cfg = traigent.get_config()`, branch "
                "on cfg['evidence_usage']: 'off' leaves the prompt unchanged; "
                "'hint_appended' appends retrieved schema or value evidence as "
                "a hint section before generation. Add eval coverage for off "
                "and hint_appended. NOTE: apply_config() only inserts the "
                "decorator + imports; runtime prompt wiring remains your code."
            ),
        },
        {
            "name": "fewshot_selector",
            "category": "prompting",
            "reasoning": "Spider significance data indicates few-shot selection matters, but this signal is observational rather than causal.",
            "impact": "medium",
            "evidence_refs": (
                EvidenceRef(
                    scope="significance",
                    metric="importance",
                    n=200,
                    model=_DEMO_MODEL,
                    baseline="observational",
                    candidate="fewshot_selector",
                    delta=None,
                    limitations=_OBSERVATIONAL_LIMITATIONS,
                ),
            ),
            "apply_guidance": (
                "Manual wiring: after `cfg = traigent.get_config()`, branch "
                "on cfg['fewshot_selector']: 'random' samples examples "
                "uniformly, 'masked_question_similarity' selects nearest "
                "masked-question examples, and 'dail_selection' uses the DAIL "
                "selection path. Add eval coverage for each selector because "
                "this evidence is observational, not causal. NOTE: "
                "apply_config() only inserts the decorator + imports; selector "
                "implementation remains your code."
            ),
        },
        {
            "name": "generation_path",
            "category": "prompting",
            "reasoning": "Generation path had low or zero isolated lift in demos, but can contribute when combined with schema, examples, and repair.",
            "impact": "low",
            "evidence_refs": (
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline="direct_dail",
                    candidate="query_plan_cot",
                    delta=0.0,
                    limitations=_LOW_JOINT_LIMITATIONS,
                ),
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline="direct_dail",
                    candidate="query_plan_cot",
                    delta=0.0,
                    limitations=_LOW_JOINT_LIMITATIONS,
                ),
            ),
            "apply_guidance": (
                "Manual wiring: after `cfg = traigent.get_config()`, branch "
                "on cfg['generation_path']: 'direct_dail' uses direct SQL "
                "generation, 'query_plan_cot' prompts for a plan before SQL, "
                "and 'divide_conquer_cot' decomposes the question before "
                "composing SQL. Add eval coverage for each path and interpret "
                "isolated wins cautiously. NOTE: apply_config() only inserts "
                "the decorator + imports; generation control flow remains "
                "your code."
            ),
        },
        {
            "name": "fewshot_k",
            "category": "prompting",
            "reasoning": "Few-shot count had low or zero isolated lift in demos, but can matter jointly with selector and schema context.",
            "impact": "low",
            "evidence_refs": (
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline=0,
                    candidate=3,
                    delta=0.0,
                    limitations=_LOW_JOINT_LIMITATIONS,
                ),
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline=0,
                    candidate=3,
                    delta=0.0,
                    limitations=_LOW_JOINT_LIMITATIONS,
                ),
            ),
            "apply_guidance": (
                "Manual wiring: after `cfg = traigent.get_config()`, read "
                "cfg['fewshot_k'] and use it as the number of few-shot "
                "examples emitted by whichever selector is active; 0 means no "
                "examples. Add eval coverage for 0 and the upper range. NOTE: "
                "apply_config() only inserts the decorator + imports; example "
                "selection remains your code."
            ),
        },
        {
            "name": "candidate_count",
            "category": "generation",
            "reasoning": "Spider significance data indicates candidate count matters, but this signal is observational rather than causal.",
            "impact": "low",
            "evidence_refs": (
                EvidenceRef(
                    scope="significance",
                    metric="importance",
                    n=200,
                    model=_DEMO_MODEL,
                    baseline="observational",
                    candidate="candidate_count",
                    delta=None,
                    limitations=_OBSERVATIONAL_LIMITATIONS,
                ),
            ),
            "apply_guidance": (
                "Manual wiring: after `cfg = traigent.get_config()`, read "
                "cfg['candidate_count'] and generate that many candidate SQL "
                "or program outputs before your selection or execution step; "
                "1 preserves single-sample behavior. Add eval coverage for "
                "1..3. NOTE: apply_config() only inserts the decorator + "
                "imports; multi-candidate generation remains your code."
            ),
        },
        {
            "name": "repair_policy",
            "category": "repair",
            "reasoning": "Repair policy had low or zero isolated lift in demos, but retry behavior can help when combined with candidate generation and schema context.",
            "impact": "low",
            "evidence_refs": (
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline="off",
                    candidate="sqlite_error_or_empty_once",
                    delta=0.0,
                    limitations=_LOW_JOINT_LIMITATIONS,
                ),
                EvidenceRef(
                    scope="isolation",
                    metric="execution_accuracy",
                    n=10,
                    model=_DEMO_MODEL,
                    baseline="off",
                    candidate="sqlite_error_or_empty_once",
                    delta=0.0,
                    limitations=_LOW_JOINT_LIMITATIONS,
                ),
            ),
            "apply_guidance": (
                "Manual wiring: after `cfg = traigent.get_config()`, branch "
                "on cfg['repair_policy']: 'off' returns the first generation, "
                "'sqlite_error_once' retries once when SQLite reports an "
                "error, and 'sqlite_error_or_empty_once' retries once on "
                "error or empty result. Add eval coverage for each policy. "
                "NOTE: apply_config() only inserts the decorator + imports; "
                "retry handling remains your code."
            ),
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
