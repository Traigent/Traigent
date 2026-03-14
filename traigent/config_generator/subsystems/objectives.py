"""Subsystem 2: Generate optimization objectives.

Produces default objectives (accuracy, cost, latency) based on heuristic
analysis of the source code and detected TVARs.  Optional LLM enrichment
adds domain-specific objectives.
"""

from __future__ import annotations

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM
from traigent.config_generator.types import ObjectiveSpec, TVarSpec
from traigent.utils.llm_response_parsing import extract_json_array_text

# Framework import patterns that suggest latency is a relevant objective
_LLM_FRAMEWORK_PATTERNS = frozenset(
    {
        "langchain",
        "openai",
        "anthropic",
        "ChatOpenAI",
        "ChatAnthropic",
        "litellm",
        "cohere",
        "google.generativeai",
        "mistralai",
    }
)


def generate_objectives(
    source_code: str,
    tvars: list[TVarSpec],
    *,
    llm: ConfigGenLLM | None = None,
    classification: ClassificationResult | None = None,
) -> list[ObjectiveSpec]:
    """Generate objectives for the optimization config.

    Parameters
    ----------
    source_code:
        The function source code.
    tvars:
        Already-resolved TVarSpecs (from subsystem 1).
    llm:
        Optional LLM for enrichment.
    classification:
        Pre-computed agent classification.
    """
    objectives = _heuristic_objectives(source_code, tvars, classification)

    if llm is not None:
        enriched = _llm_enrich_objectives(source_code, objectives, llm)
        if enriched:
            objectives = enriched

    return _normalize_weights(objectives)


def _heuristic_objectives(
    source_code: str,
    tvars: list[TVarSpec],
    classification: ClassificationResult | None,
) -> list[ObjectiveSpec]:
    """Generate objectives from heuristic analysis."""
    objectives: list[ObjectiveSpec] = []

    # Always include accuracy
    objectives.append(
        ObjectiveSpec(
            name="accuracy",
            orientation="maximize",
            weight=0.6,
            source="default",
            reasoning="Primary quality objective — always included",
        )
    )

    # If any model-type TVAR exists, cost is relevant
    has_model_tvar = any(
        t.range_type == "Choices"
        and t.name in {"model", "model_name", "model_id", "engine"}
        for t in tvars
    )
    if has_model_tvar or _has_llm_imports(source_code):
        objectives.append(
            ObjectiveSpec(
                name="cost",
                orientation="minimize",
                weight=0.2,
                source="heuristic",
                reasoning="Model selection detected — cost optimization applies",
            )
        )

    # If LLM framework imports detected, latency matters
    if _has_llm_imports(source_code):
        objectives.append(
            ObjectiveSpec(
                name="latency",
                orientation="minimize",
                weight=0.2,
                source="heuristic",
                reasoning="LLM API calls detected — latency optimization applies",
            )
        )

    # Domain-specific objectives based on agent type
    if classification is not None:
        if classification.agent_type == "rag":
            objectives.append(
                ObjectiveSpec(
                    name="faithfulness",
                    orientation="maximize",
                    weight=0.15,
                    source="heuristic",
                    reasoning="RAG agent — faithfulness to source context is important",
                )
            )
        elif classification.agent_type == "classification":
            objectives.append(
                ObjectiveSpec(
                    name="f1",
                    orientation="maximize",
                    weight=0.15,
                    source="heuristic",
                    reasoning="Classification agent — F1 score measures balance of precision/recall",
                )
            )

    return objectives


def _has_llm_imports(source_code: str) -> bool:
    """Check if source code contains LLM framework imports."""
    return any(pattern in source_code for pattern in _LLM_FRAMEWORK_PATTERNS)


def _normalize_weights(objectives: list[ObjectiveSpec]) -> list[ObjectiveSpec]:
    """Normalize objective weights to sum to 1.0."""
    total = sum(o.weight for o in objectives)
    if total <= 0:
        return objectives

    return [
        ObjectiveSpec(
            name=o.name,
            orientation=o.orientation,
            weight=round(o.weight / total, 4),
            source=o.source,
            reasoning=o.reasoning,
        )
        for o in objectives
    ]


def _parse_objective_item(item: dict, current_names: set[str]) -> ObjectiveSpec | None:
    """Parse a single LLM-suggested objective, or *None* if invalid."""
    name = item.get("name", "")
    if not name or name in current_names:
        return None
    orientation = item.get("orientation", "maximize")
    if orientation not in ("maximize", "minimize"):
        return None
    try:
        weight = float(item.get("weight", 0.15))
    except (ValueError, TypeError):
        weight = 0.15
    return ObjectiveSpec(
        name=name,
        orientation=orientation,
        weight=weight,
        source="llm",
        reasoning=item.get("reasoning", "LLM-suggested objective"),
    )


def _llm_enrich_objectives(
    source_code: str,
    current_objectives: list[ObjectiveSpec],
    llm: ConfigGenLLM,
) -> list[ObjectiveSpec] | None:
    """Ask LLM to suggest additional objectives."""
    import json

    current_names = {o.name for o in current_objectives}
    prompt = (
        "You are configuring optimization objectives for an LLM agent function.\n\n"
        f"Current objectives: {sorted(current_names)}\n\n"
        "Source code:\n"
        f"```python\n{source_code[:2000]}\n```\n\n"
        "Suggest 0-2 additional domain-specific objectives that would improve "
        "this agent's optimization. Only suggest objectives that are clearly "
        "relevant based on the code.\n\n"
        "Reply with ONLY a JSON array: "
        '[{"name": "...", "orientation": "maximize"|"minimize", "weight": 0.1-0.3, "reasoning": "..."}]\n'
        "Return an empty array [] if no additional objectives are needed.\n"
    )

    try:
        response = llm.complete(prompt, max_tokens=512)
    except BudgetExhausted:
        return None

    text = extract_json_array_text(response)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    merged = list(current_objectives)
    for item in data:
        obj = _parse_objective_item(item, current_names)
        if obj is not None:
            merged.append(obj)
            current_names.add(obj.name)

    return merged if len(merged) > len(current_objectives) else None
