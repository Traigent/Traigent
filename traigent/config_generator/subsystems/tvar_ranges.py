"""Subsystem 1: Convert detected tuned variables into resolved TVarSpecs.

Maps each ``TunedVariableCandidate`` to a ``TVarSpec`` by:

1. Checking the canonical name against rich presets
   (``presets/range_presets.py``).
2. Falling back to the ``SuggestedRange`` from the detection module.
3. Applying value-based heuristics as a last resort.
"""

from __future__ import annotations

from typing import Any

from traigent.config_generator.llm_backend import BudgetExhausted, ConfigGenLLM
from traigent.config_generator.presets.range_presets import get_preset_range
from traigent.config_generator.types import TVarSpec
from traigent.tuned_variables.detection_types import (
    DetectionConfidence,
    DetectionResult,
    TunedVariableCandidate,
)


def generate_tvar_specs(
    results: list[DetectionResult],
    *,
    llm: ConfigGenLLM | None = None,
    source_code: str = "",
) -> list[TVarSpec]:
    """Convert detection results to resolved TVarSpecs.

    Parameters
    ----------
    results:
        Detection results from ``TunedVariableDetector``.
    llm:
        Optional LLM backend for enrichment.  When ``None``, only preset
        and heuristic ranges are used.
    source_code:
        Original source code (used for LLM context).
    """
    specs: list[TVarSpec] = []
    seen_names: set[str] = set()

    for result in results:
        for candidate in result.candidates:
            # Skip LOW confidence and duplicates
            if candidate.confidence == DetectionConfidence.LOW:
                continue
            if candidate.name in seen_names:
                continue
            seen_names.add(candidate.name)

            spec = _resolve_candidate(candidate, llm=llm, source_code=source_code)
            if spec is not None:
                specs.append(spec)

    return specs


def _resolve_candidate(
    candidate: TunedVariableCandidate,
    *,
    llm: ConfigGenLLM | None = None,
    source_code: str = "",
) -> TVarSpec | None:
    """Resolve a single candidate to a TVarSpec."""
    # Priority 1: canonical preset
    canonical = candidate.canonical_name or candidate.name
    preset = get_preset_range(canonical)
    if preset is not None:
        return TVarSpec(
            name=candidate.name,
            range_type=preset["range_type"],
            range_kwargs=dict(preset["kwargs"]),
            source="preset",
            confidence=_confidence_score(candidate.confidence),
            reasoning=f"Canonical preset for '{canonical}'",
        )

    # Priority 2: SuggestedRange from detection
    if candidate.suggested_range is not None:
        return TVarSpec(
            name=candidate.name,
            range_type=candidate.suggested_range.range_type,
            range_kwargs=dict(candidate.suggested_range.kwargs),
            source="detection",
            confidence=_confidence_score(candidate.confidence),
            reasoning=candidate.reasoning or "From detection suggested range",
        )

    # Priority 3: value-based heuristic
    heuristic = _heuristic_from_value(candidate.name, candidate.current_value)
    if heuristic is not None:
        return heuristic

    # Priority 4: LLM enrichment (optional)
    if llm is not None and source_code:
        enriched = _llm_resolve(candidate, llm, source_code)
        if enriched is not None:
            return enriched

    return None


def _confidence_score(confidence: DetectionConfidence) -> float:
    """Map DetectionConfidence enum to numeric score."""
    return {
        DetectionConfidence.HIGH: 1.0,
        DetectionConfidence.MEDIUM: 0.7,
        DetectionConfidence.LOW: 0.3,
    }[confidence]


def _heuristic_from_value(name: str, value: Any) -> TVarSpec | None:
    """Infer range from the current value type and magnitude."""
    # Check bool BEFORE int because bool is a subclass of int in Python
    if isinstance(value, bool):
        return TVarSpec(
            name=name,
            range_type="Choices",
            range_kwargs={"values": [True, False]},
            source="heuristic",
            confidence=0.6,
            reasoning="Boolean toggle",
        )

    if isinstance(value, float):
        if abs(value) < 1e-10:
            low, high = 0.0, 1.0
        else:
            low = round(max(0.0, value * 0.5), 4)
            high = round(value * 2.0, 4)
        return TVarSpec(
            name=name,
            range_type="Range",
            range_kwargs={"low": low, "high": high},
            source="heuristic",
            confidence=0.5,
            reasoning=f"Inferred from current value {value!r} (0.5x–2x)",
        )

    if isinstance(value, int):
        low = max(1, value // 2)
        high = max(low + 1, value * 2)
        return TVarSpec(
            name=name,
            range_type="IntRange",
            range_kwargs={"low": low, "high": high},
            source="heuristic",
            confidence=0.5,
            reasoning=f"Inferred from current value {value!r} (//2–*2)",
        )

    if isinstance(value, str):
        return TVarSpec(
            name=name,
            range_type="Choices",
            range_kwargs={"values": [value]},
            source="heuristic",
            confidence=0.4,
            reasoning=f"Single-value choice from current value {value!r}",
        )

    return None


def _llm_resolve(
    candidate: TunedVariableCandidate,
    llm: ConfigGenLLM,
    source_code: str,
) -> TVarSpec | None:
    """Ask LLM to suggest a range for an unresolved candidate."""
    prompt = (
        "You are helping configure an optimization search space for an LLM agent.\n\n"
        f"Variable name: {candidate.name}\n"
        f"Current value: {candidate.current_value!r}\n"
        f"Type: {candidate.candidate_type.value}\n\n"
        "Source code context:\n"
        f"```python\n{source_code[:2000]}\n```\n\n"
        "Suggest a reasonable optimization range for this variable.\n"
        "Reply with ONLY a JSON object: "
        '{"range_type": "Range"|"IntRange"|"Choices", "kwargs": {...}}\n'
        'For Range: {"low": <float>, "high": <float>}\n'
        'For IntRange: {"low": <int>, "high": <int>}\n'
        'For Choices: {"values": [...]}\n'
    )

    try:
        response = llm.complete(prompt, max_tokens=256)
    except BudgetExhausted:
        return None

    return _parse_llm_range_response(candidate.name, response)


def _parse_llm_range_response(name: str, response: str) -> TVarSpec | None:
    """Parse LLM JSON response into a TVarSpec."""
    import json

    # Extract JSON from response (may have markdown fences)
    text = response.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                text = stripped
                break

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    range_type = data.get("range_type")
    kwargs = data.get("kwargs")
    if not range_type or not isinstance(kwargs, dict):
        return None

    if range_type not in ("Range", "IntRange", "LogRange", "Choices"):
        return None

    return TVarSpec(
        name=name,
        range_type=range_type,
        range_kwargs=kwargs,
        source="llm",
        confidence=0.6,
        reasoning="LLM-suggested range",
    )
