"""Subsystem 4: Generate safety constraints from agent classification.

Classifies the agent type and selects appropriate safety constraint
presets from the catalog.
"""

from __future__ import annotations

from traigent.config_generator.agent_classifier import (
    ClassificationResult,
    classify_agent,
)
from traigent.config_generator.llm_backend import ConfigGenLLM
from traigent.config_generator.presets.agent_type_catalog import get_safety_presets
from traigent.config_generator.types import SafetySpec


def generate_safety_constraints(
    source_code: str,
    *,
    llm: ConfigGenLLM | None = None,
    classification: ClassificationResult | None = None,
) -> tuple[list[SafetySpec], ClassificationResult]:
    """Generate safety constraints for the given source code.

    Parameters
    ----------
    source_code:
        The function source code.
    llm:
        Optional LLM for agent classification.
    classification:
        Pre-computed classification result (avoids redundant calls).

    Returns
    -------
    Tuple of (safety constraints, classification result).
    """
    if classification is None:
        classification = classify_agent(source_code, llm=llm)

    presets = get_safety_presets(classification.agent_type)
    return presets, classification
