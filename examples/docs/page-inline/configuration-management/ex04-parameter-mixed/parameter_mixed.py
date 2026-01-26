"""Parameter-Based Injection (Mixed Approach) example module.

Combines custom parameters with standard parameters for flexible optimization.
Mix Traigent-optimized parameters with fixed configurations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Setup for running from repo without installation ---
# Add repo root to path so we can import examples.utils and traigent
_module_path = Path(__file__).resolve()
for _depth in range(1, 7):
    try:
        _repo_root = _module_path.parents[_depth]
        if (_repo_root / "traigent").is_dir() and (_repo_root / "examples").is_dir():
            if str(_repo_root) not in sys.path:
                sys.path.insert(0, str(_repo_root))
            break
    except IndexError:
        continue
from examples.utils.langchain_compat import ChatOpenAI

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


try:
    import traigent
except ImportError:  # pragma: no cover - support IDE execution paths
    import importlib

    module_path = Path(__file__).resolve()
    for depth in (2, 3):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    traigent = importlib.import_module("traigent")

# Add optimization with mixed parameter approach
EVAL_DATASET: str = os.path.join(os.path.dirname(__file__), "content_generation.jsonl")


def _summary_f1(output: str | None, expected: str | None, llm_metrics=None) -> float:
    """Token-overlap F1 for content-style generation tasks."""
    if not output or not expected:
        return 0.0
    import re
    from collections import Counter

    def tokens(s: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", s.lower())

    p = Counter(tokens(output))
    r = Counter(tokens(expected))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    p_total = sum(p.values()) or 1
    r_total = sum(r.values()) or 1
    precision = overlap / p_total
    recall = overlap / r_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@traigent.optimize(
    configuration_space={
        # Standard parameters that Traigent can directly optimize
        "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        # Custom parameters for business logic
        "content_style": ["technical", "creative", "balanced"],
        "audience_level": ["beginner", "intermediate", "expert"],
        "output_format": ["paragraph", "bullet_points", "structured"],
    },
    eval_dataset=EVAL_DATASET,
    metric_functions={"accuracy": _summary_f1},
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def content_generator(topic: str, context: str = "") -> str:
    """Content generator using mixed parameter optimization approach."""
    # Get optimized parameters from Traigent
    config = traigent.get_config()
    if not isinstance(config, dict):
        config = {}

    # Direct LLM parameters (standard)
    temperature = config.get("temperature", 0.5)
    model = config.get("model", "gpt-3.5-turbo")

    # Custom parameters for content generation
    style = config.get("content_style", "balanced")
    audience = config.get("audience_level", "intermediate")
    format_type = config.get("output_format", "paragraph")

    # Build dynamic system prompt based on custom parameters
    style_instructions = {
        "technical": "Use precise technical language and include specific details.",
        "creative": "Be creative and engaging, use metaphors and vivid descriptions.",
        "balanced": "Balance technical accuracy with accessibility.",
    }

    audience_instructions = {
        "beginner": "Explain concepts simply, avoid jargon, use examples.",
        "intermediate": "Assume basic knowledge, provide moderate detail.",
        "expert": "Use technical terminology freely, focus on advanced concepts.",
    }

    format_instructions = {
        "paragraph": "Write in clear paragraphs with good flow.",
        "bullet_points": "Use bullet points for key information.",
        "structured": "Organize with clear sections and subheadings.",
    }

    # Combine instructions into system prompt
    system_prompt = f"""You are a content generation expert.
Style: {style_instructions.get(style, style_instructions['balanced'])}
Audience: {audience_instructions.get(audience, audience_instructions['intermediate'])}
Format: {format_instructions.get(format_type, format_instructions['paragraph'])}

Generate high-quality content based on the given topic and context."""

    # Create LLM with both standard and derived parameters
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=800,  # Fixed parameter not optimized
    )

    # Prepare the user prompt
    user_prompt = f"Topic: {topic}"
    if context:
        user_prompt += f"\nContext: {context}"
    user_prompt += "\n\nGenerate comprehensive content based on the above."

    # Construct messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Get response
    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


if __name__ == "__main__":
    import asyncio
    import dataclasses
    import json

    result = asyncio.run(content_generator.optimize(max_trials=10))
    print(json.dumps(dataclasses.asdict(result), default=str, indent=2))
