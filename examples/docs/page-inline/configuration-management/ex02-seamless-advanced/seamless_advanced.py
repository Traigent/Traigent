"""Seamless Injection (Advanced) example module.

Multiple LLM instances; Traigent will optimize parameters for each.
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

EVAL_DATASET: str = os.path.join(
    os.path.dirname(__file__), "content_requirements.jsonl"
)


def _summary_f1(output: str | None, expected: str | None, llm_metrics=None) -> float:
    """Token-overlap F1 for content/summary-style outputs."""
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
        # Separate configs for different LLM roles
        "analyzer_model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "analyzer_temp": [0.0, 0.1, 0.2],
        "generator_model": ["gpt-3.5-turbo", "gpt-4o"],
        "generator_temp": [0.5, 0.7, 0.9],
        "max_context": [2000, 4000, 8000],
    },
    eval_dataset=EVAL_DATASET,
    # Report as 'accuracy' while using a summary-friendly F1 metric
    metric_functions={"accuracy": _summary_f1},
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def intelligent_content_system(topic: str) -> str:
    """Analyze a topic and generate content based on the analysis."""
    # Step 1: Analyze topic requirements
    analyzer = ChatOpenAI(
        model="gpt-3.5-turbo",  # Uses analyzer_model from optimization
        temperature=0.1,  # Uses analyzer_temp from optimization
        max_tokens=500,
    )
    analysis_result = analyzer.invoke(f"Analyze content requirements for: {topic}")
    analysis = getattr(analysis_result, "content", str(analysis_result))

    # Step 2: Generate content based on analysis
    generator = ChatOpenAI(
        model="gpt-4o",  # Uses generator_model from optimization
        temperature=0.7,  # Uses generator_temp from optimization
        max_tokens=2000,  # Uses max_context from optimization
    )
    content_result = generator.invoke(
        f"Based on this analysis: {analysis}\n\nCreate content about: {topic}"
    )
    return getattr(content_result, "content", str(content_result))


if __name__ == "__main__":
    try:
        import asyncio
        import dataclasses
        import json

        result = asyncio.run(intelligent_content_system.optimize(max_trials=10))
        print(json.dumps(dataclasses.asdict(result), default=str, indent=2))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
