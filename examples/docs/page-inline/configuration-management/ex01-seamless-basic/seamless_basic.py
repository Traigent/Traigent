"""Seamless Injection (Basic) example module.

Contains the example function and a minimal main that runs optimization.
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


def generate_summary(text: str) -> str:
    """Generate a short summary for the given text."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)
    prompt = f"Summarize this text in 2-3 sentences: {text}"
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


# Add optimization with seamless injection

EVAL_DATASET: str = os.path.join(os.path.dirname(__file__), "summaries.jsonl")


def _summary_f1(output: str | None, expected: str | None, llm_metrics=None) -> float:
    """Simple token-overlap F1 between model output and expected summary.

    This avoids strict exact-match accuracy and is more suitable for summaries.
    """
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
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_tokens": [100, 150, 200, 300],
    },
    eval_dataset=EVAL_DATASET,
    # Map 'accuracy' to a summary-friendly F1 metric for reporting
    metric_functions={"accuracy": _summary_f1},
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def optimized_summary(text: str) -> str:
    """Same implementation; Traigent injects optimal parameters at runtime."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=150)
    prompt = f"Summarize this text in 2-3 sentences: {text}"
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


if __name__ == "__main__":
    import asyncio
    import dataclasses
    import json

    result = asyncio.run(optimized_summary.optimize(max_trials=10))
    print(json.dumps(dataclasses.asdict(result), default=str, indent=2))
