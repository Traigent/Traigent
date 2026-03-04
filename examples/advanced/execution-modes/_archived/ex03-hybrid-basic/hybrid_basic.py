#!/usr/bin/env python3
"""Execution Modes - Hybrid Mode (Basic).

Adapted from docs: demonstrates hybrid-style configuration.
Runs locally (mock-friendly) for this example extraction.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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

try:  # Fallbacks if langchain packages are unavailable
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover - simple mock for offline envs

    class _Resp:
        def __init__(self, content: str):
            self.content = content
            self.response_metadata = {"response_time_ms": 0.0}
            self.usage_metadata = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    class ChatOpenAI:  # type: ignore
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages):
            if isinstance(messages, (list, tuple)) and messages:
                last = messages[-1]
                content = getattr(last, "content", str(last))
            else:
                content = str(messages)
            return _Resp(f"MOCK_RESPONSE: {content}")


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:  # pragma: no cover

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


DATASET_FILE = os.path.join(os.path.dirname(__file__), "evaluation_set.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {"input": {"text": "Explain hybrid mode briefly."}, "output": "Explanation"},
        {"input": {"text": "What is priority optimization?"}, "output": "Explanation"},
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()


def _explanation_accuracy(
    output: str | None, expected: str | None, llm_metrics=None
) -> float:
    """Custom accuracy: correct if 'explain' keyword appears."""
    if not output:
        return 0.0
    return 1.0 if "explain" in output.lower() else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_tokens": [100, 300, 500],
    },
    eval_dataset=DATASET_FILE,
    objectives=["cost", "accuracy", "latency"],
    metric_functions={"accuracy": _explanation_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def complex_llm_pipeline(input_data: str) -> str:
    """Pipeline that would be optimized with advanced strategy in hybrid mode."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    resp = llm.invoke([HumanMessage(content=input_data)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    try:
        print(complex_llm_pipeline("Explain hybrid mode briefly."))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
