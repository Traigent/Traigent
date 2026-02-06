#!/usr/bin/env python3
"""Execution Modes - Cloud Mode (Basic).

Adapted from docs: in OSS extraction we run locally for speed/offline.
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


DATASET_FILE = os.path.join(os.path.dirname(__file__), "dataset.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {
            "input": {"article": "Traigent helps optimize LLM parameters."},
            "output": "Summary",
        },
        {
            "input": {"article": "Mock mode allows running without API keys."},
            "output": "Summary",
        },
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()


def _summary_accuracy(
    output: str | None, expected: str | None, llm_metrics=None
) -> float:
    """Custom accuracy: treat as correct if output looks like a summary.

    Simple heuristic: presence of 'summary' or 'summar' substring.
    """
    if not output:
        return 0.0
    o = output.lower()
    return 1.0 if ("summar" in o) else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "max_tokens": [100, 300, 500],
    },
    eval_dataset=DATASET_FILE,
    objectives=["cost", "accuracy"],
    metric_functions={"accuracy": _summary_accuracy},
    execution_mode="edge_analytics",  # cloud in docs; Edge Analytics here for offline run
    max_trials=10,
)
def text_summarizer(article: str) -> str:
    """Summarize an article in 2-3 sentences."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = f"Summarize this article in 2-3 sentences:\n\n{article}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    try:
        print(text_summarizer("Traigent helps optimize LLM parameters."))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
