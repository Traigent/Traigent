#!/usr/bin/env python3
"""Execution Modes - Hybrid Mode with Privacy.

Adapted from docs: shows privacy + hybrid-style params.
Runs locally (mock-friendly) in this extraction.
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


DATASET_FILE = os.path.join(os.path.dirname(__file__), "proprietary_dataset.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {"input": {"query": "Give a helpful answer."}, "output": "Helpful"},
        {"input": {"query": "Provide a concise answer."}, "output": "Concise"},
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()


def _style_accuracy(
    output: str | None, expected: str | None, llm_metrics=None
) -> float:
    """Custom accuracy: correct if expected style keyword appears."""
    if not output or not expected:
        return 0.0
    return 1.0 if expected.lower() in output.lower() else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.2, 0.5, 0.7, 1.0],
        "system_prompt": ["helpful", "concise", "detailed"],
    },
    eval_dataset=DATASET_FILE,
    objectives=["accuracy", "cost"],
    metric_functions={"accuracy": _style_accuracy},
    privacy_enabled=True,
    execution_mode="edge_analytics",
    max_trials=10,
)
def proprietary_assistant(query: str) -> str:
    config = traigent.get_config()

    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.5),
    )

    system_prompts = {
        "helpful": "You are a helpful assistant",
        "concise": "Provide concise, direct answers",
        "detailed": "Provide detailed, comprehensive answers",
    }

    style = system_prompts.get(config.get("system_prompt", "helpful"), "")
    prompt = f"{style}\n\nUser: {query}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    try:
        print(proprietary_assistant("Give a helpful answer."))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
