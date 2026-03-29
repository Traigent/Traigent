#!/usr/bin/env python3
"""Execution Modes - Local Mode (Basic).

Adapted from docs: runs fully locally with a small configuration space.
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

    _sdk = os.environ.get("TRAIGENT_SDK_PATH")
    if _sdk:
        sys.path.insert(0, _sdk)
    else:
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


# Dataset path (created below)
DATASET_FILE = os.path.join(os.path.dirname(__file__), "my_dataset.jsonl")


def _ensure_dataset() -> str:
    """Create a tiny local dataset if missing."""
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    samples = [
        {"input": {"prompt": "Say hello to the world"}, "output": "Hello world"},
        {"input": {"prompt": "Provide a short greeting"}, "output": "Hello"},
        {"input": {"prompt": "Respond politely"}, "output": "Hello, how can I help?"},
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return DATASET_FILE


_ensure_dataset()


def _greeting_accuracy(
    output: str | None, expected: str | None, llm_metrics=None
) -> float:
    """Custom accuracy: count as correct if output contains 'hello'."""
    if not output:
        return 0.0
    return 1.0 if "hello" in output.lower() else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
    },
    eval_dataset=DATASET_FILE,
    objectives=["cost", "accuracy"],
    metric_functions={"accuracy": _greeting_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def my_llm_function(prompt: str) -> str:
    """Simple LLM call that runs locally (mock-friendly)."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    try:
        print(my_llm_function("Say hello to the world"))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
