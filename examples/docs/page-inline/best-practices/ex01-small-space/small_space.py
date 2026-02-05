#!/usr/bin/env python3
"""Best Practices - Start with a Small Search Space (Classification).

Demonstrates a compact configuration and mock-friendly evaluation.
"""

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

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:

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
            lc = content.lower()
            if "cat" in lc:
                label = "animal"
            elif "car" in lc:
                label = "vehicle"
            else:
                label = "other"
            return _Resp(label)


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


DATASET_FILE = os.path.join(os.path.dirname(__file__), "classification.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {"input": {"text": "The cat sat on the mat."}, "output": "animal"},
        {"input": {"text": "A red car is parked outside."}, "output": "vehicle"},
        {"input": {"text": "Weather is nice today."}, "output": "other"},
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()


def _exact_match(output: str | None, expected: str | None, llm_metrics=None) -> float:
    if not output or not expected:
        return 0.0
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


@traigent.optimize(
    configuration_space={
        # Small, safe space for quick iteration
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3],
        "max_tokens": [80, 120],
    },
    eval_dataset=DATASET_FILE,
    objectives=["accuracy", "cost"],
    metric_functions={"accuracy": _exact_match},
    execution_mode="edge_analytics",
    max_trials=10,
)
def classify_text(text: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=80)
    prompt = f"Label this text as one of [animal, vehicle, other]: {text}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    print(classify_text("A red car is parked outside."))
