#!/usr/bin/env python3
"""Results & Analysis - Performance Metrics (FAQ classifier).

Demonstrates multi-objective (accuracy, cost, response_time) and results table.
"""
import json
import os
import sys
from pathlib import Path

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
            # Naive FAQ matcher by keywords
            lc = content.lower()
            if "reset" in lc:
                ans = "Use the Reset Password link"
            elif "track" in lc:
                ans = "Use the tracking link in the shipping email"
            else:
                ans = "Please see account settings"
            return _Resp(ans)


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


DATASET_FILE = os.path.join(os.path.dirname(__file__), "faq_pairs.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {
            "input": {"question": "How can I reset my password?"},
            "output": "Use the Reset Password link",
        },
        {
            "input": {"question": "How do I track my order?"},
            "output": "Use the tracking link in the shipping email",
        },
        {
            "input": {"question": "Where can I update my address?"},
            "output": "Update in account settings",
        },
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()


def _contains_accuracy(
    output: str | None, expected: str | None, llm_metrics=None
) -> float:
    if not output or not expected:
        return 0.0
    return 1.0 if expected.lower() in output.lower() else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [100, 200],
    },
    eval_dataset=DATASET_FILE,
    objectives=["accuracy", "cost", "response_time"],
    metric_functions={"accuracy": _contains_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def faq_answer(question: str) -> str:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0.0, model_kwargs={"max_tokens": 150}
    )
    prompt = f"Answer this customer FAQ briefly and precisely: {question}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    print(faq_answer("How can I reset my password?"))
