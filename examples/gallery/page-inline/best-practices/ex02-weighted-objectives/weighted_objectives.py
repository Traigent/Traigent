#!/usr/bin/env python3
"""Best Practices - Weighted Objectives (MCQ Selection).

Demonstrates setting objective weights to reflect priorities.
"""

from __future__ import annotations

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

from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema  # noqa: E402

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")


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
            # Heuristic pick letter based on presence of keywords
            lc = content.lower()
            if "tracking" in lc:
                return _Resp("C")
            if "reset" in lc:
                return _Resp("B")
            return _Resp("A")


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


DATASET_FILE = os.path.join(os.path.dirname(__file__), "mcq.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {
            "input": {
                "question": "How to reset password?",
                "choices": [
                    "Email support",
                    "Use Reset Password",
                    "Create new account",
                    "Wait",
                ],
            },
            "output": 1,
        },
        {
            "input": {
                "question": "How to track order?",
                "choices": [
                    "Not available",
                    "Call courier",
                    "Use tracking link",
                    "Wait 14 days",
                ],
            },
            "output": 2,
        },
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()

MCQ_OBJECTIVE_SCHEMA = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.8),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.2),
    ]
)


def _mcq_accuracy(output: int | None, expected: int | None, llm_metrics=None) -> float:
    if output is None or expected is None:
        return 0.0
    return 1.0 if int(output) == int(expected) else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [80, 120],
    },
    eval_dataset=DATASET_FILE,
    objectives=MCQ_OBJECTIVE_SCHEMA,
    metric_functions={"accuracy": _mcq_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def select_answer(question: str, choices: list[str]) -> int:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    labeled = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices))
    prompt = f"Question: {question}\nOptions:\n{labeled}\nRespond with only the letter (A-D)."
    resp = llm.invoke([HumanMessage(content=prompt)])
    text = (getattr(resp, "content", "") or "").strip().upper()
    idx = 0
    if text and text[0] in {"A", "B", "C", "D"}:
        idx = ord(text[0]) - ord("A")
    return idx


if __name__ == "__main__":
    try:
        print(
            select_answer(
                "How to reset password?",
                ["Email support", "Use Reset Password", "Create new account", "Wait"],
            )
        )
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
