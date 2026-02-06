#!/usr/bin/env python3
"""P0-2 Context Engineering & RAG (mocked).

Demonstrates toggling context inclusion and retrieval parameters.
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
                text = getattr(last, "content", str(last))
            else:
                text = str(messages)
            # Simple echo that picks key phrase from context
            lc = text.lower()
            if "reset password" in lc:
                ans = "Use the Reset Password link"
            elif "track order" in lc or "tracking" in lc:
                ans = "Use the tracking link in the shipping email"
            else:
                ans = "Consult account settings"
            return _Resp(ans)


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


DATASET = os.path.join(os.path.dirname(__file__), "qa_context.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET):
        return DATASET
    rows = [
        {
            "input": {
                "context": "Docs: To reset password use the Reset Password link.",
                "question": "How can I reset my password?",
            },
            "output": {"contains": ["Reset Password"]},
        },
        {
            "input": {
                "context": "Docs: Order tracking uses the tracking link from email.",
                "question": "How do I track my order?",
            },
            "output": {"contains": ["tracking link"]},
        },
    ]
    with open(DATASET, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET


_ensure_dataset()


def _contains_accuracy(
    output: str | None, expected: dict | None, llm_metrics=None
) -> float:
    if not output or not expected:
        return 0.0
    for term in expected.get("contains", []):
        if term.lower() in output.lower():
            return 1.0
    return 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3],
        "use_rag": [True, False],
        "top_k": [1, 2],
    },
    eval_dataset=DATASET,
    objectives=["accuracy", "cost"],
    metric_functions={"accuracy": _contains_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def answer_with_context(context: str, question: str) -> str:
    cfg = traigent.get_config()
    llm = ChatOpenAI(
        model=cfg.get("model", "gpt-3.5-turbo"), temperature=cfg.get("temperature", 0.0)
    )
    ctx = context if cfg.get("use_rag", True) else ""
    prompt = f"Context: {ctx}\nQuestion: {question}\nProvide the best concise answer."
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    try:
        print(
            answer_with_context(
                "Docs: Reset via Reset Password link", "How can I reset my password?"
            )
        )
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
