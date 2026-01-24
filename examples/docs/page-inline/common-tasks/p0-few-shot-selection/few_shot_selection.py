#!/usr/bin/env python3
"""P0-3 Few-Shot Example Selection (mocked sentiment classification).

Demonstrates tuning shots and selection strategy.
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
                text = getattr(last, "content", str(last))
            else:
                text = str(messages)
            lc = text.lower()
            if "love" in lc:
                return _Resp("positive")
            if "terrible" in lc or "hate" in lc:
                return _Resp("negative")
            return _Resp("neutral")


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


DATASET = os.path.join(os.path.dirname(__file__), "few_shot.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET):
        return DATASET
    rows = [
        {"input": {"text": "I love this!"}, "output": "positive"},
        {"input": {"text": "This is terrible."}, "output": "negative"},
        {"input": {"text": "It's okay."}, "output": "neutral"},
    ]
    with open(DATASET, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET


_ensure_dataset()


def _exact_match(output: str | None, expected: str | None, llm_metrics=None) -> float:
    if not output or not expected:
        return 0.0
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3],
        "shots": [0, 1, 2],
        "selection_strategy": ["random", "semantic"],
    },
    eval_dataset=DATASET,
    objectives=["accuracy", "cost"],
    metric_functions={"accuracy": _exact_match},
    execution_mode="edge_analytics",
    max_trials=10,
)
def classify_sentiment(text: str) -> str:
    cfg = traigent.get_config()
    examples = []
    if cfg.get("shots", 0) >= 1:
        examples.append("Example: 'I love it' -> positive")
    if cfg.get("shots", 0) >= 2:
        examples.append("Example: 'I hate it' -> negative")
    strategy_note = f"Selection: {cfg.get('selection_strategy', 'random')}"
    llm = ChatOpenAI(
        model=cfg.get("model", "gpt-3.5-turbo"), temperature=cfg.get("temperature", 0.0)
    )
    prompt = (
        f"{strategy_note}\n"
        + ("\n".join(examples) + "\n" if examples else "")
        + f"Text: {text}\nLabel one of [positive, negative, neutral]."
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    print(classify_sentiment("I love this!"))
