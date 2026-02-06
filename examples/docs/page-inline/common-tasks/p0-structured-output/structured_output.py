#!/usr/bin/env python3
"""P0-1 Structured Output - JSON extraction reliability.

Optimizes basic JSON extraction; accuracy = valid JSON with required keys.
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
            # Produce simple JSON echo for mock
            payload = ""
            if isinstance(messages, (list, tuple)) and messages:
                last = messages[-1]
                payload = getattr(last, "content", str(last))
            out = {"name": "John Doe", "email": "john@example.com", "amount": 42}
            if payload:
                out["input"] = payload
            return _Resp(json.dumps(out))


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


DATASET = os.path.join(os.path.dirname(__file__), "records.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET):
        return DATASET
    rows = [
        {
            "input": {"text": "Name: John Doe, Email: john@example.com, Amount: 42"},
            "output": {"required_keys": ["name", "email", "amount"]},
        },
        {
            "input": {"text": "Jane: jane@site.com paid $15"},
            "output": {"required_keys": ["name", "email", "amount"]},
        },
    ]
    with open(DATASET, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET


_ensure_dataset()


def _json_accuracy(
    output: str | None, expected: dict | None, llm_metrics=None
) -> float:
    if not output or not expected:
        return 0.0
    try:
        data = json.loads(output)
        req = expected.get("required_keys", [])
        ok = all(k in data for k in req)
        return 1.0 if ok else 0.0
    except Exception:
        return 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3],
        "format_hint": ["json", "plain"],
    },
    eval_dataset=DATASET,
    objectives=["accuracy", "cost"],
    metric_functions={"accuracy": _json_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def extract_json(text: str) -> str:
    cfg = traigent.get_config()
    hint = cfg.get("format_hint", "json")
    llm = ChatOpenAI(
        model=cfg.get("model", "gpt-3.5-turbo"), temperature=cfg.get("temperature", 0.0)
    )
    instr = "Return a JSON object with keys: name, email, amount."
    if hint == "json":
        instr = "Respond ONLY with valid minified JSON with keys: name, email, amount."
    prompt = f"{instr}\n\nText: {text}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


if __name__ == "__main__":
    try:
        print(extract_json("Name: John Doe, Email: john@example.com, Amount: 42"))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
