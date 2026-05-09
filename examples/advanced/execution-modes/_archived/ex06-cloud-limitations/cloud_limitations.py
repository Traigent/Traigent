#!/usr/bin/env python3
"""Execution Modes - Archived Cloud Limitations demo.

This archived example is retained for historical context. Remote cloud
execution is not available today, so the runnable function uses Edge Analytics.
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


DATASET_FILE = os.path.join(os.path.dirname(__file__), "text_classification.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {"input": {"text": "I love this!"}, "output": "positive"},
        {"input": {"text": "This is terrible."}, "output": "negative"},
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()


# Self-contained function; run locally because cloud mode is reserved.
@traigent.optimize(
    eval_dataset=DATASET_FILE,
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.2, 0.4],
    },
    execution_mode="edge_analytics",
    max_trials=10,
)
def simple_classifier(text: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    resp = llm.invoke([HumanMessage(content=f"Classify sentiment: {text}")])
    return getattr(resp, "content", str(resp))


# Local-resource examples kept as historical cloud-limitation notes.
# Kept as illustrative (not optimized/run here)
def database_function(query: str) -> str:
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("create table t(x)")
    conn.execute("insert into t values(?)", (query,))
    conn.commit()
    conn.close()
    return "done"


def file_processor(text: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "local_config.json")
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
    except FileNotFoundError:
        pass
    return text


if __name__ == "__main__":
    try:
        print(simple_classifier("I love this!"))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
