#!/usr/bin/env python3
"""Cookbook NLP - Text Summarization (Quality Metrics)

Adds a simple token-overlap F1 as 'quality'.
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

# --- Setup for running from repo without installation ---
# Add repo root to path so we can import examples.utils and traigent
_module_path = Path(__file__).resolve()
for _depth in range(1, 7):
    try:
        _repo_root = _module_path.parents[_depth]
        if (_repo_root / "traigent").is_dir() and (_repo_root / "examples").is_dir():
            if str(_repo_root) not in sys.path:
                sys.path.insert(0, str(_repo_root))
            break
    except IndexError:
        continue
from examples.utils.langchain_compat import ChatOpenAI, HumanMessage, extract_content

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

DATASET = os.path.join(os.path.dirname(__file__), "summarization_eval.jsonl")


def _summary_f1(output: str | None, expected: str | None, llm_metrics=None) -> float:
    if not output or not expected:
        return 0.0

    def toks(s: str):
        return re.findall(r"[A-Za-z0-9]+", s.lower())

    p, r = Counter(toks(output)), Counter(toks(expected))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    prec = overlap / (sum(p.values()) or 1)
    rec = overlap / (sum(r.values()) or 1)
    return 2 * prec * rec / (prec + rec or 1)


@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.3]},
    eval_dataset=DATASET,
    objectives=["quality", "cost"],
    metric_functions={"quality": _summary_f1},
    execution_mode="edge_analytics",
    max_trials=10,
)
def summarize(document: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    p = f"Summarize in 1-2 sentences:\n\n{document}"
    return extract_content(llm.invoke([HumanMessage(content=p)])).strip()


if __name__ == "__main__":
    import asyncio

    async def _main():
        res = await summarize.optimize(max_trials=10)
        summarize.set_config(res.best_config)
        print("Best config:", res.best_config)
        print("Test:", summarize("TraiGent optimizes LLM apps for quality and cost."))

    asyncio.run(_main())
