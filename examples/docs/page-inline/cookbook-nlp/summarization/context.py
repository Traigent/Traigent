#!/usr/bin/env python3
"""Cookbook NLP - Text Summarization (Context Optimization)

Adds chunk_size to handle longer docs with minimal logic.
"""

import os
import sys
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

DATASET = os.path.join(os.path.dirname(__file__), "summarization_eval.jsonl")


@traigent.optimize(
    configuration_space={
        "temperature": [0.0, 0.3],
        "chunk_size": [120, 240, 400],
    },
    eval_dataset=DATASET,
    objectives=["cost", "response_time"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def summarize(document: str) -> str:
    cfg = traigent.get_config()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=cfg.get("temperature", 0.0))

    def _sum(text: str) -> str:
        p = f"Summarize in 1-2 sentences:\n\n{text}"
        return extract_content(llm.invoke([HumanMessage(content=p)])).strip()

    words = document.split()
    cs = int(cfg.get("chunk_size", 240))
    if len(words) <= cs:
        return _sum(document)

    parts = [" ".join(words[i : i + cs]) for i in range(0, len(words), cs)]
    partials = [_sum(part) for part in parts]
    return _sum(" ".join(partials))


if __name__ == "__main__":
    import asyncio

    async def _main():
        res = await summarize.optimize(max_trials=10)
        summarize.set_config(res.best_config)
        print("Best config:", res.best_config)
        print("Test:", summarize(" ".join(["Traigent"] * 300)))

    asyncio.run(_main())
