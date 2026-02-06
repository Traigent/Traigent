#!/usr/bin/env python3
"""Cookbook Agents - Research Agent (Seamless, minimal)

Answer simple research questions. Exact-match accuracy on short facts.
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

DATASET = os.path.join(os.path.dirname(__file__), "research_eval.jsonl")


@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.3]},
    eval_dataset=DATASET,
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def answer(question: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    prompt = f"Answer concisely with one short phrase: {question}"
    return extract_content(llm.invoke([HumanMessage(content=prompt)])).strip().lower()


if __name__ == "__main__":
    try:
        import asyncio

        async def _main():
            res = await answer.optimize(max_trials=10)
            answer.set_config(res.best_config)
            print("Best config:", res.best_config)
            print("Test:", answer("capital of france"))

        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
