#!/usr/bin/env python3
"""Cookbook Data - Business Intelligence (Seamless, minimal)

Summarize KPIs into 2-3 insights. Optimize cost/latency.
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

DATASET = os.path.join(os.path.dirname(__file__), "bi_eval.jsonl")


@traigent.optimize(
    configuration_space={"temperature": [0.2, 0.6]},
    eval_dataset=DATASET,
    objectives=["cost", "response_time"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def summarize_kpis(kpis: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
    prompt = f"Given KPI values, write 2-3 concise insights with trends and recommendations. KPIs: {kpis}"
    return extract_content(llm.invoke([HumanMessage(content=prompt)])).strip()


if __name__ == "__main__":
    import asyncio

    async def _main():
        res = await summarize_kpis.optimize(max_trials=10)
        summarize_kpis.set_config(res.best_config)
        print("Best config:", res.best_config)
        print("Test:\n", summarize_kpis("revenue=+12%, churn=3.1%, nps=47"))

    asyncio.run(_main())
