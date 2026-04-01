#!/usr/bin/env python3
"""Cookbook NLP - Sentiment Analysis (Seamless, minimal)

Seamless optimization over temperature only. Exact-match accuracy.
"""

import os
import sys
from pathlib import Path

# --- Setup for running from repo without installation ---
# Set TRAIGENT_SDK_PATH to override when running from outside the repo tree.
_sdk_override = os.environ.get("TRAIGENT_SDK_PATH")
if _sdk_override:
    if _sdk_override not in sys.path:
        sys.path.insert(0, _sdk_override)
else:
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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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

DATASET = os.path.join(os.path.dirname(__file__), "sentiment_eval.jsonl")


@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.3]},
    eval_dataset=DATASET,
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def sentiment_analysis(text: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = f"Classify sentiment (positive/negative/neutral): {text}\nReturn one label only."
    return llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()


if __name__ == "__main__":
    try:
        import asyncio

        async def _main():
            res = await sentiment_analysis.optimize(max_trials=10)
            sentiment_analysis.set_config(res.best_config)
            print("Best config:", res.best_config)
            print('Test ("Great value!"):', sentiment_analysis("Great value!"))

        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
