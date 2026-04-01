#!/usr/bin/env python3
"""Cookbook NLP - Sentiment Analysis (Basic)

Simple, seamless optimization with Traigent.
Exact-match accuracy over a small evaluation set; no custom plumbing.
"""

import json
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
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3],
        "output_format": ["label_only", "json"],
    },
    eval_dataset=DATASET,
    objectives=["accuracy", "cost", "response_time"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def sentiment_analysis(text: str) -> str:
    """Classify sentiment as positive, negative, or neutral.

    Returns one of: positive | negative | neutral
    """
    cfg = traigent.get_config()

    instructions = "Classify the sentiment of the following text."
    guard = (
        "Return exactly one label from this set: positive, negative, neutral."
        if cfg.get("output_format") == "label_only"
        else 'Return a JSON object like {"label": "positive"}.'
    )

    prompt = f"""
{instructions}
Text: {text}
{guard}
""".strip()

    llm = ChatOpenAI(
        model=cfg.get("model", "gpt-3.5-turbo"), temperature=cfg.get("temperature", 0.0)
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content

    if cfg.get("output_format") == "json":
        try:
            data = json.loads(raw)
            label = str(data.get("label", "")).strip().lower()
        except Exception:
            label = raw.strip().lower()
    else:
        label = raw.strip().lower()

    # Normalize common variants
    if "pos" in label:
        label = "positive"
    elif "neg" in label:
        label = "negative"
    elif "neu" in label:
        label = "neutral"

    return label


if __name__ == "__main__":
    try:
        import asyncio

        async def _main():
            print("Optimizing sentiment_analysis (mock mode recommended)…")
            res = await sentiment_analysis.optimize(max_trials=10)
            print("Best config:", res.best_config)
            sentiment_analysis.set_config(res.best_config)
            print('Test ("I love this!"):', sentiment_analysis("I love this!"))

        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
