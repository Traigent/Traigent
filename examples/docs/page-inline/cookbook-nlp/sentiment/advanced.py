#!/usr/bin/env python3
"""Cookbook NLP - Sentiment Analysis (Advanced)

Extends the basic version with prompt tuning, optional few-shots,
and lightweight context engineering. Still minimal, still SDK-native.
"""
import json
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

DATASET = os.path.join(os.path.dirname(__file__), "sentiment_eval.jsonl")


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.1, 0.3],
        "use_examples": [True, False],
        "use_context": [True, False],
        "prompt_style": ["direct", "chain_of_thought"],
        "output_format": ["label_only", "json"],
    },
    eval_dataset=DATASET,
    objectives=["accuracy", "cost", "response_time"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def sentiment_analysis(text: str) -> str:
    cfg = traigent.get_config()

    guidance = {
        "direct": "Classify sentiment. Be concise.",
        "chain_of_thought": (
            "Think step-by-step about indicative words, then provide the final label."
        ),
    }[cfg.get("prompt_style", "direct")]

    context = (
        "This is a customer product review. Focus on overall polarity."
        if cfg.get("use_context", False)
        else ""
    )

    fewshots = (
        """
Examples:
Input: "Amazing quality and great value!" -> positive
Input: "Terrible experience, I want a refund." -> negative
Input: "It's okay, nothing special." -> neutral
"""
        if cfg.get("use_examples", False)
        else ""
    )

    guard = (
        "Return exactly one label: positive, negative, or neutral."
        if cfg.get("output_format") == "label_only"
        else 'Return JSON: {"label": "positive|negative|neutral"}.'
    )

    prompt = f"""
{guidance}
{context}
{fewshots}
Text: {text}
{guard}
""".strip()

    llm = ChatOpenAI(
        model=cfg.get("model", "gpt-3.5-turbo"), temperature=cfg.get("temperature", 0.0)
    )
    raw = extract_content(llm.invoke([HumanMessage(content=prompt)]))

    if cfg.get("output_format") == "json":
        try:
            data = json.loads(raw)
            label = str(data.get("label", "")).strip().lower()
        except Exception:
            label = raw.strip().lower()
    else:
        label = raw.strip().lower()

    if "pos" in label:
        label = "positive"
    elif "neg" in label:
        label = "negative"
    elif "neu" in label:
        label = "neutral"

    return label


if __name__ == "__main__":
    import asyncio

    async def _main():
        print("Optimizing sentiment_analysis (advanced)…")
        res = await sentiment_analysis.optimize(max_trials=10)
        print("Best config:", res.best_config)
        sentiment_analysis.set_config(res.best_config)
        print('Test ("Terrible support"):', sentiment_analysis("Terrible support"))

    asyncio.run(_main())
