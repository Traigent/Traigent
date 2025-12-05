#!/usr/bin/env python3
"""Cookbook Agents - Customer Support (Advanced)

Adds optional few-shots, guidance style, and output format. Minimal knobs.
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

DATASET = os.path.join(os.path.dirname(__file__), "support_eval.jsonl")


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.1, 0.3],
        "use_examples": [True, False],
        "prompt_style": ["direct", "reasoned"],
        "output_format": ["label_only", "json"],
    },
    eval_dataset=DATASET,
    objectives=["accuracy", "cost", "response_time"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def support_intent(message: str) -> str:
    cfg = traigent.get_current_config()

    guidance = {
        "direct": "Classify customer support intent.",
        "reasoned": "Think briefly about the goal, then classify the intent.",
    }[cfg.get("prompt_style", "direct")]

    fewshots = (
        """
Examples:
"I need to update my password" -> account
"I was billed incorrectly" -> billing
"The app keeps crashing" -> technical
"How do I use feature X?" -> general
"Please cancel my subscription" -> cancellation
"""
        if cfg.get("use_examples", False)
        else ""
    )

    guard = (
        "Return exactly one label: billing, account, technical, general, or cancellation."
        if cfg.get("output_format") == "label_only"
        else 'Return JSON: {"intent": "billing|account|technical|general|cancellation"}.'
    )

    prompt = f"""
{guidance}
{fewshots}
Message: {message}
{guard}
""".strip()

    llm = ChatOpenAI(
        model=cfg.get("model", "gpt-3.5-turbo"), temperature=cfg.get("temperature", 0.0)
    )
    raw = extract_content(llm.invoke([HumanMessage(content=prompt)]))

    if cfg.get("output_format") == "json":
        try:
            data = json.loads(raw)
            label = str(data.get("intent", "")).strip().lower()
        except Exception:
            label = raw.strip().lower()
    else:
        label = raw.strip().lower()

    return label


if __name__ == "__main__":
    import asyncio

    async def _main():
        print("Optimizing support_intent (advanced)…")
        res = await support_intent.optimize(max_trials=10)
        print("Best config:", res.best_config)
        support_intent.set_config(res.best_config)
        print("Test:", support_intent("I want to cancel my plan"))

    asyncio.run(_main())
