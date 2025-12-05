"""Parameter-Based Injection (Custom Parameters) example module.

Define custom parameters and let TraiGent find optimal values.
This gives explicit control over which parameters to optimize.
"""

from __future__ import annotations

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
from examples.utils.langchain_compat import ChatOpenAI

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

# Add optimization with custom parameter names
EVAL_DATASET: str = os.path.join(os.path.dirname(__file__), "chat_interactions.jsonl")


def _summary_f1(output: str | None, expected: str | None, llm_metrics=None) -> float:
    """Token-overlap F1 between output and expected for free-form replies."""
    if not output or not expected:
        return 0.0
    import re
    from collections import Counter

    def tokens(s: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", s.lower())

    p = Counter(tokens(output))
    r = Counter(tokens(expected))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    p_total = sum(p.values()) or 1
    r_total = sum(r.values()) or 1
    precision = overlap / p_total
    recall = overlap / r_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@traigent.optimize(
    configuration_space={
        "llm_creativity": [0.1, 0.3, 0.5, 0.7, 0.9],  # Custom parameter
        "response_length": ["short", "medium", "long"],  # Custom parameter
        "tone": ["formal", "casual", "friendly"],  # Custom parameter
        "model_tier": ["fast", "balanced", "accurate"],  # Custom parameter
    },
    eval_dataset=EVAL_DATASET,
    # Report a summary-friendly score as 'accuracy'
    metric_functions={"accuracy": _summary_f1},
    objectives=["accuracy"],
    execution_mode="edge_analytics",
    max_trials=10,
)
def adaptive_chat_bot(user_message: str) -> str:
    """Adaptive chatbot using custom parameters optimized by TraiGent."""
    # Get optimized parameters from TraiGent
    config = traigent.get_current_config()
    if not isinstance(config, dict):
        config = {}

    # Map custom parameters to actual LLM parameters
    model_map = {
        "fast": "gpt-3.5-turbo",
        "balanced": "gpt-4o-mini",
        "accurate": "gpt-4o",
    }

    max_tokens_map = {"short": 100, "medium": 300, "long": 500}

    # Build system prompt based on tone
    system_prompts = {
        "formal": "You are a professional assistant. Be formal and precise.",
        "casual": "You are a helpful assistant. Keep it casual and conversational.",
        "friendly": "You are a friendly assistant. Be warm and engaging.",
    }

    # Create LLM with optimized parameters
    model = model_map.get(config.get("model_tier", "balanced"), "gpt-3.5-turbo")
    temperature = float(config.get("llm_creativity", 0.5))
    max_tokens = max_tokens_map.get(config.get("response_length", "medium"), 300)

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs={"max_tokens": max_tokens},
    )

    # Get appropriate system prompt
    system_prompt = system_prompts.get(
        config.get("tone", "casual"), system_prompts["casual"]
    )

    # Construct messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Get response
    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


if __name__ == "__main__":
    import asyncio
    import dataclasses
    import json

    result = asyncio.run(adaptive_chat_bot.optimize(max_trials=10))
    print(json.dumps(dataclasses.asdict(result), default=str, indent=2))
