#!/usr/bin/env python3
"""Few-shot sentiment classification with parameter injection (k, selection, temperature)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

MOCK = str(os.getenv("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes", "y"}
BASE = Path(__file__).parent
if MOCK:
    os.environ["HOME"] = str(BASE)
    results_dir = BASE / ".traigent_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRAIGENT_RESULTS_FOLDER"] = str(results_dir)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

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

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "few-shot-classification"
if MOCK:
    try:
        traigent.initialize(execution_mode="edge_analytics")
    except Exception:
        pass
DATASET = str(DATA_ROOT / "evaluation_set.jsonl")
PROMPT_PATH = BASE / "prompt.txt"
EXAMPLES_PATH = DATA_ROOT / "example_set.jsonl"


def _load_text(p: Path) -> str:
    return p.read_text().strip()


def _load_examples() -> list[dict]:
    out: list[dict] = []
    if not EXAMPLES_PATH.exists():
        return out
    with open(EXAMPLES_PATH) as f:
        for line in f:
            out.append(json.loads(line))
    return out


_PROMPT = _load_text(PROMPT_PATH)
_EXAMPLES = _load_examples()


def _select_examples(k: int, strategy: str) -> list[dict]:
    if k <= 0:
        return []
    if strategy == "diverse":
        # Round-robin by label for diversity
        grouped: dict[str, list[dict]] = {}
        for ex in _EXAMPLES:
            grouped.setdefault(ex["label"], []).append(ex)
        labels = list(grouped.keys())
        sel: list[dict] = []
        i = 0
        while len(sel) < k and any(grouped.values()):
            lab = labels[i % len(labels)]
            if grouped[lab]:
                sel.append(grouped[lab].pop(0))
            i += 1
        return sel[:k]
    # default: top_k (first k)
    return _EXAMPLES[:k]


def _build_prompt(text: str, fewshots: list[dict]) -> str:
    fs = "\n\n".join([f"User: {e['text']}\nAssistant: {e['label']}" for e in fewshots])
    fs_block = (fs + "\n\n") if fs else ""
    return f"{fs_block}User: {text}\n\n{_PROMPT}"


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["accuracy"],
    configuration_space={
        "model": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "temperature": [0.0, 0.3],
        "k": [0, 2, 4],
        "selection_strategy": ["top_k", "diverse"],
    },
    injection_mode="parameter",
    config_param="config",
    execution_mode="edge_analytics",
)
def classify_sentiment(text: str, config: dict | None = None) -> str:
    # Mock mode: simple keyword-based classifier
    if MOCK:
        t = (text or "").lower()
        pos_kw = ["love", "loved", "great", "amazing", "good", "delightful"]
        neg_kw = ["hate", "hated", "terrible", "bad", "worst", "awful"]
        if any(k in t for k in pos_kw):
            return "positive"
        if any(k in t for k in neg_kw):
            return "negative"
        return "neutral"
    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY"
    cfg = config or {}
    k = int(cfg.get("k", 0))
    strategy = str(cfg.get("selection_strategy", "top_k"))
    examples = _select_examples(k, strategy)
    prompt = _build_prompt(text, examples)
    llm = ChatAnthropic(
        model_name=cfg.get("model", "claude-3-5-sonnet-20241022"),
        temperature=float(cfg.get("temperature", 0.0)),
        timeout=None,
        stop=None,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = str(response.content).strip()
    for label in ["positive", "negative", "neutral"]:
        if label in raw.lower():
            return label
    return raw.split()[0][:16]


if __name__ == "__main__":
    print(
        "Ever fought over which k and selection strategy make your few-shot prompt actually work?"
    )

    async def main() -> None:
        trials = 8 if not MOCK else 4
        r = await classify_sentiment.optimize(algorithm="grid", max_trials=trials)
        print({"best_config": r.best_config, "best_score": r.best_score})

    asyncio.run(main())
