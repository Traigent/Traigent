#!/usr/bin/env python3
"""Cookbook Data - Structured Extraction (Seamless, minimal)

Extract company and amount from invoice-like text. Temperature-only.
Custom accuracy metric checks key equality in JSON output.
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

DATASET = os.path.join(os.path.dirname(__file__), "extraction_eval.jsonl")


def _json_accuracy(
    output: str | None, expected: dict | None, llm_metrics=None
) -> float:
    if not output or not expected:
        return 0.0
    try:
        data = json.loads(output)
    except Exception:
        return 0.0
    wanted_company = str(expected.get("company", "")).strip()
    wanted_amount = expected.get("amount")
    got_company = str(data.get("company", "")).strip()
    got_amount = data.get("amount")
    return (
        1.0 if (got_company == wanted_company and got_amount == wanted_amount) else 0.0
    )


@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.3]},
    eval_dataset=DATASET,
    objectives=["accuracy", "cost"],
    metric_functions={"accuracy": _json_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def extract_fields(text: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    prompt = (
        "Extract company and amount as JSON with keys 'company' and 'amount'."
        f"\nText: {text}\nReturn only JSON."
    )
    return extract_content(llm.invoke([HumanMessage(content=prompt)])).strip()


if __name__ == "__main__":
    import asyncio

    async def _main():
        res = await extract_fields.optimize(max_trials=10)
        extract_fields.set_config(res.best_config)
        print("Best config:", res.best_config)
        print(
            "Test:", extract_fields("Invoice: Globex Corp billed $2500 on 2024-06-10.")
        )

    asyncio.run(_main())
