#!/usr/bin/env python3
"""By Goal - Speed & Latency (LLM Calculator)
Target response_time; same calculator, different decorator.
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
from examples.utils.langchain_compat import ChatOpenAI, HumanMessage

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

DATASET = os.path.join(os.path.dirname(__file__), "calculator_eval.jsonl")


def _calc_accuracy(
    output: str | None, expected: dict | None, llm_metrics=None
) -> float:
    if not output or not expected:
        return 0.0
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(data, dict):
        return 0.0

    got_raw = data.get("result")
    want_raw = expected.get("result") if isinstance(expected, dict) else None
    if got_raw is None or want_raw is None:
        return 0.0

    try:
        got = float(got_raw)
        want = float(want_raw)
        return 1.0 if abs(got - want) < 1e-6 else 0.0
    except (TypeError, ValueError):
        return 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.2],
    },
    eval_dataset=DATASET,
    objectives=["response_time"],
    metric_functions={"accuracy": _calc_accuracy},
    execution_mode="edge_analytics",
    max_trials=10,
)
def calculate(expression: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    prompt = f"""
You are a calculator. Compute the exact numeric result of the expression.
Return a JSON object with keys: formula (string), result (number), explanation (string).
Expression: {expression}
Output only JSON.
""".strip()
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(getattr(response, "content", response)).strip()


if __name__ == "__main__":
    import asyncio

    async def _main():
        res = await calculate.optimize(max_trials=10)
        calculate.set_config(res.best_config)
        print("Best config:", res.best_config)
        print("Test:", calculate("12 + 7 * 2"))

    asyncio.run(_main())
