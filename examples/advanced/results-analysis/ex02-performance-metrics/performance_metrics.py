#!/usr/bin/env python3
"""Results & Analysis - Performance Metrics (FAQ classifier).

Demonstrates multi-objective (accuracy, cost, response_time) and results table.
"""

import json
import os
import sys
from pathlib import Path

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

from traigent.core.objectives import create_default_objectives

try:
    from langchain_openai import ChatOpenAI  # type: ignore

    _USING_FALLBACK_LLM = False
except Exception:
    _USING_FALLBACK_LLM = True

    class _Resp:
        def __init__(self, content: str):
            self.content = content
            self.response_metadata = {"response_time_ms": 0.0}
            self.usage_metadata = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    class ChatOpenAI:  # type: ignore
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages):
            if isinstance(messages, (list, tuple)) and messages:
                last = messages[-1]
                content = getattr(last, "content", str(last))
            else:
                content = str(messages)
            # Naive FAQ matcher by keywords
            lc = content.lower()
            if "reset" in lc:
                ans = "Use the Reset Password link"
            elif "track" in lc:
                ans = "Use the tracking link in the shipping email"
            else:
                ans = "Please see account settings"
            return _Resp(ans)


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


COST_SOURCE_SIMULATED = "simulated_estimate"
SIMULATED_COST_NOTICE = (
    "NOTE: costs below are SIMULATED estimates for the offline stand-in model "
    "(estimated tokens at list price), not provider-reported spend."
)


def _costs_are_simulated() -> bool:
    """True when this run's costs are simulated, not provider-reported spend.

    That is the case for the offline stand-in LLM AND for mock mode
    (``TRAIGENT_MOCK_LLM``, the default of ``run.py``), whether or not
    ``langchain_openai`` imports: mock mode simulates the usage either way.
    """
    from traigent.utils.env_config import is_mock_llm

    return _USING_FALLBACK_LLM or is_mock_llm()


def simulated_cost_provenance() -> dict[str, str]:
    """Marker merged into exported results next to the cost totals.

    Empty for a real run (real client, mock mode off), so nothing changes there.
    """
    if not _costs_are_simulated():
        return {}
    return {"cost_source": COST_SOURCE_SIMULATED}


def print_simulated_cost_notice() -> None:
    """Print one clear line before results when the stand-in LLM was used."""
    if _costs_are_simulated():
        print(SIMULATED_COST_NOTICE)


def _fallback_usage(text: str, prompt: str) -> str | dict:
    """Price the offline stand-in LLM so the `cost` objective has a value.

    The stand-in ``ChatOpenAI`` above makes no provider call, so nothing
    reports usage and honest cost accounting would refuse the run. Report
    token-derived usage at the tuned model's list price. These are SIMULATED
    estimates (not provider-reported spend) and are labelled as such in the
    run output and exported results. With the real
    ``langchain_openai`` client, usage is captured automatically instead.
    """
    if not _USING_FALLBACK_LLM:
        return text
    from traigent.utils.cost_calculator import cost_from_tokens

    model = str((traigent.get_config() or {}).get("model", "gpt-3.5-turbo"))
    input_tokens = max(1, len(prompt) // 4)
    output_tokens = max(1, len(text) // 4)
    input_cost, output_cost = cost_from_tokens(input_tokens, output_tokens, model)
    return traigent.with_usage(
        text,
        total_cost=input_cost + output_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


DATASET_FILE = os.path.join(os.path.dirname(__file__), "faq_pairs.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {
            "input": {"question": "How can I reset my password?"},
            "output": "Use the Reset Password link",
        },
        {
            "input": {"question": "How do I track my order?"},
            "output": "Use the tracking link in the shipping email",
        },
        {
            "input": {"question": "Where can I update my address?"},
            "output": "Update in account settings",
        },
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()


def _output_text(output):
    """Unwrap the usage envelope that ``traigent.with_usage`` adds in a trial."""
    return output.get("text") if isinstance(output, dict) else output


def _contains_accuracy(
    output: str | None, expected: str | None, llm_metrics=None
) -> float:
    output = _output_text(output)
    if not output or not expected:
        return 0.0
    return 1.0 if expected.lower() in output.lower() else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [100, 200],
    },
    eval_dataset=DATASET_FILE,
    objectives=create_default_objectives(
        ["accuracy", "cost", "response_time"],
        orientations={"response_time": "minimize"},
    ),
    metric_functions={"accuracy": _contains_accuracy},
    offline=True,
    max_trials=10,
)
def faq_answer(question: str) -> str | dict:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=150)
    prompt = f"Answer this customer FAQ briefly and precisely: {question}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return _fallback_usage(getattr(resp, "content", str(resp)), prompt)


if __name__ == "__main__":
    try:
        print(faq_answer("How can I reset my password?"))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
