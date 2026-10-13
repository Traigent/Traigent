#!/usr/bin/env python3
"""Results & Analysis - Viewing Results (Sentiment Classification).

Simple classifier optimized with Traigent; demonstrates results table.
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

try:  # Fallbacks if langchain packages are unavailable
    from langchain_openai import ChatOpenAI  # type: ignore

    _USING_FALLBACK_LLM = False
except Exception:  # pragma: no cover - simple mock for offline envs
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
            # Naive mock sentiment: contains 'love' -> positive; 'terrible' -> negative
            sentiment = (
                "positive"
                if "love" in content.lower()
                else ("negative" if "terrible" in content.lower() else "neutral")
            )
            return _Resp(sentiment)


try:
    from langchain.schema import HumanMessage  # type: ignore
except Exception:  # pragma: no cover

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


def _fallback_usage(text: str, prompt: str) -> str | dict:
    """Price the offline stand-in LLM so the `cost` objective is real data.

    The stand-in ``ChatOpenAI`` above makes no provider call, so nothing
    reports usage and honest cost accounting would refuse the run. Report
    token-derived usage at the tuned model's list price. With the real
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


DATASET_FILE = os.path.join(os.path.dirname(__file__), "sentiment.jsonl")


def _ensure_dataset():
    if os.path.exists(DATASET_FILE):
        return DATASET_FILE
    rows = [
        {"input": {"text": "I love this product!"}, "output": "positive"},
        {"input": {"text": "This is terrible."}, "output": "negative"},
        {"input": {"text": "It's okay, I guess."}, "output": "neutral"},
    ]
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return DATASET_FILE


_ensure_dataset()


def _output_text(output):
    """Unwrap the usage envelope that ``traigent.with_usage`` adds in a trial."""
    return output.get("text") if isinstance(output, dict) else output


def _sentiment_accuracy(
    output: str | None, expected: str | None, llm_metrics=None
) -> float:
    output = _output_text(output)
    if not output or not expected:
        return 0.0
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],
        "temperature": [0.0, 0.3, 0.7],
    },
    eval_dataset=DATASET_FILE,
    objectives=["accuracy", "cost"],
    metric_functions={"accuracy": _sentiment_accuracy},
    offline=True,
    max_trials=10,
)
def sentiment_classifier(text: str) -> str | dict:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    prompt = f"Classify sentiment as positive, negative, or neutral: {text}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return _fallback_usage(getattr(resp, "content", str(resp)), prompt)


if __name__ == "__main__":
    try:
        print(sentiment_classifier("I love this!"))
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130) from None
