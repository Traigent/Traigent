#!/usr/bin/env python3
"""Demonstrate RAGAS metrics that require an LLM (faithfulness, answer_relevancy)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable

try:  # pragma: no cover - allow direct execution
    import traigent
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    import traigent

from traigent.metrics import configure_ragas_defaults

BASE = Path(__file__).parent
DATASET = str(BASE / "evaluation_set.jsonl")

os.environ.setdefault("TRAIGENT_FORCE_LOCAL", "true")
os.environ.setdefault("RAGAS_DISABLE_ANALYTICS", "true")

_RAGAS_READY = True
_RAGAS_IMPORT_ERROR: Exception | None = None
try:
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLM
except Exception as exc:  # pragma: no cover - optional dependency
    _RAGAS_READY = False
    _RAGAS_IMPORT_ERROR = exc
    LangchainLLM = ChatOpenAI = None  # type: ignore[assignment]

ragas_llm = None
_METRIC_FUNCTIONS: dict[str, Callable[..., float]] | None = None

if _RAGAS_READY:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run this example.")
    ragas_llm = LangchainLLM(ChatOpenAI(model="gpt-4o-mini", temperature=0.0))
else:

    def _token_overlap_score(text: str, expected: str) -> float:
        text_tokens = {token for token in text.lower().split() if token}
        expected_tokens = {token for token in expected.lower().split() if token}
        if not text_tokens or not expected_tokens:
            return 0.0
        return min(1.0, len(text_tokens & expected_tokens) / len(expected_tokens))

    def _resolve_expected(example, expected) -> str:
        if expected:
            return str(expected)
        if example and getattr(example, "expected_output", None):
            return str(example.expected_output)
        return ""

    def _mock_faithfulness(*, output=None, expected=None, example=None, **_) -> float:
        candidate = str(output or "")
        reference = _resolve_expected(example, expected)
        if not candidate and reference:
            candidate = reference
        return _token_overlap_score(candidate, reference)

    def _mock_answer_relevancy(*, output=None, example=None, **_) -> float:
        question = ""
        if example and getattr(example, "input_data", None):
            question = str(example.input_data.get("question", ""))
        candidate = str(output or "")
        if not candidate and question:
            candidate = question
        return _token_overlap_score(candidate, question or candidate)

    _METRIC_FUNCTIONS = {
        "faithfulness": _mock_faithfulness,
        "answer_relevancy": _mock_answer_relevancy,
    }

_RESPONSES = {
    "Explain retrieval augmented generation.": "RAG combines document retrieval with an LLM generator to answer questions.",
    "What does a faithfulness metric measure?": "Faithfulness checks whether the answer is backed by the retrieved evidence.",
}

_HEDGED_RESPONSES = {
    "Explain retrieval augmented generation.": "I think RAG mixes some retrieval with generation, though the process might vary.",
    "What does a faithfulness metric measure?": "It probably checks whether an answer loosely follows the evidence provided.",
}


@traigent.optimize(
    eval_dataset=DATASET,
    objectives=["faithfulness", "answer_relevancy"],
    configuration_space={
        "evidence_policy": ["cite_context", "omit_context"],
        "answer_style": ["succinct", "speculative"],
    },
    execution_mode="edge_analytics",
    metric_functions=_METRIC_FUNCTIONS,
)
def answer_question(question: str) -> str:
    cfg = traigent.get_trial_config()
    policy = cfg.get("evidence_policy", "cite_context")
    style = cfg.get("answer_style", "succinct")

    if policy == "cite_context":
        base = _RESPONSES.get(question, "I am not sure.")
    else:
        base = _HEDGED_RESPONSES.get(question, "I am not sure.")

    if style == "speculative":
        base = f"{base} I'll need to double-check the sources to be certain."
    else:
        base = base.rstrip(".") + "."

    return base


async def main() -> None:
    if _RAGAS_READY:
        # Provide the LLM grader and confirm default column names are used.
        configure_ragas_defaults(
            llm=ragas_llm,
            column_map={
                "retrieved_contexts": "retrieved_contexts",
                "reference_contexts": "reference_contexts",
            },
        )
    else:
        print("⚠️  ragas extras not installed; using deterministic fallback metrics.")
        if _RAGAS_IMPORT_ERROR:
            print(f"   Reason: {_RAGAS_IMPORT_ERROR}")

    result = await answer_question.optimize(max_trials=6)

    raw = result.to_dataframe()

    print("Optimization summary:")
    print(f"  trials: {len(result.trials)}")
    print(f"  best_config: {result.best_config}")
    print(f"  metrics: {result.metrics}")

    if raw is not None and not raw.empty:
        print("\n=== Raw trial results ===")
        print(raw.to_string(index=False))

    print("Faithfulness-heavy metrics:")
    for name in ("faithfulness", "answer_relevancy"):
        value = result.metrics.get(name)
        if value is not None:
            print(f"  {name}: {value:.3f}")

    best_trial = next(
        (trial for trial in result.trials if trial.config == result.best_config),
        None,
    )
    if (
        best_trial
        and best_trial.metadata
        and best_trial.metadata.get("evaluation_result")
    ):
        evaluation_result = best_trial.metadata["evaluation_result"]
        print("Best trial metrics:")
        for metric, value in sorted(evaluation_result.metrics.items()):
            print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
