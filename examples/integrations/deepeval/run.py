#!/usr/bin/env python3
"""DeepEval Integration — Optimize LLM outputs using DeepEval evaluation metrics.

This example shows how to use DeepEval's research-backed metrics (relevancy,
faithfulness, etc.) as Traigent optimization objectives.  Traigent runs trials
with different configurations and uses DeepEval scores to find the best one.

Requirements:
    pip install 'traigent[deepeval,integrations]'

Quick run (mock mode — no API keys needed):
    TRAIGENT_MOCK_LLM=true python examples/integrations/deepeval/run.py
"""

from __future__ import annotations

import os

import traigent
from traigent.metrics import DeepEvalScorer

# ---------------------------------------------------------------------------
# 1. Create a DeepEvalScorer with the metrics you care about.
#    String shortcuts map to DeepEval metric classes automatically.
# ---------------------------------------------------------------------------

scorer = DeepEvalScorer(
    ["relevancy", "faithfulness"],
    model="gpt-4o-mini",  # judge model (used by DeepEval to score outputs)
    threshold=0.5,
)


# ---------------------------------------------------------------------------
# 2. Decorate your function — Traigent optimizes the config space against
#    the DeepEval metrics returned by the scorer.
# ---------------------------------------------------------------------------


@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": (0.0, 1.0),
    },
    metric_functions=scorer.to_metric_functions(),
    objectives=["relevancy", "faithfulness"],
    eval_dataset=os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "datasets",
        "deepeval",
        "evaluation_set.jsonl",
    ),
    max_trials=4,
)
def answer_question(question: str) -> str:
    """Simple Q&A agent whose parameters are optimized by Traigent."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return llm.invoke(question).content


# ---------------------------------------------------------------------------
# 3. Run the optimization.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = answer_question.run()
    print(f"\nBest config: {result.best_config}")
    print(f"Best scores: {result.best_metrics}")
