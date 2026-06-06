#!/usr/bin/env python3
"""Composite knobs: adaptive-RAG-style router dispatch, offline.

This example makes no LLM calls and no network calls. It declares a pre-cascade
``router`` pattern, routes deterministic question payloads through stub RAG
stages, and merges the emitted composite telemetry into ordinary metrics.

Run it::

    uv run python examples/advanced/composite-knobs/router_example.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRAIGENT_OFFLINE_MODE", "true")
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

from traigent.knobs.patterns import router
from traigent.knobs.runtime import ResultKind, StageRunner, execute_composite
from traigent.knobs.telemetry import merge_composite_measures

COMPLEXITY_THRESHOLD = "complexity_threshold"

# Adaptive-RAG recipe:
# - Route on an adequacy/inverted complexity signal, not raw "hardness":
#   P(light_rag adequate | query_light_adequacy(question) >= threshold) >= p.
# - Keep ``signal_inputs=("question",)`` freshness-covered for the threshold
#   CVAR so the gate is calibrated against the same input surface it declares.
# - ``retrieval_confidence_min`` is a heavy-arm CVAR owned by ``rag_heavy``; it
#   is not consumed by this outer router unless the heavy arm is modeled as a
#   nested composite.
# - The terminal heavy arm handles abstention/fall-through when the light-arm
#   adequacy signal is below threshold or unavailable.
ADAPTIVE_RAG_GATE = router(
    "adaptive_rag_gate",
    arms=("rag_light", "rag_heavy"),
    signals=("query_light_adequacy",),
    thresholds=(COMPLEXITY_THRESHOLD,),
    signal_inputs=(("question",),),
    tuned_params=(
        ("retrieval_mode", "query_complexity_strategy", "retrieval_k"),
        (
            "retrieval_mode",
            "query_complexity_strategy",
            "retrieval_k",
            "reranker",
            "web_fallback",
            "decompose_query",
        ),
    ),
)


def _rag_stage(label: str) -> StageRunner:
    def run(payload):
        question = str(payload["question"])
        return [f"{label} answer for: {question}"]

    return StageRunner(run=run, key_fn=lambda x: x, samples=1)


def _query_light_adequacy(payload) -> float:
    question = str(payload["question"]).lower()
    tokens = question.split()
    hard_terms = {"compare", "derive", "audit", "multi-hop", "counterfactual"}
    penalty = 0.2 * sum(1 for term in hard_terms if term in question)
    length_penalty = max(0.0, (len(tokens) - 8) * 0.06)
    return max(0.0, min(1.0, 0.95 - penalty - length_penalty))


def main() -> None:
    stages = {
        "rag_light": _rag_stage("light RAG"),
        "rag_heavy": _rag_stage("heavy RAG"),
    }
    questions = (
        "What is the refund policy?",
        "Compare the audit findings and derive a multi-hop remediation plan.",
    )

    print("router example (offline, deterministic)\n")
    for question in questions:
        run = execute_composite(
            ADAPTIVE_RAG_GATE.structure,
            stages,
            config={"question": question},
            calibrated_values={COMPLEXITY_THRESHOLD: 0.6},
            signals={"query_light_adequacy": _query_light_adequacy},
        )
        metrics: dict[str, float | int] = {
            "accepted": 1 if run.result_kind is ResultKind.OUTPUT else 0
        }
        merge_composite_measures(metrics, run)

        print(f"  question: {question}")
        print(f"    result: {run.result_kind.value}")
        print(f"    output: {run.output}")
        print(f"    raw composite telemetry: {run.measures}")
        print(f"    merged metrics: {metrics}")


if __name__ == "__main__":
    main()
