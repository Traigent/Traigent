#!/usr/bin/env python3
"""Composite knobs: execute a verifier gate loop with offline stub stages.

This example is deterministic and offline. It makes no LLM calls and talks to no
backend. It demonstrates:

1. declaring the ``verification_gate`` pattern;
2. executing a generate-verify-revise loop with ``execute_composite``;
3. using a verifier score signal to accept a draft; and
4. merging the content-free loop telemetry into ordinary numeric measures.

Run it::

    uv run python examples/advanced/composite-knobs/verification_gate_example.py
"""

from __future__ import annotations

import os

# Offline by default: this example never needs a backend.
os.environ.setdefault("TRAIGENT_OFFLINE", "1")
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

from traigent.knobs.patterns import self_refine, verification_gate
from traigent.knobs.runtime import LoopBodyResult, LoopBodyRunner, execute_composite
from traigent.knobs.telemetry import merge_composite_measures

PASS_THRESHOLD = "verifier_pass_threshold"
VERIFIER_SIGNAL = "verifier_pass_score"


VERIFIER_GATE = verification_gate(
    "qa_verified",
    stage="generate_verify_revise",
    verifier_signal=VERIFIER_SIGNAL,
    verifier_pass_threshold=PASS_THRESHOLD,
    verification_style="verification_style",
    verification_question_count="verification_question_count",
    verifier_model="verifier_model",
    independent_context="independent_context",
    revision_policy="revision_policy",
    max_iters=2,
)


# bounded_refine_loop RECIPE, not a new factory:
# - use self_refine() with a literal max_iters envelope;
# - thread previous_score and improvement_delta through the loop body state;
# - expose only one acceptance-direction signal threshold to the IR.
# Not expressible in v1: a simultaneous raw improvement_min_delta stop or a
# tuned max_repair_rounds structural loop bound.
BOUNDED_REFINE_RECIPE = self_refine(
    name="bounded_refine_loop",
    stage="critique_repair",
    signal="refine_accept_score",
    threshold="acceptance_threshold",
    max_iters=3,
    state_keys=(
        "draft",
        "critique",
        "score",
        "previous_score",
        "improvement_delta",
        "round",
    ),
    signal_inputs=("draft", "score", "previous_score", "improvement_delta"),
    stage_tuned_params=(
        "critic_model",
        "feedback_rubric",
        "repair_prompt",
        "max_repair_rounds",
        "stop_condition",
    ),
)


def _generate_verify_revise(
    scores: list[float],
) -> tuple[LoopBodyRunner, dict[str, float]]:
    """Return a deterministic loop body for generate -> verify -> revise."""
    cursor = {"i": 0}
    last = {"score": 0.0, "contradiction_score": 0.0}

    def run(config, state):
        i = cursor["i"]
        cursor["i"] += 1
        score = scores[min(i, len(scores) - 1)]
        revised = bool(state)
        draft = (
            f"{config['question']} -> concise answer"
            if not revised
            else f"{state['draft']} [revised against reference]"
        )
        contradiction_score = max(0.0, 1.0 - score)
        last["score"] = score
        last["contradiction_score"] = contradiction_score
        return LoopBodyResult(
            output=draft,
            state={
                "draft": draft,
                "verification_questions": 3,
                "verification_answers": "all required checks completed",
                "verifier_pass_score": score,
                "contradiction_score": contradiction_score,
                "revision": "tightened" if revised else "none",
                "independent_context": config["reference"],
            },
        )

    return LoopBodyRunner(run=run), last


def _verifier_pass_score(state: dict[str, object]) -> float:
    return float(state["verifier_pass_score"])


def main() -> None:
    body, last = _generate_verify_revise([0.52, 0.93])
    run = execute_composite(
        VERIFIER_GATE.structure,
        {"generate_verify_revise": body},
        config={
            "question": "What does the verifier gate demonstrate?",
            "reference": "Use only offline stub evidence.",
        },
        calibrated_values={PASS_THRESHOLD: 0.9},
        signals={VERIFIER_SIGNAL: _verifier_pass_score},
    )

    metrics = {
        "verifier_pass_score": last["score"],
        "contradiction_score": last["contradiction_score"],
        "verification_questions_evaluated": 3,
    }
    merge_composite_measures(metrics, run)

    print("Verification gate example (offline, deterministic)\n")
    print(f"  result kind: {run.result_kind.value}")
    print(f"  output: {run.output}")
    print(f"  measures: {metrics}")
    print(f"  recipe pattern: {BOUNDED_REFINE_RECIPE.provenance.pattern}")


if __name__ == "__main__":
    main()
