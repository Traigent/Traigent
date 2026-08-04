#!/usr/bin/env python3
"""
Guided Generation - privacy-preserving prompt rewrite + benchmark growth.

Traigent can generate *new* tuning material — improved prompt candidates and new
evaluation examples — guided by the backend's proprietary tuning signals, while
the actual generation runs on YOUR OWN LLM so prompt text and example content
never leave your environment. The backend only ever returns an opaque
``GuidancePlan`` (which seeds to act on, an action verb, and a coarse priority);
it never reveals the signals, values, or selection policy behind the plan.

This example is fully offline and deterministic. It uses:
  * a fake "user LLM" (a plain callback) in place of your real model, and
  * a fake GuidancePlanProvider returning a canned opaque plan in place of the
    backend,
so you can see the mechanics end-to-end without any API key or network.

It demonstrates three things:
  1. PromptRewriter  -> new prompt candidates folded into a Choices via merge.
  2. ExampleSynthesizer -> new evaluation examples that grow the dataset.
  3. GuidanceLoop    -> optimize -> fetch opaque plan -> generate locally ->
                        re-optimize, tracking the best result across rounds.

Run with (from the SDK repo root):
    python examples/core/guided-generation/run.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")
os.environ.setdefault("TRAIGENT_MOCK_LLM", "true")

# --- Import Traigent (works from a source checkout or an installed package) ---
try:
    import traigent  # noqa: F401
except ImportError:
    module_path = Path(__file__).resolve()
    for depth in (2, 3, 4):
        try:
            sys.path.append(str(module_path.parents[depth]))
        except IndexError:
            continue
    import traigent  # noqa: F401

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.generation import (  # noqa: F401  (shown in the docstring below)
    BackendGuidanceProvider,
    ExampleSynthesizer,
    GuidanceAction,
    GuidanceLoop,
    GuidancePlan,
    GuidancePlanItem,
    GuidancePlanRequest,
    PlanKind,
    PromptRewriter,
    merge_prompt_candidates,
)
from traigent.generation.models import CoarsePriority


# ---------------------------------------------------------------------------
# Stand-ins for "your own LLM" and "the backend". In production you would pass
# your real LLM (a callable fn(prompt) -> str, or any client exposing
# .complete()) and a BackendGuidanceProvider bound to your session.
# ---------------------------------------------------------------------------
def fake_user_llm(prompt: str) -> str:
    """A deterministic stand-in for the user's own LLM.

    The real rewriter/synthesizer asks the model for a JSON array; here we return
    a fixed one so the demo is reproducible. Content stays entirely local — this
    function is the ONLY place prompt/example text is ever seen.
    """
    if "improved prompt" in prompt.lower() or "prompt variant" in prompt.lower():
        return json.dumps(
            [
                "Decide if the review is positive or negative. Review: {text}",
                "Sentiment (reply only 'positive' or 'negative'): {text}",
            ]
        )
    # example synthesis: return new {input, expected_output} objects
    return json.dumps(
        [
            {"input": {"text": "Absolutely loved every minute of it."}, "expected_output": "positive"},
            {"input": {"text": "A complete waste of my money."}, "expected_output": "negative"},
        ]
    )


class DemoPlanProvider:
    """A fake GuidancePlanProvider. In production: BackendGuidanceProvider.

    The request it receives is content-free; the plan it returns is opaque
    (selection only — no tuning signal).
    """

    def __init__(self, plan_kind: PlanKind, items: list[GuidancePlanItem]) -> None:
        self._plan = GuidancePlan(
            plan_id="demo-plan",
            policy_version="gp-2026.05",
            plan_kind=plan_kind,
            items=items,
            plan_token="opaque-demo-token",
            expires_at="2099-01-01T00:00:00Z",
            total_generations=4,
        )
        self.requests: list[GuidancePlanRequest] = []

    def get_guidance_plan(self, request: GuidancePlanRequest) -> GuidancePlan:
        self.requests.append(request)
        return self._plan


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def demo_prompt_rewriter() -> None:
    section("1. PromptRewriter: new prompt candidates from your own LLM")
    rewriter = PromptRewriter(fake_user_llm)
    current = ["Classify the sentiment of: {text}"]
    weak = [({"text": "meh, it was fine"}, "neutral", "positive")]  # a failing case
    candidates = rewriter.rewrite(current, weak)
    print("Existing prompt variants:", current)
    print("New candidates (generated locally):")
    for c in candidates:
        print(f"  - {c}")

    config_space = {"prompt_template": current}
    expanded = merge_prompt_candidates(config_space, "prompt_template", candidates)
    print("\nmerge_prompt_candidates -> expanded Choices the optimizer searches:")
    for v in expanded.values:
        print(f"  * {v}")
    # Purity: the original config space is untouched.
    assert config_space["prompt_template"] == current


def demo_example_synth() -> None:
    section("2. ExampleSynthesizer: grow the dataset toward the harder frontier")
    synth = ExampleSynthesizer(fake_user_llm)
    seeds = [EvaluationExample(input_data={"text": "great film"}, expected_output="positive")]
    new = synth.synthesize(seeds, GuidanceAction.GENERATE_HARDER, seed_ids=["ex_seed_0"])
    print(f"Generated {len(new)} new examples (tagged synthetic):")
    for ex in new:
        print(f"  + {ex.input_data} -> {ex.expected_output}  metadata={ex.metadata}")


def demo_guidance_loop() -> None:
    section("3. GuidanceLoop: optimize -> opaque plan -> generate locally -> re-optimize")

    # An injected 'optimize round' so the loop runs offline. In production this is
    # OptimizedFunction.optimize_with_guidance(), which drives the real optimizer.
    scores = iter([0.62, 0.81])

    class _Result:
        def __init__(self, score: float) -> None:
            self.best_score = score

    def optimize_round(config_space, dataset):  # noqa: ANN001, ANN202
        return _Result(next(scores))

    provider = DemoPlanProvider(
        PlanKind.PROMPT_REWRITE,
        [GuidancePlanItem("prompt_template", GuidanceAction.REWRITE_PROMPT, CoarsePriority.HIGH)],
    )
    loop = GuidanceLoop(
        provider=provider,
        rewriter=PromptRewriter(fake_user_llm),
        prompt_options=None,
    )
    outcome = loop.run(
        optimize_round=optimize_round,
        config_space={"prompt_template": ["Classify the sentiment of: {text}"]},
        dataset=Dataset(examples=[]),
        plan_kind=PlanKind.PROMPT_REWRITE,
        prompt_param="prompt_template",
        weak_examples=[({"text": "meh"}, "neutral", "positive")],
    )
    print(f"Rounds run: {len(outcome.rounds)} (round 0 = baseline)")
    print(f"Best score across rounds: {outcome.best_result.best_score}")
    print("Final searched prompt set:")
    for v in outcome.config_space["prompt_template"].values:
        print(f"  * {v}")

    # IP boundary: the provider only ever saw a content-free request.
    for req in provider.requests:
        blob = json.dumps(req.to_dict())
        assert "meh" not in blob and "Classify" not in blob
    print("\n[privacy] The provider only received content-free GuidancePlanRequests.")


if __name__ == "__main__":
    print("Traigent Guided Generation - offline demo (no API key / backend needed)")
    demo_prompt_rewriter()
    demo_example_synth()
    demo_guidance_loop()
    print(
        "\nDone. In production, replace fake_user_llm with your own model and "
        "DemoPlanProvider with BackendGuidanceProvider.from_async_post(session_id, post)."
    )
