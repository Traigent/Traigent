"""The guided-generation loop.

Drives rounds of: optimize -> fetch the opaque backend GuidancePlan (REQUIRED) ->
resolve plan seeds to LOCAL content -> generate with the user's own LLM (rewrite
expands the prompt Choices; synth grows the dataset) -> re-optimize. Tracks the
best result across rounds and stops on rounds / empty generation / budget cap.

Two invariants enforced here:

* **Require the backend plan** — there is no offline fallback. If the provider
  cannot return a plan, the error propagates; the loop never fabricates guidance.
* **Content stays local** — the only thing handed to the provider is a
  ``GuidancePlanRequest`` (plan_kind + budget + scope), which carries no content.
  Seed/prompt/example content is passed only to the user-supplied LLM via the
  rewriter/synthesizer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from traigent.evaluators.base import EvaluationExample

from .example_synth import ExampleSynthesizer
from .models import (
    GuidanceAction,
    GuidancePlan,
    GuidancePlanItem,
    GuidancePlanRequest,
    PlanKind,
)
from .options import DatasetGrowthOptions, PromptRewriteOptions
from .prompt_rewriter import PromptRewriter, WeakExample, merge_prompt_candidates

# Run one optimization round over (config_space, dataset) and return a result.
OptimizeRound = Callable[[dict[str, Any], Any], Any]
# Resolve a plan seed_ref to a locally-held EvaluationExample (or None if unknown).
SeedResolver = Callable[[str], EvaluationExample | None]


@runtime_checkable
class GuidancePlanProvider(Protocol):
    """Source of opaque GuidancePlans. Implemented by the backend client.

    The request carries no content; the returned plan carries selection only.
    """

    def get_guidance_plan(self, request: GuidancePlanRequest) -> GuidancePlan: ...


@dataclass
class GuidanceRoundResult:
    round_index: int
    result: Any
    candidates_added: int = 0
    examples_added: int = 0


@dataclass
class GuidanceLoopResult:
    best_result: Any
    rounds: list[GuidanceRoundResult] = field(default_factory=list)
    config_space: dict[str, Any] = field(default_factory=dict)
    dataset: Any = None


def _default_score(result: Any) -> float | None:
    for attr in ("best_score", "score", "best_value"):
        value = getattr(result, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


class GuidanceLoop:
    """Orchestrates guided rewrite/synthesis rounds around an optimize callable."""

    def __init__(
        self,
        *,
        provider: GuidancePlanProvider,
        rewriter: PromptRewriter | None = None,
        synthesizer: ExampleSynthesizer | None = None,
        prompt_options: PromptRewriteOptions | None = None,
        growth_options: DatasetGrowthOptions | None = None,
        score_of: Callable[[Any], float | None] = _default_score,
    ) -> None:
        self._provider = provider
        self._rewriter = rewriter
        self._synthesizer = synthesizer
        self._prompt_options = prompt_options or PromptRewriteOptions()
        self._growth_options = growth_options or DatasetGrowthOptions()
        self._score_of = score_of

    # --- generation steps (content stays local) -------------------------------

    def _apply_rewrite(
        self,
        plan: GuidancePlan,
        config_space: dict[str, Any],
        prompt_param: str,
        weak_examples: Sequence[WeakExample],
    ) -> int:
        if self._rewriter is None:
            return 0
        existing = config_space.get(prompt_param)
        current = list(getattr(existing, "values", existing or []))
        current = [v for v in current if isinstance(v, str)]
        candidates = self._rewriter.rewrite(current, weak_examples, plan)
        if not candidates:
            return 0
        config_space[prompt_param] = merge_prompt_candidates(
            config_space, prompt_param, candidates
        )
        return len(candidates)

    def _apply_synth(
        self,
        plan: GuidancePlan,
        dataset: Any,
        seed_resolver: SeedResolver,
    ) -> int:
        if self._synthesizer is None:
            return 0
        added = 0
        for action in (
            GuidanceAction.GENERATE_SIMILAR,
            GuidanceAction.GENERATE_HARDER,
            GuidanceAction.DIVERSIFY_AROUND,
        ):
            items: list[GuidancePlanItem] = plan.items_for(action)
            if not items:
                continue
            seeds = [
                s for s in (seed_resolver(it.seed_ref) for it in items) if s is not None
            ]
            if not seeds:
                continue
            new = self._synthesizer.synthesize(
                seeds,
                action,
                seed_ids=[it.seed_ref for it in items],
                existing=list(getattr(dataset, "examples", [])),
            )
            for example in new:
                dataset.add_example(example)
                added += 1
        return added

    # --- the loop -------------------------------------------------------------

    def run(
        self,
        *,
        optimize_round: OptimizeRound,
        config_space: dict[str, Any],
        dataset: Any,
        plan_kind: PlanKind,
        prompt_param: str | None = None,
        seed_resolver: SeedResolver | None = None,
        weak_examples: Sequence[WeakExample] = (),
    ) -> GuidanceLoopResult:
        is_rewrite = plan_kind is PlanKind.PROMPT_REWRITE
        if is_rewrite and not prompt_param:
            raise ValueError("prompt_param is required for prompt_rewrite guided loops")
        if not is_rewrite and seed_resolver is None:
            raise ValueError(
                "seed_resolver is required for benchmark_guide guided loops"
            )

        rounds_n = (
            self._prompt_options.rounds if is_rewrite else self._growth_options.rounds
        )
        max_candidates = self._prompt_options.max_total_candidates
        max_examples = self._growth_options.max_total_examples_added

        base = optimize_round(config_space, dataset)
        best = base
        best_score = self._score_of(base)
        history: list[GuidanceRoundResult] = [GuidanceRoundResult(0, base)]

        total_candidates = 0
        total_examples = 0
        for r in range(1, rounds_n + 1):
            request = GuidancePlanRequest(
                plan_kind=plan_kind,
                max_items=(
                    self._prompt_options.candidates_per_round
                    if is_rewrite
                    else self._growth_options.examples_per_round
                ),
            )
            # REQUIRED: propagate any provider error; never fabricate a plan.
            plan = self._provider.get_guidance_plan(request)

            added_candidates = 0
            added_examples = 0
            if is_rewrite and prompt_param and total_candidates < max_candidates:
                added_candidates = self._apply_rewrite(
                    plan, config_space, prompt_param, weak_examples
                )
                total_candidates += added_candidates
            elif (
                not is_rewrite
                and seed_resolver is not None
                and total_examples < max_examples
            ):
                added_examples = self._apply_synth(plan, dataset, seed_resolver)
                total_examples += added_examples

            if added_candidates == 0 and added_examples == 0:
                break  # nothing new to search; stop rather than spin

            result = optimize_round(config_space, dataset)
            history.append(
                GuidanceRoundResult(r, result, added_candidates, added_examples)
            )
            score = self._score_of(result)
            if best_score is None or (score is not None and score > best_score):
                best, best_score = result, score

        return GuidanceLoopResult(
            best_result=best, rounds=history, config_space=config_space, dataset=dataset
        )


__all__ = [
    "GuidanceLoop",
    "GuidanceLoopResult",
    "GuidanceRoundResult",
    "GuidancePlanProvider",
    "OptimizeRound",
    "SeedResolver",
]
