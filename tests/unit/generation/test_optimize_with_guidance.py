"""Integration glue test for OptimizedFunction.optimize_with_guidance.

Stubs optimize_sync (so the real optimization pipeline doesn't run) and drives
the hook with a fake provider + fake LLM, asserting: the dataset override is set
during rounds and cleared after, prompt rewrite expands the config space, and the
best result is returned.
"""

from __future__ import annotations

import json

import pytest

from traigent.api.parameter_ranges import Choices
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.generation.models import (
    CoarsePriority,
    GuidanceAction,
    GuidancePlan,
    GuidancePlanItem,
    GuidancePlanRequest,
    PlanKind,
)


class _FakeProvider:
    def __init__(self, plan: GuidancePlan) -> None:
        self._plan = plan
        self.requests: list[GuidancePlanRequest] = []

    def get_guidance_plan(self, request: GuidancePlanRequest) -> GuidancePlan:
        self.requests.append(request)
        return self._plan


class _Result:
    def __init__(self, score: float) -> None:
        self.best_score = score


def _make_opt_fn() -> OptimizedFunction:
    def fn(**kwargs):  # noqa: ANN003, ANN202 - trivial stub target
        return "ok"

    return OptimizedFunction(
        func=fn,
        eval_dataset=Dataset(
            examples=[EvaluationExample(input_data={"q": "x"}, expected_output="y")]
        ),
        objectives=["accuracy"],
        configuration_space={"prompt": Choices(["base prompt"])},
    )


def test_optimize_with_guidance_rewrite_expands_space_and_clears_override() -> None:
    opt = _make_opt_fn()
    plan = GuidancePlan(
        plan_id="p1",
        policy_version="gp-2026.05",
        plan_kind=PlanKind.PROMPT_REWRITE,
        items=[
            GuidancePlanItem(
                "prompt", GuidanceAction.REWRITE_PROMPT, CoarsePriority.HIGH
            )
        ],
        plan_token="gp1.x.y",
        expires_at="2026-05-30T00:00:00Z",
        total_generations=5,
    )

    seen_spaces: list[list[str]] = []
    scores = iter([0.3, 0.8])

    def fake_optimize_sync(*, configuration_space=None, **kwargs):
        # the override must be active while a round runs
        assert opt._dataset_override is not None
        seen_spaces.append(list(configuration_space["prompt"].values))
        return _Result(next(scores))

    opt.optimize_sync = fake_optimize_sync  # type: ignore[method-assign]

    best = opt.optimize_with_guidance(
        _FakeProvider(plan),
        plan_kind="prompt_rewrite",
        rewrite_llm=lambda prompt: json.dumps(["sharper prompt A", "sharper prompt B"]),
        prompt_param="prompt",
        prompt_rewrite={"rounds": 1, "candidates_per_round": 2},
    )

    assert best.best_score == 0.8
    # the guided round searched the expanded space
    assert "sharper prompt A" in seen_spaces[-1]
    # override cleared after the loop
    assert opt._dataset_override is None


def test_optimize_with_guidance_requires_explicit_llm() -> None:
    opt = _make_opt_fn()
    plan = GuidancePlan(
        plan_id="p1",
        policy_version="v",
        plan_kind=PlanKind.PROMPT_REWRITE,
        items=[],
        plan_token="t",
        expires_at="2026-05-30T00:00:00Z",
    )
    from traigent.generation import GenerationProviderError

    with pytest.raises(GenerationProviderError):
        opt.optimize_with_guidance(
            _FakeProvider(plan), plan_kind="prompt_rewrite", prompt_param="prompt"
        )
