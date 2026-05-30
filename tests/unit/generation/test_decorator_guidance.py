"""The @optimize(prompt_rewrite=.../grow_dataset=...) decorator sugar.

Asserts the options are stored on the decorated OptimizedFunction and that
optimize_with_guidance falls back to them when not passed at the call site.
"""

from __future__ import annotations

import json

from traigent.api.decorators import optimize
from traigent.api.parameter_ranges import Choices
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

    def get_guidance_plan(self, request: GuidancePlanRequest) -> GuidancePlan:
        return self._plan


class _Result:
    def __init__(self, score: float) -> None:
        self.best_score = score


def _decorate():
    @optimize(
        eval_dataset=Dataset(
            examples=[EvaluationExample(input_data={"q": "x"}, expected_output="y")]
        ),
        objectives=["accuracy"],
        configuration_space={"prompt": Choices(["base prompt"])},
        prompt_rewrite={"rounds": 1, "candidates_per_round": 2},
    )
    def fn(**kwargs):  # noqa: ANN003, ANN202
        return "ok"

    return fn


def test_decorator_stores_guidance_options() -> None:
    fn = _decorate()
    assert fn.prompt_rewrite_options == {"rounds": 1, "candidates_per_round": 2}
    assert fn.grow_dataset_options is None


def test_optimize_with_guidance_uses_stored_options() -> None:
    fn = _decorate()
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
    )
    seen_spaces: list[list[str]] = []
    scores = iter([0.3, 0.8])

    def fake_optimize_sync(*, configuration_space=None, **kwargs):
        # config_space["prompt"] may be a plain list (decorator-normalized) or a
        # Choices (after merge); accept either.
        vals = configuration_space["prompt"]
        seen_spaces.append(list(getattr(vals, "values", vals)))
        return _Result(next(scores))

    fn.optimize_sync = fake_optimize_sync  # type: ignore[method-assign]

    # No prompt_rewrite passed here -> falls back to the decoration-time options.
    best = fn.optimize_with_guidance(
        _FakeProvider(plan),
        plan_kind="prompt_rewrite",
        rewrite_llm=lambda prompt: json.dumps(["sharper A", "sharper B"]),
        prompt_param="prompt",
    )
    assert best.best_score == 0.8
    assert "sharper A" in seen_spaces[-1]
