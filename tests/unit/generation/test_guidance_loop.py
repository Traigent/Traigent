"""Tests for the guided-generation loop.

Uses a fake provider + fake LLM + a fake optimize_round to drive the loop with
no backend or real optimizer. Asserts: prompt rewrite expands the config space,
synthesis grows the dataset, best result is tracked across rounds, the loop
stops when generation is empty, the backend plan is REQUIRED (provider errors
propagate), and the content-stays-local outbound canary (the provider only ever
sees a content-free GuidancePlanRequest).
"""

from __future__ import annotations

import json

import pytest

from traigent.api.parameter_ranges import Choices
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.generation import (
    DatasetGrowthOptions,
    ExampleSynthesizer,
    GuidanceLoop,
    PromptRewriteOptions,
    PromptRewriter,
)
from traigent.generation.models import (
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


class _RaisingProvider:
    def get_guidance_plan(self, request: GuidancePlanRequest) -> GuidancePlan:
        raise RuntimeError("backend unavailable")


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, prompt: str) -> str:
        return self.response


class _Result:
    def __init__(self, score: float) -> None:
        self.best_score = score


def _plan(items: list[GuidancePlanItem], kind: PlanKind) -> GuidancePlan:
    return GuidancePlan(
        plan_id="p1",
        policy_version="gp-2026.05",
        plan_kind=kind,
        items=items,
        plan_token="gp1.x.y",
        expires_at="2026-05-30T00:00:00Z",
        total_generations=10,
    )


def test_require_backend_plan_propagates_error() -> None:
    """No offline fallback: a provider failure aborts rather than fabricating."""
    loop = GuidanceLoop(provider=_RaisingProvider(), rewriter=PromptRewriter(_FakeLLM("[]")))
    scores = iter([_Result(0.5)])
    with pytest.raises(RuntimeError, match="backend unavailable"):
        loop.run(
            optimize_round=lambda cs, ds: next(scores),
            config_space={"prompt": Choices(["base"])},
            dataset=Dataset(examples=[]),
            plan_kind=PlanKind.PROMPT_REWRITE,
            prompt_param="prompt",
        )


def test_prompt_rewrite_expands_config_space_and_tracks_best() -> None:
    plan = _plan(
        [GuidancePlanItem("prompt", GuidanceAction.REWRITE_PROMPT, _CP())],
        PlanKind.PROMPT_REWRITE,
    )
    rewriter = PromptRewriter(_FakeLLM(json.dumps(["better A", "better B"])))
    loop = GuidanceLoop(
        provider=_FakeProvider(plan),
        rewriter=rewriter,
        prompt_options=PromptRewriteOptions(rounds=1, candidates_per_round=2),
    )
    cs = {"prompt": Choices(["base"])}
    results = iter([_Result(0.4), _Result(0.9)])
    out = loop.run(
        optimize_round=lambda c, d: next(results),
        config_space=cs,
        dataset=Dataset(examples=[]),
        plan_kind=PlanKind.PROMPT_REWRITE,
        prompt_param="prompt",
        weak_examples=[("q", "a", "wrong")],
    )
    assert set(out.config_space["prompt"].values) == {"base", "better A", "better B"}
    assert out.best_result.best_score == 0.9
    assert len(out.rounds) == 2  # base + 1 guided round


def test_benchmark_guide_grows_dataset() -> None:
    plan = _plan(
        [GuidancePlanItem("ex_seed_0", GuidanceAction.GENERATE_HARDER, _CP())],
        PlanKind.BENCHMARK_GUIDE,
    )
    payload = json.dumps([{"input": {"q": "new"}, "expected_output": "y"}])
    synth = ExampleSynthesizer(_FakeLLM(payload))
    seed = EvaluationExample(input_data={"q": "seed"}, expected_output="s")
    ds = Dataset(examples=[seed])
    loop = GuidanceLoop(
        provider=_FakeProvider(plan),
        synthesizer=synth,
        growth_options=DatasetGrowthOptions(rounds=1, examples_per_round=5),
    )
    results = iter([_Result(0.4), _Result(0.7)])
    out = loop.run(
        optimize_round=lambda c, d: next(results),
        config_space={},
        dataset=ds,
        plan_kind=PlanKind.BENCHMARK_GUIDE,
        seed_resolver=lambda ref: seed if ref == "ex_seed_0" else None,
    )
    assert len(out.dataset.examples) == 2
    assert out.dataset.examples[-1].metadata["synthetic"] is True


def test_loop_stops_when_generation_is_empty() -> None:
    plan = _plan(
        [GuidancePlanItem("prompt", GuidanceAction.REWRITE_PROMPT, _CP())],
        PlanKind.PROMPT_REWRITE,
    )
    # LLM returns only an existing variant -> nothing new -> loop stops after round 1.
    rewriter = PromptRewriter(_FakeLLM(json.dumps(["base"])))
    loop = GuidanceLoop(
        provider=_FakeProvider(plan),
        rewriter=rewriter,
        prompt_options=PromptRewriteOptions(rounds=5, candidates_per_round=2),
    )
    results = iter([_Result(0.4)] + [_Result(0.4)] * 5)
    out = loop.run(
        optimize_round=lambda c, d: next(results),
        config_space={"prompt": Choices(["base"])},
        dataset=Dataset(examples=[]),
        plan_kind=PlanKind.PROMPT_REWRITE,
        prompt_param="prompt",
    )
    assert len(out.rounds) == 1  # only the base round; guided round produced nothing


def test_content_stays_local_outbound_canary() -> None:
    """The provider only ever receives a content-free GuidancePlanRequest."""
    plan = _plan(
        [GuidancePlanItem("ex_seed_0", GuidanceAction.GENERATE_SIMILAR, _CP())],
        PlanKind.BENCHMARK_GUIDE,
    )
    provider = _FakeProvider(plan)
    sentinel = "SENTINEL_PRIVATE_CONTENT_99"
    seed = EvaluationExample(input_data={"q": sentinel}, expected_output=sentinel)
    payload = json.dumps([{"input": {"q": "n"}, "expected_output": "y"}])
    loop = GuidanceLoop(
        provider=provider,
        synthesizer=ExampleSynthesizer(_FakeLLM(payload)),
        growth_options=DatasetGrowthOptions(rounds=1),
    )
    results = iter([_Result(0.4), _Result(0.5)])
    loop.run(
        optimize_round=lambda c, d: next(results),
        config_space={},
        dataset=Dataset(examples=[seed]),
        plan_kind=PlanKind.BENCHMARK_GUIDE,
        seed_resolver=lambda ref: seed,
    )
    # No seed content ever reached the provider request.
    for req in provider.requests:
        assert sentinel not in json.dumps(req.to_dict())


def _CP():
    from traigent.generation.models import CoarsePriority

    return CoarsePriority.HIGH
