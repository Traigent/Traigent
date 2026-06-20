#!/usr/bin/env python3
"""Composite knobs: a deterministic mixture-of-experts committee example.

This example is offline by default. It makes no model calls and talks to no
backend; the expert and judge stages are deterministic stubs so the output is
reproducible.

Run it::

    uv run python examples/advanced/composite-knobs/moe_example.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRAIGENT_OFFLINE", "1")
os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")

from traigent.knobs.patterns import moe
from traigent.knobs.runtime import StageRunner, execute_composite
from traigent.knobs.telemetry import merge_composite_measures

VOTE_MARGIN_MIN = "patch_moe.vote_margin_min"

VOTE_MOE = moe(
    "patch_moe",
    experts=("fast_patch", "semantic_patch", "test_driven_patch"),
    aggregate="vote",
    accept_threshold=VOTE_MARGIN_MIN,
    expert_tuned_params=(
        ("repo_context_strategy", "edit_granularity"),
        ("repo_context_strategy",),
        ("test_selection_strategy", "patch_review_mode"),
    ),
)

JUDGE_MOE = moe(
    "review_moe",
    experts=("draft_a", "draft_b"),
    aggregate="judge",
    judge_stage="rubric_judge",
    expert_tuned_params=(("summary_style",), ("citation_policy",)),
    judge_tuned_params=("judge_rubric",),
)


def _stage(outputs: list[str]) -> StageRunner:
    return StageRunner(
        run=lambda _item: list(outputs),
        key_fn=lambda x: x,
        samples=len(outputs),
    )


def _judge() -> StageRunner:
    scores = {"brief_patch": 0.4, "tested_patch": 0.9}
    return StageRunner(run=lambda candidate: [scores[str(candidate)]], samples=1)


def main() -> None:
    vote_run = execute_composite(
        VOTE_MOE.structure,
        {
            "fast_patch": _stage(["tested_patch"]),
            "semantic_patch": _stage(["tested_patch"]),
            "test_driven_patch": _stage(["brief_patch"]),
        },
        config={},
        calibrated_values={VOTE_MARGIN_MIN: 0.5},
    )
    vote_metrics = {"accepted": 1.0 if vote_run.output == "tested_patch" else 0.0}
    merge_composite_measures(vote_metrics, vote_run, prefix="vote_moe")

    judge_run = execute_composite(
        JUDGE_MOE.structure,
        {
            "draft_a": _stage(["brief_patch"]),
            "draft_b": _stage(["tested_patch"]),
            "rubric_judge": _judge(),
        },
        config={},
        calibrated_values={},
    )
    judge_metrics = {"accepted": 1.0 if judge_run.output == "tested_patch" else 0.0}
    merge_composite_measures(judge_metrics, judge_run, prefix="judge_moe")

    print("Mixture-of-experts example (offline, deterministic)\n")
    print(f"  vote aggregate output:  {vote_run.output!r}")
    print(f"    metrics: {vote_metrics}")
    print(f"  judge aggregate output: {judge_run.output!r}")
    print(f"    metrics: {judge_metrics}")


if __name__ == "__main__":
    main()
