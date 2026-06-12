"""Prompt builders for local skill-document reflection.

All prompt text in this module is original to the Traigent SDK implementation.
The prompts are sent only to the user-supplied RewriteLLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from .document import SkillDocument

JSON_OUTPUT_CONTRACT = """Return ONLY JSON with this shape:
{
  "edits": [
    {
      "op": "append" | "insert_after" | "replace" | "delete",
      "target": "exact document substring, omitted only for append",
      "content": "new text, omitted only for delete",
      "rationale": "brief evidence-backed reason",
      "support_count": 1
    }
  ]
}
Do not include markdown fences or commentary."""


def _format_rollout(rollout: Any) -> list[str]:
    example_id = getattr(rollout, "example_id", None)
    input_data = getattr(rollout, "input_data", None)
    expected = getattr(rollout, "expected", None)
    actual = getattr(rollout, "actual", None)
    metrics = getattr(rollout, "metrics", None)
    success = getattr(rollout, "success", None)
    return [
        f"- example_id: {example_id!r}",
        f"  input: {input_data!r}",
        f"  expected: {expected!r}",
        f"  actual: {actual!r}",
        f"  metrics: {metrics!r}",
        f"  success: {success!r}",
    ]


def build_analyst_prompt(
    document: SkillDocument,
    rollouts: Sequence[Any],
    polarity: Literal["failure", "success"],
    max_edits: int,
    rejected_digest: str | None = None,
    meta_skill: str | None = None,
) -> str:
    """Build the reflection prompt for one rollout minibatch."""

    lines = [
        "You are proposing small edits to a text skill document used by a model.",
        "The goal is to improve future behavior with general procedural guidance, not to memorize individual cases.",
        f"Focus on {polarity} evidence from the rollout records below.",
        f"Propose at most {max_edits} bounded edits.",
        "",
        "Rules:",
        "- Targets for insert_after, replace, and delete must quote exact text already present in the document.",
        "- Do not propose edits touching text inside <!-- PROTECTED --> or <!-- SLOW_UPDATE --> regions.",
        "- Prefer narrow edits that preserve unrelated document behavior.",
        "- Avoid case-specific names, IDs, or answers unless they are already general rules in the document.",
    ]
    _extend_optimizer_context(lines, rejected_digest, meta_skill)
    lines.extend(
        [
            "",
            "Current skill document:",
            document.text,
            "",
            "Rollout evidence:",
        ]
    )
    for rollout in rollouts:
        lines.extend(_format_rollout(rollout))
    lines.extend(["", JSON_OUTPUT_CONTRACT])
    return "\n".join(lines)


def build_merge_prompt(
    failure_edits: Sequence[Any],
    success_edits: Sequence[Any],
    max_edits: int,
    rejected_digest: str | None = None,
    meta_skill: str | None = None,
) -> str:
    """Build the one M1 merge prompt for failure and success analyst edits."""

    lines = [
        "Merge candidate skill-document edits into a short prioritized edit list.",
        "Favor edits supported by failure evidence, but keep success-derived edits when they preserve useful behavior.",
        "Deduplicate equivalent edits and increase support_count when multiple candidates express the same change.",
        f"Return at most {max_edits} edits.",
    ]
    _extend_optimizer_context(lines, rejected_digest, meta_skill)
    lines.extend(
        [
            "",
            "Failure-derived candidates:",
            repr(list(failure_edits)),
            "",
            "Success-derived candidates:",
            repr(list(success_edits)),
            "",
            JSON_OUTPUT_CONTRACT,
        ]
    )
    return "\n".join(lines)


def build_slow_update_prompt(
    prev_doc: SkillDocument,
    cur_doc: SkillDocument,
    categorized: Any,
    prior_guidance: str,
) -> str:
    """Build the prompt for epoch-wise slow-update guidance."""

    lines = [
        "Create fresh slow-update guidance for the executing agent.",
        "Return direct, actionable guidance only; do not restate or duplicate the main skill document body.",
        "Prioritize preventing regressions first, then fixing persistent failures.",
        "The output will replace only the content inside <!-- SLOW_UPDATE --> markers.",
        "",
        "Previous-epoch best document:",
        prev_doc.text,
        "",
        "Current best document:",
        cur_doc.text,
        "",
        "Prior slow-update guidance:",
        prior_guidance or "(none)",
        "",
        "Probe movement by category:",
    ]
    lines.extend(_format_slow_update_categories(categorized))
    lines.extend(
        [
            "",
            "Return ONLY JSON with this shape:",
            '{ "guidance_markdown": "markdown guidance for the executing agent" }',
            "Do not include markdown fences or commentary.",
        ]
    )
    return "\n".join(lines)


def build_meta_skill_prompt(
    accept_history: Sequence[Any],
    reject_history: Sequence[Any],
    prior_meta: str,
) -> str:
    """Build the prompt for optimizer-side meta-skill guidance."""

    lines = [
        "Update optimizer-side meta guidance for FUTURE OPTIMIZER CALLS.",
        "Summarize which edit kinds helped, which hurt, the useful abstraction level, and regression risks.",
        "Address the optimizer that proposes and merges future edits, not the executing agent.",
        "This meta skill is never inserted into the trained document.",
        "",
        "Prior optimizer meta skill:",
        prior_meta or "(none)",
        "",
        "Accepted edit history:",
        repr(list(accept_history)),
        "",
        "Rejected edit history:",
        repr(list(reject_history)),
        "",
        "Return ONLY JSON with this shape:",
        '{ "meta_skill_content": "guidance for future optimizer calls" }',
        "Do not include markdown fences or commentary.",
    ]
    return "\n".join(lines)


def _extend_optimizer_context(
    lines: list[str],
    rejected_digest: str | None,
    meta_skill: str | None,
) -> None:
    if rejected_digest:
        lines.extend(
            [
                "",
                "BEGIN NEGATIVE_FEEDBACK_REJECTED_EDITS",
                "The edits below were already tried and rejected by the selection gate.",
                "Do not repeat these failed edits; focus on unresolved failures and materially different fixes.",
                rejected_digest,
                "END NEGATIVE_FEEDBACK_REJECTED_EDITS",
            ]
        )
    if meta_skill:
        lines.extend(
            [
                "",
                "BEGIN OPTIMIZER_META_SKILL",
                "Use this optimizer-side guidance when choosing the next edit abstraction level and risk controls.",
                meta_skill,
                "END OPTIMIZER_META_SKILL",
            ]
        )


def _format_slow_update_categories(categorized: Any) -> list[str]:
    lines: list[str] = []
    for category in (
        "regressed",
        "persistent_failure",
        "improved",
        "stable_success",
    ):
        cases = (
            list(categorized.get(category, [])) if hasattr(categorized, "get") else []
        )
        lines.append(f"{category}: {len(cases)}")
        for case in cases[:10]:
            previous = getattr(case, "previous", None)
            current = getattr(case, "current", None)
            lines.extend(
                [
                    f"- example_id: {getattr(case, 'example_id', None)!r}",
                    f"  previous_actual: {getattr(previous, 'actual', None)!r}",
                    f"  current_actual: {getattr(current, 'actual', None)!r}",
                    f"  expected: {getattr(current, 'expected', getattr(previous, 'expected', None))!r}",
                    f"  previous_metrics: {getattr(previous, 'metrics', None)!r}",
                    f"  current_metrics: {getattr(current, 'metrics', None)!r}",
                ]
            )
    return lines


__all__ = [
    "JSON_OUTPUT_CONTRACT",
    "build_analyst_prompt",
    "build_merge_prompt",
    "build_meta_skill_prompt",
    "build_slow_update_prompt",
]
