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
        "",
        "Current skill document:",
        document.text,
        "",
        "Rollout evidence:",
    ]
    for rollout in rollouts:
        lines.extend(_format_rollout(rollout))
    lines.extend(["", JSON_OUTPUT_CONTRACT])
    return "\n".join(lines)


def build_merge_prompt(
    failure_edits: Sequence[Any],
    success_edits: Sequence[Any],
    max_edits: int,
) -> str:
    """Build the one M1 merge prompt for failure and success analyst edits."""

    lines = [
        "Merge candidate skill-document edits into a short prioritized edit list.",
        "Favor edits supported by failure evidence, but keep success-derived edits when they preserve useful behavior.",
        "Deduplicate equivalent edits and increase support_count when multiple candidates express the same change.",
        f"Return at most {max_edits} edits.",
        "",
        "Failure-derived candidates:",
        repr(list(failure_edits)),
        "",
        "Success-derived candidates:",
        repr(list(success_edits)),
        "",
        JSON_OUTPUT_CONTRACT,
    ]
    return "\n".join(lines)


__all__ = ["JSON_OUTPUT_CONTRACT", "build_analyst_prompt", "build_merge_prompt"]
