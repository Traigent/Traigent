"""LLM-backed reflection for skill-document edits."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from traigent.generation.llm_provider import RewriteLLM
from traigent.generation.validators import extract_json_block

from .document import SkillDocument
from .edits import EditOp, parse_edit_ops
from .prompts import (
    build_analyst_prompt,
    build_merge_prompt,
    build_meta_skill_prompt,
    build_slow_update_prompt,
)

if TYPE_CHECKING:
    from .trainer import RolloutRecord


class Reflector:
    """Generate candidate edits through an already-resolved RewriteLLM."""

    def __init__(self, llm: RewriteLLM) -> None:
        self._llm = llm

    def analyze(
        self,
        document: SkillDocument,
        rollouts: list[RolloutRecord],
        polarity: Literal["failure", "success"],
        max_edits: int,
        rejected_digest: str | None = None,
        meta_skill: str | None = None,
    ) -> list[EditOp]:
        prompt = build_analyst_prompt(
            document,
            rollouts,
            polarity,
            max_edits,
            rejected_digest=rejected_digest,
            meta_skill=meta_skill,
        )
        raw = self._llm.complete(prompt)
        ops = parse_edit_ops(raw)
        return [replace(op, source_type=polarity) for op in ops[:max_edits]]

    def merge(
        self,
        failure_edits: list[EditOp],
        success_edits: list[EditOp],
        max_edits: int,
        rejected_digest: str | None = None,
        meta_skill: str | None = None,
    ) -> list[EditOp]:
        prompt = build_merge_prompt(
            failure_edits,
            success_edits,
            max_edits,
            rejected_digest=rejected_digest,
            meta_skill=meta_skill,
        )
        raw = self._llm.complete(prompt)
        return parse_edit_ops(raw)[:max_edits]

    def slow_update(
        self,
        prev_doc: SkillDocument,
        cur_doc: SkillDocument,
        categorized: object,
        prior_guidance: str,
    ) -> str:
        prompt = build_slow_update_prompt(
            prev_doc, cur_doc, categorized, prior_guidance
        )
        raw = self._llm.complete(prompt)
        return _extract_text_field(raw, "guidance_markdown")

    def meta_skill(
        self,
        accept_history: list[dict[str, object]],
        reject_history: list[dict[str, object]],
        prior_meta: str,
    ) -> str:
        prompt = build_meta_skill_prompt(accept_history, reject_history, prior_meta)
        raw = self._llm.complete(prompt)
        return _extract_text_field(raw, "meta_skill_content")


def _extract_text_field(raw: str, field: str) -> str:
    parsed = extract_json_block(raw)
    if not isinstance(parsed, dict):
        return ""
    value = parsed.get(field)
    return value.strip() if isinstance(value, str) else ""


__all__ = ["Reflector"]
