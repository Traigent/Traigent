"""LLM-backed reflection for skill-document edits."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from traigent.generation.llm_provider import RewriteLLM

from .document import SkillDocument
from .edits import EditOp, parse_edit_ops
from .prompts import build_analyst_prompt, build_merge_prompt

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
    ) -> list[EditOp]:
        prompt = build_analyst_prompt(document, rollouts, polarity, max_edits)
        raw = self._llm.complete(prompt)
        ops = parse_edit_ops(raw)
        return [replace(op, source_type=polarity) for op in ops[:max_edits]]

    def merge(
        self,
        failure_edits: list[EditOp],
        success_edits: list[EditOp],
        max_edits: int,
    ) -> list[EditOp]:
        prompt = build_merge_prompt(failure_edits, success_edits, max_edits)
        raw = self._llm.complete(prompt)
        return parse_edit_ops(raw)[:max_edits]


__all__ = ["Reflector"]
