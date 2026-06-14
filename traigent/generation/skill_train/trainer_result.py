"""Skill training result dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .edits import EditApplyRecord


@dataclass(slots=True)
class SkillTrainResult:
    best_document: str
    best_selection_score: float
    baseline_selection_score: float
    test_score: float | None
    evaluation_basis: Literal["selection_only", "held_out_test"]
    accepted_edits: list[EditApplyRecord]
    all_edit_records: list[EditApplyRecord]
    epoch_summaries: list[dict[str, Any]]
    artifacts_dir: str | None
    summary: dict[str, Any] = field(default_factory=dict)


__all__ = ["SkillTrainResult"]
