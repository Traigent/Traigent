"""Structured edit parsing and protected application for skill training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from traigent.generation.validators import extract_json_block, looks_like_injection
from traigent.utils.logging import get_logger

from .document import (
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    _find_marker_spans,
    find_protected_regions,
)

logger = get_logger(__name__)

MAX_DOC_CHARS = 16000
EditStatus = Literal[
    "applied",
    "skipped_target_not_found",
    "skipped_protected_region",
    "skipped_invalid",
    "skipped_duplicate",
]

_OPS = {"append", "insert_after", "replace", "delete"}
_SOURCE_TYPES = {"failure", "success", "human"}


@dataclass(frozen=True, slots=True)
class EditOp:
    op: Literal["append", "insert_after", "replace", "delete"]
    target: str | None
    content: str | None
    rationale: str
    support_count: int = 1
    source_type: Literal["failure", "success", "human"] = "failure"

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"Unsupported edit op: {self.op!r}")
        if self.source_type not in _SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type: {self.source_type!r}")
        if self.op != "append" and not self.target:
            raise ValueError(f"target is required for edit op {self.op!r}")
        if self.op != "delete" and self.content is None:
            raise ValueError(f"content is required for edit op {self.op!r}")
        if self.support_count < 1:
            raise ValueError("support_count must be >= 1")

    @property
    def edit_id(self) -> str:
        payload = json.dumps(
            {"op": self.op, "target": self.target, "content": self.content},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class EditApplyRecord:
    edit: EditOp
    epoch: int
    step: int
    status: EditStatus
    selection_score_before: float | None
    selection_score_after: float | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit.edit_id,
            "op": self.edit.op,
            "target": self.edit.target,
            "content": self.edit.content,
            "rationale": self.edit.rationale,
            "support_count": self.edit.support_count,
            "source_type": self.edit.source_type,
            "epoch": self.epoch,
            "step": self.step,
            "status": self.status,
            "selection_score_before": self.selection_score_before,
            "selection_score_after": self.selection_score_after,
            "reason": self.reason,
        }


def _coerce_edit(entry: Any) -> EditOp | None:
    if not isinstance(entry, dict):
        return None

    op = entry.get("op")
    target = entry.get("target")
    content = entry.get("content")
    rationale = entry.get("rationale")
    source_type = entry.get("source_type", "failure")
    support_count = entry.get("support_count", 1)

    if not isinstance(op, str):
        return None
    if target is not None and not isinstance(target, str):
        return None
    if content is not None and not isinstance(content, str):
        return None
    if not isinstance(rationale, str):
        return None
    if not isinstance(source_type, str):
        return None
    if isinstance(support_count, bool):
        return None

    try:
        support = int(support_count)
        return EditOp(
            op=op,  # type: ignore[arg-type]
            target=target,
            content=content,
            rationale=rationale,
            support_count=support,
            source_type=source_type,  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        return None


def parse_edit_ops(raw: str) -> list[EditOp]:
    """Parse LLM JSON into validated edit ops, skipping malformed entries."""

    parsed = extract_json_block(raw)
    if isinstance(parsed, dict):
        entries = parsed.get("edits", [])
    elif isinstance(parsed, list):
        entries = parsed
    else:
        entries = []

    if not isinstance(entries, list):
        logger.debug("Skill edit response had non-list edits payload")
        return []

    ops: list[EditOp] = []
    for entry in entries:
        op = _coerce_edit(entry)
        if op is None:
            logger.debug("Skipping malformed skill edit entry: %r", entry)
            continue
        ops.append(op)
    return ops


def _range_intersects(start: int, end: int, protected: list[tuple[int, int]]) -> bool:
    return any(start < p_end and end > p_start for p_start, p_end in protected)


def _point_inside(point: int, protected: list[tuple[int, int]]) -> bool:
    return any(p_start < point < p_end for p_start, p_end in protected)


def _terminal_slow_update_start(text: str) -> int | None:
    spans = _find_marker_spans(text, SLOW_UPDATE_START, SLOW_UPDATE_END)
    for start, end in spans:
        if end == len(text) or not text[end:].strip():
            return start
    return None


def _invalid_record(edit: EditOp, reason: str) -> EditApplyRecord:
    return EditApplyRecord(
        edit=edit,
        epoch=0,
        step=0,
        status="skipped_invalid",
        selection_score_before=None,
        selection_score_after=None,
        reason=reason,
    )


def apply_edits(text: str, ops: list[EditOp]) -> tuple[str, list[EditApplyRecord]]:
    """Apply structured edits using exact first-occurrence anchors."""

    current = text
    records: list[EditApplyRecord] = []
    seen: set[str] = set()

    for index, edit in enumerate(ops):
        if edit.edit_id in seen:
            records.append(EditApplyRecord(edit, 0, 0, "skipped_duplicate", None, None))
            continue
        seen.add(edit.edit_id)

        if edit.op != "delete" and edit.content is not None:
            if looks_like_injection(edit.content):
                records.append(_invalid_record(edit, "content looks like injection"))
                continue

        protected = find_protected_regions(current)
        status: EditStatus = "applied"
        reason: str | None = None
        candidate = current

        if edit.op == "append":
            insertion = _terminal_slow_update_start(current)
            if insertion is None:
                insertion = len(current)
            if _point_inside(insertion, protected):
                status = "skipped_protected_region"
            else:
                candidate = (
                    current[:insertion] + (edit.content or "") + current[insertion:]
                )

        elif edit.op == "insert_after":
            target = edit.target or ""
            start = current.find(target)
            if start == -1:
                status = "skipped_target_not_found"
            else:
                end = start + len(target)
                if _range_intersects(start, end, protected) or _point_inside(
                    end, protected
                ):
                    status = "skipped_protected_region"
                else:
                    candidate = current[:end] + (edit.content or "") + current[end:]

        elif edit.op == "replace":
            target = edit.target or ""
            start = current.find(target)
            if start == -1:
                status = "skipped_target_not_found"
            else:
                end = start + len(target)
                if _range_intersects(start, end, protected):
                    status = "skipped_protected_region"
                else:
                    candidate = current[:start] + (edit.content or "") + current[end:]

        elif edit.op == "delete":
            target = edit.target or ""
            start = current.find(target)
            if start == -1:
                status = "skipped_target_not_found"
            else:
                end = start + len(target)
                if _range_intersects(start, end, protected):
                    status = "skipped_protected_region"
                else:
                    candidate = current[:start] + current[end:]
        else:
            status = "skipped_invalid"
            reason = "unknown edit op"

        if status == "applied" and len(candidate) > MAX_DOC_CHARS:
            records.append(
                _invalid_record(edit, f"MAX_DOC_CHARS {MAX_DOC_CHARS} exceeded")
            )
            for remaining in ops[index + 1 :]:
                records.append(
                    _invalid_record(
                        remaining,
                        f"stopped after MAX_DOC_CHARS {MAX_DOC_CHARS} would be exceeded",
                    )
                )
            break

        records.append(EditApplyRecord(edit, 0, 0, status, None, None, reason))
        if status == "applied":
            current = candidate

    return current, records


__all__ = [
    "MAX_DOC_CHARS",
    "EditOp",
    "EditApplyRecord",
    "parse_edit_ops",
    "apply_edits",
]
