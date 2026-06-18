"""Epoch-wise slow-update guidance helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from traigent.evaluators.base import Dataset
from traigent.generation.validators import looks_like_injection

from .document import (
    PROTECTED_END,
    PROTECTED_START,
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    _find_marker_spans,
    find_protected_regions,
)
from .edits import MAX_DOC_CHARS, EditApplyRecord, EditOp

SlowUpdateCategory = Literal[
    "improved", "regressed", "persistent_failure", "stable_success"
]


@dataclass(frozen=True, slots=True)
class SlowUpdateCase:
    example_id: str
    category: SlowUpdateCategory
    previous: Any
    current: Any


def build_slow_update_probe(train: Dataset, probe_size: int, seed: int) -> Dataset:
    """Build the fixed train-probe dataset reused across epochs."""

    count = min(probe_size, len(train.examples))
    rng = random.Random(seed)  # NOSONAR - deterministic statistical sampling, no crypto
    indices = rng.sample(range(len(train.examples)), count)
    return Dataset(
        examples=[train.examples[i] for i in indices],
        name=f"{train.name}__slow_update_probe",
        description=train.description,
        metadata=dict(train.metadata or {}),
    )


def categorize_slow_update_rollouts(
    previous_rollouts: list[Any], current_rollouts: list[Any]
) -> dict[SlowUpdateCategory, list[SlowUpdateCase]]:
    """Categorize per-example movement between previous and current docs."""

    current_by_id = {
        _rollout_id(rollout, index): rollout
        for index, rollout in enumerate(current_rollouts)
    }
    categorized: dict[SlowUpdateCategory, list[SlowUpdateCase]] = {
        "improved": [],
        "regressed": [],
        "persistent_failure": [],
        "stable_success": [],
    }
    for index, previous in enumerate(previous_rollouts):
        example_id = _rollout_id(previous, index)
        current = current_by_id.get(example_id)
        if current is None:
            continue

        previous_failed = _rollout_failed(previous)
        current_failed = _rollout_failed(current)
        if previous_failed and not current_failed:
            category: SlowUpdateCategory = "improved"
        elif not previous_failed and current_failed:
            category = "regressed"
        elif previous_failed and current_failed:
            category = "persistent_failure"
        else:
            category = "stable_success"
        categorized[category].append(
            SlowUpdateCase(example_id, category, previous, current)
        )
    return categorized


def extract_slow_update_content(text: str) -> str:
    """Return current slow-update guidance, or empty string when absent."""

    find_protected_regions(text)
    spans = _find_marker_spans(text, SLOW_UPDATE_START, SLOW_UPDATE_END)
    if not spans:
        return ""
    if len(spans) > 1:
        raise ValueError("Multiple SLOW_UPDATE marker pairs are not supported")
    start, end = spans[0]
    content_start = start + len(SLOW_UPDATE_START)
    content_end = end - len(SLOW_UPDATE_END)
    return text[content_start:content_end].strip()


def apply_slow_update(
    text: str,
    guidance_markdown: str,
    *,
    epoch: int,
    step: int,
) -> tuple[str, EditApplyRecord]:
    """Replace only the SLOW_UPDATE region, appending markers when absent."""

    find_protected_regions(text)
    protected_spans = _find_marker_spans(text, PROTECTED_START, PROTECTED_END)
    slow_spans = _find_marker_spans(text, SLOW_UPDATE_START, SLOW_UPDATE_END)
    guidance = guidance_markdown.strip()
    if not guidance:
        raise ValueError("slow-update guidance_markdown must be non-empty")

    if not slow_spans:
        block = _format_appended_block(text, guidance)
        edit = EditOp(
            "append",
            None,
            block,
            "append slow-update guidance region",
            source_type="slow_update",
        )
        candidate = text + block
        invalid_reason = _invalid_guidance_reason(guidance, candidate)
        if invalid_reason is not None:
            return text, _slow_update_record(
                edit,
                epoch=epoch,
                step=step,
                status="skipped_invalid",
                reason=invalid_reason,
            )
        record = EditApplyRecord(
            edit=edit,
            epoch=epoch,
            step=step,
            status="applied",
            selection_score_before=None,
            selection_score_after=None,
            reason="appended_slow_update_markers",
        )
        return candidate, record

    if len(slow_spans) > 1:
        raise ValueError("Multiple SLOW_UPDATE marker pairs are not supported")

    start, end = slow_spans[0]
    content_start = start + len(SLOW_UPDATE_START)
    content_end = end - len(SLOW_UPDATE_END)
    if _range_intersects(content_start, content_end, protected_spans):
        raise ValueError("SLOW_UPDATE region overlaps a PROTECTED region")

    replacement = f"\n{guidance}\n"
    candidate = text[:content_start] + replacement + text[content_end:]
    edit = EditOp(
        "replace",
        text[start:end],
        text[start:content_start] + replacement + text[content_end:end],
        "replace slow-update guidance region",
        source_type="slow_update",
    )
    invalid_reason = _invalid_guidance_reason(guidance, candidate)
    if invalid_reason is not None:
        return text, _slow_update_record(
            edit,
            epoch=epoch,
            step=step,
            status="skipped_invalid",
            reason=invalid_reason,
        )
    record = EditApplyRecord(
        edit=edit,
        epoch=epoch,
        step=step,
        status="applied",
        selection_score_before=None,
        selection_score_after=None,
    )
    return candidate, record


def _invalid_guidance_reason(guidance: str, candidate: str) -> str | None:
    if looks_like_injection(guidance):
        return "guidance looks like injection"
    if len(candidate) > MAX_DOC_CHARS:
        return f"MAX_DOC_CHARS {MAX_DOC_CHARS} exceeded"
    return None


def _slow_update_record(
    edit: EditOp,
    *,
    epoch: int,
    step: int,
    status: Literal["applied", "skipped_invalid"],
    reason: str | None = None,
) -> EditApplyRecord:
    return EditApplyRecord(
        edit=edit,
        epoch=epoch,
        step=step,
        status=status,
        selection_score_before=None,
        selection_score_after=None,
        reason=reason,
    )


def _format_appended_block(text: str, guidance: str) -> str:
    if not text:
        prefix = ""
    elif text.endswith("\n"):
        prefix = "\n"
    else:
        prefix = "\n\n"
    return f"{prefix}{SLOW_UPDATE_START}\n{guidance}\n{SLOW_UPDATE_END}"


def _rollout_failed(rollout: Any) -> bool:
    is_failure = getattr(rollout, "is_failure", None)
    if is_failure is not None:
        return bool(is_failure)
    return not bool(getattr(rollout, "success", False))


def _rollout_id(rollout: Any, index: int) -> str:
    example_id = getattr(rollout, "example_id", None)
    return str(example_id if example_id not in (None, "") else index)


def _range_intersects(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


__all__ = [
    "SlowUpdateCase",
    "SlowUpdateCategory",
    "apply_slow_update",
    "build_slow_update_probe",
    "categorize_slow_update_rollouts",
    "extract_slow_update_content",
]
