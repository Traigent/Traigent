"""Skill document state and protected-region parsing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

PROTECTED_START = "<!-- PROTECTED -->"
PROTECTED_END = "<!-- /PROTECTED -->"
SLOW_UPDATE_START = "<!-- SLOW_UPDATE -->"
SLOW_UPDATE_END = "<!-- /SLOW_UPDATE -->"


@dataclass(frozen=True, slots=True)
class SkillDocument:
    """Versioned text document optimized by skill training."""

    text: str
    version: int = 0
    parent_hash: str | None = None
    doc_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "doc_hash",
            hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16],
        )


def _find_marker_spans(
    text: str, start_marker: str, end_marker: str
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < len(text):
        next_start = text.find(start_marker, pos)
        next_end = text.find(end_marker, pos)
        if next_start == -1:
            if next_end != -1:
                raise ValueError(f"Unbalanced marker pair: {end_marker!r} before start")
            break
        if next_end != -1 and next_end < next_start:
            raise ValueError(f"Unbalanced marker pair: {end_marker!r} before start")

        close = text.find(end_marker, next_start + len(start_marker))
        if close == -1:
            raise ValueError(f"Unbalanced marker pair: missing {end_marker!r}")

        nested_start = text.find(start_marker, next_start + len(start_marker))
        if nested_start != -1 and nested_start < close:
            raise ValueError(
                f"Nested marker pair is not supported for {start_marker!r}"
            )

        end = close + len(end_marker)
        spans.append((next_start, end))
        pos = end
    return spans


def find_protected_regions(text: str) -> list[tuple[int, int]]:
    """Return inclusive marker spans that must not be edited.

    Both PROTECTED and SLOW_UPDATE regions are guarded in M1. Unbalanced marker
    pairs fail loudly because editing an ambiguously marked document would make
    the validation gate untrustworthy.
    """

    spans = [
        *_find_marker_spans(text, PROTECTED_START, PROTECTED_END),
        *_find_marker_spans(text, SLOW_UPDATE_START, SLOW_UPDATE_END),
    ]
    return sorted(spans)


__all__ = [
    "SkillDocument",
    "find_protected_regions",
    "PROTECTED_START",
    "PROTECTED_END",
    "SLOW_UPDATE_START",
    "SLOW_UPDATE_END",
]
