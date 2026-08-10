"""Rejected edit memory for epoch-local negative feedback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .edits import EditOp


@dataclass(frozen=True, slots=True)
class RejectedEditAttempt:
    """One selection-gate rejection and the edits that produced it."""

    edits: tuple[EditOp, ...]
    selection_delta: float
    epoch: int
    step: int


class RejectedEditBuffer:
    """Small FIFO buffer of failed gate attempts for optimizer prompts."""

    def __init__(
        self, max_entries: int, *, persist_across_epochs: bool = False
    ) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        self.max_entries = max_entries
        self.persist_across_epochs = persist_across_epochs
        self._attempts: deque[RejectedEditAttempt] = deque(maxlen=max_entries or 1)

    def clear_epoch(self) -> None:
        """Clear epoch-local entries unless configured to persist."""

        if not self.persist_across_epochs:
            self.clear()

    def clear(self) -> None:
        self._attempts.clear()

    def record(
        self,
        edits: list[EditOp],
        *,
        selection_delta: float,
        epoch: int,
        step: int,
    ) -> None:
        if self.max_entries == 0 or not edits:
            return
        self._attempts.append(
            RejectedEditAttempt(
                edits=tuple(edits),
                selection_delta=float(selection_delta),
                epoch=epoch,
                step=step,
            )
        )

    def digest(self, max_chars: int = 2000) -> str:
        """Return compact prompt text summarizing rejected edits."""

        if max_chars <= 0 or not self._attempts:
            return ""

        lines: list[str] = []
        for attempt in reversed(self._attempts):
            for edit in attempt.edits:
                target = edit.target if edit.target is not None else edit.content or ""
                target = _truncate(target)
                lines.append(
                    "previously tried and REJECTED: "
                    f"{edit.op} on {target!r} -> Δ{attempt.selection_delta:+.4f} "
                    f"(epoch={attempt.epoch}, step={attempt.step})"
                )

        digest = "\n".join(lines)
        if len(digest) <= max_chars:
            return digest
        return digest[: max_chars - 3].rstrip() + "..."


def _truncate(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


__all__ = ["RejectedEditAttempt", "RejectedEditBuffer"]
