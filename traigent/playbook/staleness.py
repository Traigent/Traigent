"""Computed staleness for agent build playbooks."""

from __future__ import annotations

from datetime import datetime

from traigent.playbook.model import STAGE_ORDER, Playbook, StageStatus


def compute_staleness(playbook: Playbook) -> dict[str, bool]:
    """Return computed staleness by stage.

    Stage S is stale iff some earlier stage in ``STAGE_ORDER`` is pinned and has
    ``pinned_at`` strictly later than S's ``pinned_at``. A comparison is made only
    when both stages are pinned and both timestamps exist; missing timestamps and
    non-pinned stages are not stale.
    """
    stale_by_stage: dict[str, bool] = {}
    earlier_pinned_timestamps: list[datetime] = []

    for stage_name in STAGE_ORDER:
        stage = playbook.stages.get(stage_name)
        if (
            stage is None
            or stage.status is not StageStatus.PINNED
            or stage.pinned_at is None
        ):
            stale_by_stage[stage_name] = False
            continue

        stale_by_stage[stage_name] = any(
            earlier_pinned_at > stage.pinned_at
            for earlier_pinned_at in earlier_pinned_timestamps
        )
        earlier_pinned_timestamps.append(stage.pinned_at)

    return stale_by_stage
