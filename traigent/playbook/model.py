"""Dataclasses for the agent build playbook."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class StageStatus(StrEnum):
    """Lifecycle states for an agent build playbook stage."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PINNED = "pinned"
    DEPRECATED = "deprecated"


STAGE_ORDER = ("dataset", "metric", "evaluator", "optimize", "gate")


@dataclass
class Stage:
    """Parsed representation of a single playbook stage."""

    status: StageStatus
    pinned_at: datetime | None = None
    deprecation_reason: str | None = None
    pin: dict[str, Any] | None = None


@dataclass
class Playbook:
    """Parsed agent build playbook plus the original mapping."""

    playbook_version: str
    agent: dict[str, Any]
    stages: dict[str, Stage]
    provenance: dict[str, Any] | None
    raw: dict[str, Any]
