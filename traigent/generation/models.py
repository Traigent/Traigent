"""Client-side mirror of the GuidancePlan contract.

These DTOs mirror ``TraigentSchema/.../guidance/*`` and the backend Pydantic
models. The client only ever *consumes* a GuidancePlan (selection: seed_ref +
action verb + coarse priority) and generates locally from it. The plan carries
no tuning signals, so there is nothing sensitive to parse here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GuidanceAction(StrEnum):
    GENERATE_SIMILAR = "generate_similar"
    GENERATE_HARDER = "generate_harder"
    DIVERSIFY_AROUND = "diversify_around"
    REWRITE_PROMPT = "rewrite_prompt"


class CoarsePriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PlanKind(StrEnum):
    PROMPT_REWRITE = "prompt_rewrite"
    BENCHMARK_GUIDE = "benchmark_guide"


@dataclass
class GuidancePlanItem:
    seed_ref: str
    action: GuidanceAction
    coarse_priority: CoarsePriority

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuidancePlanItem:
        return cls(
            seed_ref=str(data["seed_ref"]),
            action=GuidanceAction(data["action"]),
            coarse_priority=CoarsePriority(data["coarse_priority"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_ref": self.seed_ref,
            "action": self.action.value,
            "coarse_priority": self.coarse_priority.value,
        }


@dataclass
class GuidancePlan:
    plan_id: str
    policy_version: str
    plan_kind: PlanKind
    items: list[GuidancePlanItem]
    plan_token: str
    expires_at: str
    total_generations: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuidancePlan:
        budget = data.get("plan_budget") or {}
        return cls(
            plan_id=str(data["plan_id"]),
            policy_version=str(data["policy_version"]),
            plan_kind=PlanKind(data["plan_kind"]),
            items=[GuidancePlanItem.from_dict(it) for it in data.get("items", [])],
            plan_token=str(data["plan_token"]),
            expires_at=str(data["expires_at"]),
            total_generations=int(budget.get("total_generations", 0)),
        )

    def items_for(self, *actions: GuidanceAction) -> list[GuidancePlanItem]:
        """Plan items whose action is one of ``actions`` (e.g. filter rewrite vs synth)."""
        wanted = set(actions)
        return [it for it in self.items if it.action in wanted]


@dataclass
class GuidancePlanRequest:
    plan_kind: PlanKind
    seed_scope: str = "auto"
    max_items: int | None = None
    max_total_generations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "plan_kind": self.plan_kind.value,
            "seed_scope": self.seed_scope,
        }
        budget: dict[str, int] = {}
        if self.max_items is not None:
            budget["max_items"] = self.max_items
        if self.max_total_generations is not None:
            budget["max_total_generations"] = self.max_total_generations
        if budget:
            body["budget"] = budget
        return body


@dataclass
class GuidanceResultItem:
    seed_ref: str
    action: GuidanceAction
    generated_count: int
    new_example_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_ref": self.seed_ref,
            "action": self.action.value,
            "generated_count": self.generated_count,
            "new_example_refs": list(self.new_example_refs),
        }


@dataclass
class GuidanceResultSubmission:
    plan_id: str
    plan_token: str
    results: list[GuidanceResultItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_token": self.plan_token,
            "results": [r.to_dict() for r in self.results],
        }


__all__ = [
    "GuidanceAction",
    "CoarsePriority",
    "PlanKind",
    "GuidancePlanItem",
    "GuidancePlan",
    "GuidancePlanRequest",
    "GuidanceResultItem",
    "GuidanceResultSubmission",
]
