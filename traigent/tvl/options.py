"""Shared TVL option models used across the SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-TVLSPEC REQ-TVLSPEC-012 SYNC-OptimizationFlow

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_validator

from traigent.tvl.models import RegistryResolver


class TVLOptions(BaseModel):
    """Structured options for supplying TVL specification data."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    spec_path: str
    environment: str | None = None
    validate_constraints: bool = True
    registry_resolver: RegistryResolver | None = None
    apply_evaluation_set: bool = True
    apply_configuration_space: bool = True
    apply_objectives: bool = True
    apply_constraints: bool = True
    apply_budget: bool = True

    @field_validator("spec_path")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> str:
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def merged_with(self, *, environment: str | None = None) -> TVLOptions:
        """Return a copy with the provided environment overriding the current one."""

        if environment and environment != self.environment:
            return cast(
                TVLOptions, self.model_copy(update={"environment": environment})
            )
        return self

    def to_kwargs(self) -> dict[str, Any]:
        """Return loader kwargs for convenience."""

        return {
            "spec_path": self.spec_path,
            "environment": self.environment,
            "validate_constraints": self.validate_constraints,
            "registry_resolver": self.registry_resolver,
        }
