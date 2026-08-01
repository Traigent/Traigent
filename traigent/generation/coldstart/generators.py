"""Concrete input-only cold-start scenario generators."""

from __future__ import annotations

from typing import Any

from .contracts import (
    ColdStartConfigurationError,
    ScenarioCandidate,
    SystemSpec,
)


def _value_for_annotation(annotation: str, *, seed: int, index: int, name: str) -> Any:
    """Return a deterministic, conservative sample for a supported input type."""
    normalized = annotation.replace("typing.", "").replace(" ", "").lower()
    if normalized in {"str", "optional[str]", "str | none", "none | str"}:
        return f"coldstart-{name}-{seed}-{index}"
    if normalized in {"int", "optional[int]", "int | none", "none | int"}:
        return seed + index
    if normalized in {"float", "optional[float]", "float | none", "none | float"}:
        return float(seed + index)
    if normalized in {"bool", "optional[bool]", "bool | none", "none | bool"}:
        return bool((seed + index) % 2)
    return None


class ContractGroundedGenerator:
    """Deterministically propose inputs from a sufficiently typed static contract.

    This generator does not infer expected outputs. Every proposal must be
    independently grounded by the injected :class:`~.contracts.Oracle` before
    it can be admitted to an evaluation set.
    """

    technique_id = "contract_grounded.v1"

    def propose(
        self, system: SystemSpec, count: int, seed: int
    ) -> list[ScenarioCandidate]:
        if count <= 0:
            raise ColdStartConfigurationError(
                "Scenario proposal count must be positive."
            )
        values = {
            parameter.name: _value_for_annotation(
                parameter.annotation or "", seed=seed, index=0, name=parameter.name
            )
            for parameter in system.parameters
        }
        if not values or any(value is None for value in values.values()):
            return []
        return [
            ScenarioCandidate(
                candidate_id=f"{self.technique_id}:{seed}:{index}",
                inputs={
                    parameter.name: _value_for_annotation(
                        parameter.annotation or "",
                        seed=seed,
                        index=index,
                        name=parameter.name,
                    )
                    for parameter in system.parameters
                },
            )
            for index in range(count)
        ]


__all__ = ["ContractGroundedGenerator"]
