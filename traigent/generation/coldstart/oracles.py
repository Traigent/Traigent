"""Local deterministic oracle adapter for cold-start ground truth."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .contracts import (
    ColdStartConfigurationError,
    GroundTruth,
    GroundTruthSource,
    ScoringContract,
)


class CallableOracle:
    """Wrap an explicitly supplied local ground-truth callable as an oracle."""

    def __init__(
        self,
        ground_truth_callable: Callable[[Mapping[str, Any]], Any],
        *,
        oracle_id: str = "callable_oracle.v1",
    ) -> None:
        if not callable(ground_truth_callable):
            raise ColdStartConfigurationError("ground_truth_callable must be callable.")
        if not isinstance(oracle_id, str) or not oracle_id.strip():
            raise ColdStartConfigurationError("oracle_id must be a non-empty string.")
        self._ground_truth_callable = ground_truth_callable
        self.oracle_id = oracle_id
        self.scoring_contract = ScoringContract.EXACT_MATCH

    def ground_truth(self, inputs: Mapping[str, Any]) -> GroundTruth:
        """Call the supplied local oracle and record its independent provenance."""
        return GroundTruth(
            expected_output=self._ground_truth_callable(inputs),
            source=GroundTruthSource.ORACLE_COMPUTED,
            scoring_contract=self.scoring_contract,
        )


__all__ = ["CallableOracle"]
