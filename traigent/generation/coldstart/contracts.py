"""Frozen public contracts for offline cold-start evaluation-set generation.

These types describe the boundary between static repository inspection,
scenario proposal, independently produced ground truth, and local artifacts.
They deliberately do not expose execution, scoring, writer, or admission
extension points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from traigent.contract import EvaluationContractReport


COLDSTART_SCHEMA_VERSION = "traigent.coldstart.v1"


class ColdStartOutcome(StrEnum):
    """Whether cold-start construction produced an optimizer-ready eval set."""

    EVAL_SET = "eval_set"
    DISCOVERY_ONLY = "discovery_only"


class DiscoveryGap(StrEnum):
    """Typed reasons a repository cannot safely produce a tuning dataset."""

    NO_ORACLE = "no_oracle"
    UNTYPED_INPUT_CONTRACT = "untyped_input_contract"
    NO_GOLD_DERIVABLE = "no_gold_derivable"
    NO_ELIGIBLE_ROWS = "no_eligible_rows"


class CandidateState(StrEnum):
    """Construction state recorded for each proposed scenario."""

    PROPOSED = "proposed"
    GROUNDED = "grounded"
    ELIGIBLE = "eligible"
    QUARANTINED = "quarantined"


class QuarantineReason(StrEnum):
    """Fail-closed reasons a proposed scenario cannot enter the tuning set."""

    MODEL_PROPOSED_GOLD = "model_proposed_gold"
    NO_GOLD = "no_gold"
    ORACLE_ERROR = "oracle_error"
    INVALID_INPUT = "invalid_input"
    INJECTION_MARKER = "injection_marker"
    DUPLICATE = "duplicate"
    CONTRACT_MISMATCH = "contract_mismatch"


class GroundTruthSource(StrEnum):
    """Origin of a scenario's expected output."""

    SPEC_DERIVED = "spec_derived"
    ORACLE_COMPUTED = "oracle_computed"
    MODEL_PROPOSED = "model_proposed"


class ScoringContract(StrEnum):
    """Supported deterministic comparisons for independently grounded output."""

    EXACT_MATCH = "exact_match"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    SET_EQUALITY = "set_equality"
    JSON_SUBSET = "json_subset"


class ColdStartError(Exception):
    """Base error for cold-start construction failures."""


class ColdStartConfigurationError(ColdStartError, ValueError):
    """Raised when a cold-start extension or configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Static description of one inspected callable parameter."""

    name: str
    annotation: str | None
    required: bool


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Digest of one allowlisted repository file used in static inspection."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SystemSpec:
    """Versioned static callable and repository description supplied to generators."""

    callable_name: str
    module_name: str | None
    parameters: tuple[ParameterSpec, ...]
    return_annotation: str | None
    files: tuple[FileDigest, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Expected output with explicit provenance and comparison semantics."""

    expected_output: Any
    source: GroundTruthSource
    scoring_contract: ScoringContract


@dataclass(frozen=True, slots=True)
class ScenarioCandidate:
    """A proposed input and any independently established ground truth."""

    candidate_id: str
    inputs: Mapping[str, Any]
    ground_truth: GroundTruth | None = None
    state: CandidateState = CandidateState.PROPOSED
    quarantine_reason: QuarantineReason | None = None

    def __post_init__(self) -> None:
        """Prevent later caller mutation from changing a frozen candidate's inputs."""
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


@dataclass(frozen=True, slots=True)
class ColdStartResult:
    """Paths and typed construction outcome returned by the cold-start pipeline."""

    outcome: ColdStartOutcome
    tuning_path: Path | None
    audit_path: Path
    manifest_path: Path
    gaps: tuple[DiscoveryGap, ...] = ()
    counts: Mapping[CandidateState, int] = field(default_factory=dict)
    contract_report: EvaluationContractReport | None = None

    def __post_init__(self) -> None:
        """Preserve immutable result containers and concrete local-path types."""
        for path in (self.tuning_path, self.audit_path, self.manifest_path):
            if path is not None and not isinstance(path, Path):
                raise ColdStartConfigurationError(
                    "Cold-start result artifact paths must be pathlib.Path instances."
                )
        object.__setattr__(self, "gaps", tuple(self.gaps))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))


@runtime_checkable
class ScenarioGenerator(Protocol):
    """A deterministic technique that proposes input scenarios from a system spec."""

    technique_id: str

    def propose(
        self, system: SystemSpec, count: int, seed: int
    ) -> Sequence[ScenarioCandidate]:
        """Propose at most ``count`` scenarios without establishing their gold."""


@runtime_checkable
class Oracle(Protocol):
    """An independent source of ground truth for a candidate input mapping."""

    oracle_id: str
    scoring_contract: ScoringContract

    def ground_truth(self, inputs: Mapping[str, Any]) -> GroundTruth:
        """Produce independently grounded expected output for ``inputs``."""


class ColdStartOptions(BaseModel):
    """Strict local bounds for cold-start static inspection and proposal."""

    model_config = ConfigDict(extra="forbid")

    num_candidates: int = Field(20, gt=0)
    seed: int = 0
    max_files: int = Field(200, gt=0)
    max_file_bytes: int = Field(200_000, gt=0)
    include_globs: tuple[str, ...] = Field(("*.py",), min_length=1)
    dataset_name: str | None = None


__all__ = [
    "COLDSTART_SCHEMA_VERSION",
    "CandidateState",
    "ColdStartConfigurationError",
    "ColdStartError",
    "ColdStartOptions",
    "ColdStartOutcome",
    "ColdStartResult",
    "DiscoveryGap",
    "FileDigest",
    "GroundTruth",
    "GroundTruthSource",
    "Oracle",
    "ParameterSpec",
    "QuarantineReason",
    "ScenarioCandidate",
    "ScenarioGenerator",
    "ScoringContract",
    "SystemSpec",
]
