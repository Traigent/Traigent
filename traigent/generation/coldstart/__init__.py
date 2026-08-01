"""Fail-closed offline construction of tuning-only evaluation datasets."""

from __future__ import annotations

from .contracts import (
    COLDSTART_SCHEMA_VERSION,
    CandidateState,
    ColdStartConfigurationError,
    ColdStartError,
    ColdStartInputContractError,
    ColdStartOptions,
    ColdStartOutcome,
    ColdStartResult,
    DiscoveryGap,
    FileDigest,
    GroundTruth,
    GroundTruthSource,
    Oracle,
    ParameterSpec,
    QuarantineReason,
    ScenarioCandidate,
    ScenarioGenerator,
    ScoringContract,
    SystemSpec,
)
from .generators import ContractGroundedGenerator
from .oracles import CallableOracle
from .pipeline import assert_optimizer_eligible, generate_eval_set

__all__ = [
    "COLDSTART_SCHEMA_VERSION",
    "CandidateState",
    "CallableOracle",
    "ColdStartConfigurationError",
    "ColdStartError",
    "ColdStartInputContractError",
    "ColdStartOptions",
    "ColdStartOutcome",
    "ColdStartResult",
    "ContractGroundedGenerator",
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
    "assert_optimizer_eligible",
    "generate_eval_set",
]
