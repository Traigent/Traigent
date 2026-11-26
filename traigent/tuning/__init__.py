"""Continuous Tuning: sugar that emits a TVL 1.7 module (TVL is the source of truth).

``Tuning`` compiles to TVL (``to_tvl``); the TVL reference toolchain validates it
(``validate``).  Without a TVL toolchain that supports 1.7, executing paths refuse
with ``E_TVL_VALIDATOR_UNAVAILABLE``.
"""

from traigent.tuning.audit_sampling import (
    CommittedSampler,
    SamplingCommitment,
    sampler_from_monitor,
)
from traigent.tuning.audit_sampling import verify as verify_sampling
from traigent.tuning.authority import (
    ApproverAuthority,
    load_authority,
)
from traigent.tuning.context import derive_module_context
from traigent.tuning.controller import (
    AcceptanceOutcome,
    Controller,
    HeadLoader,
    StepReport,
    traigent_optimize_fn,
)
from traigent.tuning.errors import (
    E_FLOAT_RATE,
    E_INVALID_TUNING,
    E_NO_APPROVER_AUTHORITY,
    E_NOT_ROUNDTRIPPABLE,
    E_RATE_PRECISION,
    E_TVL_VALIDATION_FAILED,
    E_TVL_VALIDATOR_UNAVAILABLE,
    E_UNSUPPORTED_IN_RELEASE,
    TuningError,
    TvlValidationFailed,
    ValidatorUnavailable,
)
from traigent.tuning.policy import (
    Every,
    FailureRate,
    ModelReleased,
    OnBreach,
    ProposeFix,
    RandomAudit,
    Tuning,
)
from traigent.tuning.state import TuningState, attach_tuning
from traigent.tuning.validator import (
    Diagnostic,
    conditions_trace,
    require_validator,
    validator_available,
)

__all__ = [
    "AcceptanceOutcome",
    "ApproverAuthority",
    "CommittedSampler",
    "SamplingCommitment",
    "Controller",
    "Diagnostic",
    "E_FLOAT_RATE",
    "E_INVALID_TUNING",
    "E_NO_APPROVER_AUTHORITY",
    "E_NOT_ROUNDTRIPPABLE",
    "E_RATE_PRECISION",
    "E_TVL_VALIDATION_FAILED",
    "E_TVL_VALIDATOR_UNAVAILABLE",
    "E_UNSUPPORTED_IN_RELEASE",
    "Every",
    "FailureRate",
    "HeadLoader",
    "ModelReleased",
    "OnBreach",
    "ProposeFix",
    "RandomAudit",
    "StepReport",
    "Tuning",
    "TuningError",
    "TuningState",
    "TvlValidationFailed",
    "ValidatorUnavailable",
    "attach_tuning",
    "sampler_from_monitor",
    "verify_sampling",
    "conditions_trace",
    "derive_module_context",
    "load_authority",
    "require_validator",
    "traigent_optimize_fn",
    "validator_available",
]
