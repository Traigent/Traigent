"""Continuous Tuning: sugar that emits a TVL 1.7 module (TVL is the source of truth).

``Tuning`` compiles to TVL (``to_tvl``); the TVL reference toolchain validates it
(``validate``).  Without a TVL toolchain that supports 1.7, executing paths refuse
with ``E_TVL_VALIDATOR_UNAVAILABLE``.
"""

from traigent.tuning.authority import (
    ApproverAuthority,
    EventVerifier,
    load_authority,
    sign_event,
)
from traigent.tuning.context import derive_module_context
from traigent.tuning.controller import (
    AcceptanceOutcome,
    Controller,
    EventReceipt,
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
    "Controller",
    "Diagnostic",
    "E_FLOAT_RATE",
    "E_INVALID_TUNING",
    "E_NO_APPROVER_AUTHORITY",
    "E_NOT_ROUNDTRIPPABLE",
    "E_RATE_PRECISION",
    "E_TVL_VALIDATION_FAILED",
    "E_TVL_VALIDATOR_UNAVAILABLE",
    "EventReceipt",
    "EventVerifier",
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
    "conditions_trace",
    "derive_module_context",
    "load_authority",
    "require_validator",
    "sign_event",
    "traigent_optimize_fn",
    "validator_available",
]
