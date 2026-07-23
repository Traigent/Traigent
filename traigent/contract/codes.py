"""Stable identifiers for the no-execution evaluation-compatibility contract.

``ContractCode`` is the closed, additive-only vocabulary of diagnostics that
:func:`traigent.contract.evaluation.validate_evaluation_contract` may emit.
``EVALUATION_CONTRACT_VERSION`` versions the report *schema and this code
vocabulary* (mirroring the pattern used by
``traigent.economics.contract.CONTRACT_VERSION``).

Stability policy (see ``docs/evaluation_contract.md``):

* Adding a new ``ContractCode`` member or a new report field is a **minor**
  version bump (additive, backward compatible).
* Removing/renaming a member, or changing what an existing code means, is a
  **major** version bump.

Nothing here carries a payload value: these are contract identifiers only.
"""

from __future__ import annotations

from enum import StrEnum

#: Version of the evaluation-contract report schema + ``ContractCode`` vocabulary.
EVALUATION_CONTRACT_VERSION = "1.0.0"


class ContractCode(StrEnum):
    """Closed, additive-only vocabulary of contract diagnostics."""

    # --- Dataset normalization -------------------------------------------------
    #: Dataset could not be normalized; the original error message is preserved.
    DATASET_NORMALIZATION_FAILED = "DATASET_NORMALIZATION_FAILED"
    #: Dataset normalized to zero examples.
    DATASET_EMPTY = "DATASET_EMPTY"
    #: A normalized example has an empty / missing input payload.
    DATASET_MISSING_INPUT = "DATASET_MISSING_INPUT"

    # --- Injection resolution --------------------------------------------------
    #: The same injection option was supplied via both flat kwargs and a bundle.
    INJECTION_OPTIONS_CONFLICT = "INJECTION_OPTIONS_CONFLICT"
    #: Injection mode is removed ("attribute"/"decorator") or otherwise unknown.
    INJECTION_MODE_UNSUPPORTED = "INJECTION_MODE_UNSUPPORTED"
    #: Parameter-mode injection requires a config parameter the function lacks.
    INJECTION_CONFIG_PARAM_MISSING = "INJECTION_CONFIG_PARAM_MISSING"
    #: The provider could not construct its wrapper for another reason.
    INJECTION_PROVIDER_SETUP_FAILED = "INJECTION_PROVIDER_SETUP_FAILED"
    #: Seamless (AST) injection cannot transform this function (best-effort).
    SEAMLESS_INJECTION_UNAVAILABLE = "SEAMLESS_INJECTION_UNAVAILABLE"

    # --- Agent (target function) call shape -----------------------------------
    #: The target function's signature could not be introspected.
    AGENT_SIGNATURE_UNAVAILABLE = "AGENT_SIGNATURE_UNAVAILABLE"
    #: The resolved call shape does not bind to the target function's signature.
    AGENT_BIND_FAILED = "AGENT_BIND_FAILED"

    # --- Evaluator / metric binding -------------------------------------------
    #: A metric function's signature could not be introspected.
    EVALUATOR_SIGNATURE_UNAVAILABLE = "EVALUATOR_SIGNATURE_UNAVAILABLE"
    #: No argument candidate binds to a metric function's signature.
    EVALUATOR_BIND_FAILED = "EVALUATOR_BIND_FAILED"
    #: A metric binds ONLY via bare positional fallback with zero recognized
    #: parameter names -- the silent-misscoring hazard.
    EVALUATOR_NO_RECOGNIZED_PARAMS = "EVALUATOR_NO_RECOGNIZED_PARAMS"
    #: A custom evaluator callback is present; the contract cannot inspect it
    #: without executing it, so it is reported as unsupported (never run).
    CUSTOM_EVALUATOR_UNSUPPORTED = "CUSTOM_EVALUATOR_UNSUPPORTED"

    # --- Meta ------------------------------------------------------------------
    #: An unexpected internal error occurred while building the contract report.
    CONTRACT_INTERNAL_ERROR = "CONTRACT_INTERNAL_ERROR"
