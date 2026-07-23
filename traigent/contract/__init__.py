"""No-execution evaluation-compatibility contract for the Traigent SDK.

Public entry point: :func:`validate_evaluation_contract`. It statically checks
whether a (function, dataset, evaluator, injection) combination would work under
a real optimization run -- without ever calling the function, a metric, or a
custom evaluator, entering a configuration context, or touching the network.

See ``docs/evaluation_contract.md`` for the stability guarantees.
"""

from __future__ import annotations

from traigent.contract.codes import EVALUATION_CONTRACT_VERSION, ContractCode
from traigent.contract.evaluation import validate_evaluation_contract
from traigent.contract.report import (
    ContractFinding,
    EvaluationContractReport,
    EvaluatorBinding,
    ExampleCallShape,
    InjectionSummary,
    NormalizedExampleSummary,
)

__all__ = [
    "EVALUATION_CONTRACT_VERSION",
    "ContractCode",
    "ContractFinding",
    "EvaluationContractReport",
    "EvaluatorBinding",
    "ExampleCallShape",
    "InjectionSummary",
    "NormalizedExampleSummary",
    "validate_evaluation_contract",
]
