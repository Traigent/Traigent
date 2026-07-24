"""Return model for the no-execution evaluation-compatibility contract.

Frozen dataclasses with ``to_dict()`` -- deliberately plain (no pydantic) so
importing this module is near-free and every report is trivially
JSON-serializable. The report reports configuration **keys only**, never
configuration values, so a serialized report cannot leak a secret.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractFinding:
    """A single diagnostic emitted while validating the contract.

    ``severity == "error"`` is the only severity that flips
    :attr:`EvaluationContractReport.ok` to ``False``. ``warning`` / ``info``
    are advisory and never fail the report.
    """

    code: str
    severity: str  # "error" | "warning" | "info"
    message: str
    action: str | None = None
    location: str | None = None
    example_index: int | None = None
    metric_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": str(self.code),
            "severity": self.severity,
            "message": self.message,
            "action": self.action,
            "location": self.location,
            "example_index": self.example_index,
            "metric_name": self.metric_name,
        }


@dataclass(frozen=True)
class NormalizedExampleSummary:
    """Shape of one normalized dataset example (never its input *values*)."""

    index: int
    correlation_key: str
    input_is_mapping: bool
    input_type: str
    input_keys: tuple[str, ...] | None
    has_expected_output: bool
    metadata_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "correlation_key": self.correlation_key,
            "input_is_mapping": self.input_is_mapping,
            "input_type": self.input_type,
            "input_keys": (
                list(self.input_keys) if self.input_keys is not None else None
            ),
            "has_expected_output": self.has_expected_output,
            "metadata_keys": list(self.metadata_keys),
        }


@dataclass(frozen=True)
class InjectionSummary:
    """How configuration injection resolves for the target function.

    ``config_keys`` lists configuration KEYS only -- never values -- so the
    serialized report cannot leak a secret sitting in the config.
    """

    effective_mode: str
    source: str
    config_param: str | None
    provider_class: str | None
    config_keys: tuple[str, ...]
    setup_ok: bool
    seamless_injected_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_mode": self.effective_mode,
            "source": self.source,
            "config_param": self.config_param,
            "provider_class": self.provider_class,
            "config_keys": list(self.config_keys),
            "setup_ok": self.setup_ok,
            "seamless_injected_names": list(self.seamless_injected_names),
        }


@dataclass(frozen=True)
class ExampleCallShape:
    """The call the evaluator would make for one example (computed, not executed).

    ``runtime_*`` is what evaluation passes to the provider wrapper;
    ``effective_*`` is what the raw underlying function ultimately receives
    (``runtime_*`` plus the injected config parameter in parameter mode). Only
    counts and keyword *names* are recorded -- never argument values.
    """

    example_id: str
    expanded: bool
    runtime_args_count: int
    runtime_keyword_names: tuple[str, ...]
    effective_positional_count: int
    effective_keyword_names: tuple[str, ...]
    bind_ok: bool
    bind_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "expanded": self.expanded,
            "runtime_args_count": self.runtime_args_count,
            "runtime_keyword_names": list(self.runtime_keyword_names),
            "effective_positional_count": self.effective_positional_count,
            "effective_keyword_names": list(self.effective_keyword_names),
            "bind_ok": self.bind_ok,
            "bind_error": self.bind_error,
        }


@dataclass(frozen=True)
class EvaluatorBinding:
    """How a metric callback would be invoked (computed, not executed)."""

    metric_name: str
    binding_mode: str
    matched_parameters: tuple[str, ...]
    unmatched_parameters: tuple[str, ...]
    bind_ok: bool
    bind_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "binding_mode": self.binding_mode,
            "matched_parameters": list(self.matched_parameters),
            "unmatched_parameters": list(self.unmatched_parameters),
            "bind_ok": self.bind_ok,
            "bind_error": self.bind_error,
        }


@dataclass(frozen=True)
class EvaluationContractReport:
    """Result of a no-execution evaluation-compatibility check.

    ``ok`` is ``True`` iff no finding has ``severity == "error"``. The report is
    a faithful, execution-free model of what a real optimization run *would*
    do with these inputs -- no user function, metric, or custom evaluator is
    ever called to produce it.
    """

    ok: bool
    sdk_version: str
    contract_version: str
    dataset_name: str
    dataset_source: str | None
    examples: tuple[NormalizedExampleSummary, ...]
    injection: InjectionSummary
    call_shapes: tuple[ExampleCallShape, ...]
    evaluator_bindings: tuple[EvaluatorBinding, ...]
    unsupported: tuple[str, ...]
    findings: tuple[ContractFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sdk_version": self.sdk_version,
            "contract_version": self.contract_version,
            "dataset_name": self.dataset_name,
            "dataset_source": self.dataset_source,
            "examples": [example.to_dict() for example in self.examples],
            "injection": self.injection.to_dict(),
            "call_shapes": [shape.to_dict() for shape in self.call_shapes],
            "evaluator_bindings": [
                binding.to_dict() for binding in self.evaluator_bindings
            ],
            "unsupported": list(self.unsupported),
            "findings": [finding.to_dict() for finding in self.findings],
        }
