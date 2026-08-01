"""Offline, fail-closed cold-start evaluation-dataset construction.

This pipeline never invokes the customer's target callable.  It statically
inspects the callable, asks a scenario technique for inputs, admits only
independently grounded gold, then writes fixed local artifacts.  Existing
``@traigent.optimize`` remains responsible for executing configurations.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from traigent import __version__ as _SDK_VERSION
from traigent.contract import validate_evaluation_contract
from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.base import Dataset, EvaluationExample

from .contracts import (
    COLDSTART_SCHEMA_VERSION,
    CandidateState,
    ColdStartConfigurationError,
    ColdStartOptions,
    ColdStartOutcome,
    ColdStartResult,
    DiscoveryGap,
    GroundTruth,
    GroundTruthSource,
    Oracle,
    QuarantineReason,
    ScenarioCandidate,
    ScenarioGenerator,
    SystemSpec,
)
from .evidence import (
    MAX_INPUT_BYTES,
    EvidenceAdmission,
    admit_candidate,
    build_audit_row,
    build_tuning_row,
    validate_tuning_row,
)
from .writer import (
    ColdStartArtifactError,
    jsonl_bytes,
    sha256_bytes,
    write_coldstart_artifacts,
)


_TUNING_FILENAME = "coldstart_tuning.jsonl"
_MANIFEST_FILENAME = "coldstart_manifest.json"


def _inspect_system(
    *, func: Callable[..., Any], repo_root: str | Path, options: ColdStartOptions
) -> SystemSpec:
    """Use W1's static inspector without importing or executing user modules."""
    # Deliberately deferred: this keeps the public pipeline importable while the
    # optional static-inspection implementation is unavailable in a partial SDK
    # build, and turns that condition into a typed construction error at use.
    try:
        from .spec import extract_system_spec
    except ImportError as exc:  # pragma: no cover - partial-package guard
        raise ColdStartConfigurationError(
            "Cold-start static inspection is unavailable in this SDK build."
        ) from exc
    return extract_system_spec(func, repo_root=repo_root, options=options)


def _default_generator() -> ScenarioGenerator:
    """Load the non-networked built-in input proposer only when it is needed."""
    try:
        from .generators import ContractGroundedGenerator
    except ImportError as exc:  # pragma: no cover - partial-package guard
        raise ColdStartConfigurationError(
            "Cold-start built-in generation is unavailable in this SDK build."
        ) from exc
    return ContractGroundedGenerator()


def _require_generator(generator: ScenarioGenerator | None) -> ScenarioGenerator:
    selected = generator if generator is not None else _default_generator()
    if not isinstance(selected, ScenarioGenerator):
        raise ColdStartConfigurationError(
            "generator must implement ScenarioGenerator (technique_id and propose)."
        )
    if not isinstance(selected.technique_id, str) or not selected.technique_id:
        raise ColdStartConfigurationError(
            "generator.technique_id must be a non-empty string."
        )
    return selected


def _require_oracle(oracle: Oracle | None) -> Oracle | None:
    if oracle is None:
        return None
    if not isinstance(oracle, Oracle):
        raise ColdStartConfigurationError(
            "oracle must implement Oracle (oracle_id, scoring_contract, ground_truth)."
        )
    if not isinstance(oracle.oracle_id, str) or not oracle.oracle_id:
        raise ColdStartConfigurationError(
            "oracle.oracle_id must be a non-empty string."
        )
    return oracle


def _has_sufficient_input_contract(system: SystemSpec) -> bool:
    return bool(system.parameters) and all(
        parameter.annotation is not None for parameter in system.parameters
    )


def _require_loader_trusted_output(output_dir: str | Path) -> None:
    """Keep generated artifacts eligible for Dataset.from_jsonl's path policy."""
    raw_output = Path(output_dir).expanduser()
    if not raw_output.is_absolute():
        raw_output = Path.cwd() / raw_output
    output = raw_output.resolve(strict=False)
    configured_root = os.environ.get("TRAIGENT_DATASET_ROOT")
    trusted_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else Path.cwd().resolve()
    )
    try:
        output.relative_to(trusted_root)
    except ValueError as exc:
        raise ColdStartConfigurationError(
            "output_dir must be under the current Dataset trusted root; use a "
            "subdirectory of the current working directory or set "
            "TRAIGENT_DATASET_ROOT before generation."
        ) from exc


def _quarantine_reason(reason: str | None) -> QuarantineReason:
    mapping = {
        "ineligible_ground_truth_source": QuarantineReason.MODEL_PROPOSED_GOLD,
        "missing_ground_truth": QuarantineReason.NO_GOLD,
        "missing_expected_output": QuarantineReason.NO_GOLD,
        "input_injection_marker": QuarantineReason.INJECTION_MARKER,
        "duplicate_input": QuarantineReason.DUPLICATE,
        "input_too_large": QuarantineReason.INVALID_INPUT,
        "missing_or_empty_input": QuarantineReason.INVALID_INPUT,
        "input_not_json_serializable": QuarantineReason.INVALID_INPUT,
        "expected_output_not_json_serializable": QuarantineReason.INVALID_INPUT,
        "unsupported_scoring_contract": QuarantineReason.CONTRACT_MISMATCH,
    }
    return mapping.get(reason or "", QuarantineReason.CONTRACT_MISMATCH)


def _counts(counter: Counter[CandidateState]) -> dict[CandidateState, int]:
    return {state: counter.get(state, 0) for state in CandidateState}


def _safe_descriptor(system: SystemSpec) -> dict[str, Any]:
    """Persist structural system information, never callable code or values."""
    return {
        "fingerprint": system.fingerprint,
        "callable_name": system.callable_name,
        "module_name": system.module_name,
        "parameter_names": [parameter.name for parameter in system.parameters],
        "file_count": len(system.files),
    }


def _manifest(
    *,
    outcome: ColdStartOutcome,
    system: SystemSpec | None,
    generator: ScenarioGenerator | None,
    oracle: Oracle | None,
    tuning_rows: Sequence[Mapping[str, Any]] | None,
    counts: Mapping[CandidateState, int],
    gaps: Sequence[DiscoveryGap],
) -> dict[str, Any]:
    payload = jsonl_bytes(tuning_rows) if tuning_rows is not None else None
    return {
        "schema_version": COLDSTART_SCHEMA_VERSION,
        "outcome": outcome.value,
        "dataset_path": _TUNING_FILENAME if payload is not None else None,
        "dataset_sha256": sha256_bytes(payload) if payload is not None else None,
        "holdout_prohibited": True,
        "system": _safe_descriptor(system) if system is not None else None,
        "generator": {"technique_id": generator.technique_id} if generator else None,
        "oracle": (
            {
                "oracle_id": oracle.oracle_id,
                "scoring_contract": oracle.scoring_contract.value,
            }
            if oracle
            else None
        ),
        "scoring_contract": oracle.scoring_contract.value if oracle else None,
        "counts": {state.value: count for state, count in counts.items()},
        "gaps": [gap.value for gap in gaps],
        "created_at": datetime.now(UTC).isoformat(),
        "sdk_version": _SDK_VERSION,
    }


def _write_result(
    *,
    output_dir: str | Path,
    outcome: ColdStartOutcome,
    system: SystemSpec | None,
    generator: ScenarioGenerator | None,
    oracle: Oracle | None,
    tuning_rows: list[dict[str, Any]] | None,
    audit_rows: list[dict[str, Any]],
    counts: Counter[CandidateState],
    gaps: tuple[DiscoveryGap, ...],
    contract_report: Any = None,
) -> ColdStartResult:
    audit_rows.insert(
        0,
        {
            "artifact": "coldstart_audit",
            "schema_version": COLDSTART_SCHEMA_VERSION,
            "outcome": outcome.value,
        },
    )
    manifest = _manifest(
        outcome=outcome,
        system=system,
        generator=generator,
        oracle=oracle,
        tuning_rows=tuning_rows,
        counts=_counts(counts),
        gaps=gaps,
    )
    try:
        paths = write_coldstart_artifacts(
            output_dir=output_dir,
            tuning_rows=tuning_rows,
            audit_rows=audit_rows,
            manifest=manifest,
        )
    except ColdStartArtifactError as exc:
        raise ColdStartConfigurationError(str(exc)) from exc
    return ColdStartResult(
        outcome=outcome,
        tuning_path=paths.tuning_path,
        audit_path=paths.audit_path,
        manifest_path=paths.manifest_path,
        gaps=gaps,
        counts=_counts(counts),
        contract_report=contract_report,
    )


def _discovery_only(
    *,
    output_dir: str | Path,
    system: SystemSpec | None,
    generator: ScenarioGenerator | None,
    oracle: Oracle | None,
    audit_rows: list[dict[str, Any]],
    counts: Counter[CandidateState],
    gaps: tuple[DiscoveryGap, ...],
    contract_report: Any = None,
) -> ColdStartResult:
    return _write_result(
        output_dir=output_dir,
        outcome=ColdStartOutcome.DISCOVERY_ONLY,
        system=system,
        generator=generator,
        oracle=oracle,
        tuning_rows=None,
        audit_rows=audit_rows,
        counts=counts,
        gaps=gaps,
        contract_report=contract_report,
    )


def generate_eval_set(
    *,
    func: Callable[..., Any],
    repo_root: str | Path,
    oracle: Oracle | None,
    output_dir: str | Path,
    generator: ScenarioGenerator | None = None,
    options: ColdStartOptions | None = None,
    budget: ExecutionBudget | None = None,
) -> ColdStartResult:
    """Construct a tuning-only eval set without executing *func*.

    The construction path has no baseline result, target runner, or score
    comparison.  Missing independent ground truth or an insufficient static
    callable contract produces ``DISCOVERY_ONLY`` and never writes a partial
    tuning dataset.
    """
    if not callable(func):
        raise ColdStartConfigurationError("func must be callable.")
    if options is not None and not isinstance(options, ColdStartOptions):
        raise ColdStartConfigurationError("options must be ColdStartOptions or None.")
    if budget is not None and not isinstance(budget, ExecutionBudget):
        raise ColdStartConfigurationError("budget must be ExecutionBudget or None.")
    selected_options = options or ColdStartOptions()
    _require_loader_trusted_output(output_dir)
    selected_oracle = _require_oracle(oracle)
    audit_rows: list[dict[str, Any]] = []
    counts: Counter[CandidateState] = Counter()

    if selected_oracle is None:
        return _discovery_only(
            output_dir=output_dir,
            system=None,
            generator=None,
            oracle=None,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(DiscoveryGap.NO_ORACLE,),
        )

    try:
        system = _inspect_system(
            func=func, repo_root=repo_root, options=selected_options
        )
    except ColdStartConfigurationError:
        return _discovery_only(
            output_dir=output_dir,
            system=None,
            generator=None,
            oracle=selected_oracle,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(DiscoveryGap.UNTYPED_INPUT_CONTRACT,),
        )
    if not _has_sufficient_input_contract(system):
        return _discovery_only(
            output_dir=output_dir,
            system=system,
            generator=None,
            oracle=selected_oracle,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(DiscoveryGap.UNTYPED_INPUT_CONTRACT,),
        )

    selected_generator = _require_generator(generator)
    proposed = selected_generator.propose(
        system, selected_options.num_candidates, selected_options.seed
    )
    if (
        not isinstance(proposed, Sequence)
        or len(proposed) > selected_options.num_candidates
    ):
        raise ColdStartConfigurationError(
            "generator.propose must return at most count ScenarioCandidate values."
        )
    if budget is not None:
        budget.record_external(cost_usd=None, examples=len(proposed))

    seen_inputs: set[str] = set()
    eligible_rows: list[dict[str, Any]] = []
    independently_grounded = 0
    oracle_calls = 0
    for candidate in proposed:
        if not isinstance(candidate, ScenarioCandidate):
            raise ColdStartConfigurationError(
                "generator.propose must return ScenarioCandidate values."
            )
        counts[CandidateState.PROPOSED] += 1
        grounded = candidate
        if (
            candidate.ground_truth is None
            or candidate.ground_truth.source == GroundTruthSource.MODEL_PROPOSED
        ):
            try:
                oracle_truth = selected_oracle.ground_truth(candidate.inputs)
                oracle_calls += 1
            except Exception:
                counts[CandidateState.QUARANTINED] += 1
                audit_rows.append(
                    build_audit_row(
                        candidate=candidate,
                        admission=EvidenceAdmission(
                            False, "", QuarantineReason.ORACLE_ERROR.value
                        ),
                        schema_version=COLDSTART_SCHEMA_VERSION,
                    )
                )
                continue
            if (
                not isinstance(oracle_truth, GroundTruth)
                or oracle_truth.source != GroundTruthSource.ORACLE_COMPUTED
            ):
                counts[CandidateState.QUARANTINED] += 1
                audit_rows.append(
                    build_audit_row(
                        candidate=candidate,
                        admission=EvidenceAdmission(
                            False, "", QuarantineReason.CONTRACT_MISMATCH.value
                        ),
                        schema_version=COLDSTART_SCHEMA_VERSION,
                    )
                )
                continue
            grounded = ScenarioCandidate(
                candidate_id=candidate.candidate_id,
                inputs=candidate.inputs,
                ground_truth=oracle_truth,
                state=CandidateState.GROUNDED,
            )
            counts[CandidateState.GROUNDED] += 1
            independently_grounded += 1
        elif candidate.ground_truth.source == GroundTruthSource.SPEC_DERIVED:
            counts[CandidateState.GROUNDED] += 1
            independently_grounded += 1

        admission = admit_candidate(
            grounded,
            seen_input_digests=seen_inputs,
            max_input_bytes=MAX_INPUT_BYTES,
        )
        audit = build_audit_row(
            candidate=grounded,
            admission=admission,
            schema_version=COLDSTART_SCHEMA_VERSION,
        )
        if not admission.admitted:
            audit["quarantine_reason"] = _quarantine_reason(
                admission.quarantine_reason
            ).value
        audit_rows.append(audit)
        if not admission.admitted:
            counts[CandidateState.QUARANTINED] += 1
            continue
        counts[CandidateState.ELIGIBLE] += 1
        eligible_rows.append(
            build_tuning_row(
                candidate=grounded,
                schema_version=COLDSTART_SCHEMA_VERSION,
                oracle_id=selected_oracle.oracle_id,
                generator_id=selected_generator.technique_id,
                seed=selected_options.seed,
                system_fingerprint=system.fingerprint,
            )
        )

    if budget is not None and oracle_calls:
        budget.record_external(cost_usd=None, examples=oracle_calls)

    if not eligible_rows:
        gap = (
            DiscoveryGap.NO_GOLD_DERIVABLE
            if independently_grounded == 0
            else DiscoveryGap.NO_ELIGIBLE_ROWS
        )
        return _discovery_only(
            output_dir=output_dir,
            system=system,
            generator=selected_generator,
            oracle=selected_oracle,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(gap,),
        )

    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data=dict(row["input"]),
                expected_output=row["expected_output"],
                metadata={
                    "example_id": row["example_id"],
                    "traigent_coldstart": row["traigent_coldstart"],
                },
            )
            for row in eligible_rows
        ],
        name=selected_options.dataset_name or "coldstart_tuning",
    )
    report = validate_evaluation_contract(func=func, dataset=dataset)
    if not report.ok:
        audit_rows.append(
            {
                "artifact": "coldstart_audit",
                "schema_version": COLDSTART_SCHEMA_VERSION,
                "state": CandidateState.QUARANTINED.value,
                "quarantine_reason": QuarantineReason.CONTRACT_MISMATCH.value,
            }
        )
        return _discovery_only(
            output_dir=output_dir,
            system=system,
            generator=selected_generator,
            oracle=selected_oracle,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(DiscoveryGap.UNTYPED_INPUT_CONTRACT,),
            contract_report=report,
        )

    return _write_result(
        output_dir=output_dir,
        outcome=ColdStartOutcome.EVAL_SET,
        system=system,
        generator=selected_generator,
        oracle=selected_oracle,
        tuning_rows=eligible_rows,
        audit_rows=audit_rows,
        counts=counts,
        gaps=(),
        contract_report=report,
    )


def assert_optimizer_eligible(tuning_path: str | Path) -> None:
    """Check local cold-start artifact integrity before passing it to optimize.

    This verifies accidental alteration and construction-policy eligibility.  It
    is not authentication and does not prove the semantic correctness of an
    oracle or prevent a party who can edit both dataset and manifest from
    recomputing hashes.
    """
    dataset_path = Path(tuning_path)
    if dataset_path.name != _TUNING_FILENAME or dataset_path.suffix != ".jsonl":
        raise ColdStartConfigurationError("Expected a coldstart_tuning.jsonl path.")
    if dataset_path.is_symlink():
        raise ColdStartConfigurationError(
            "Cold-start integrity refuses symlink datasets."
        )
    manifest_path = dataset_path.parent / _MANIFEST_FILENAME
    if manifest_path.is_symlink():
        raise ColdStartConfigurationError(
            "Cold-start integrity refuses symlink manifests."
        )
    try:
        raw_dataset = dataset_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColdStartConfigurationError(
            "Cold-start dataset and manifest must be readable local JSON artifacts."
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ColdStartConfigurationError("Cold-start manifest must be a JSON object.")
    if (
        manifest.get("schema_version") != COLDSTART_SCHEMA_VERSION
        or manifest.get("outcome") != ColdStartOutcome.EVAL_SET.value
        or manifest.get("holdout_prohibited") is not True
        or manifest.get("dataset_path") != _TUNING_FILENAME
        or manifest.get("dataset_sha256") != sha256_bytes(raw_dataset)
    ):
        raise ColdStartConfigurationError("Cold-start manifest integrity check failed.")

    try:
        Dataset.from_jsonl(str(dataset_path))
        parsed_rows = [
            json.loads(line)
            for line in raw_dataset.decode("utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        raise ColdStartConfigurationError(
            "Cold-start tuning data is not Dataset.from_jsonl-compatible."
        ) from exc
    if not parsed_rows or not all(isinstance(row, Mapping) for row in parsed_rows):
        raise ColdStartConfigurationError(
            "Cold-start tuning dataset is empty or malformed."
        )

    seen_inputs: set[str] = set()
    seen_example_ids: set[str] = set()
    for row in parsed_rows:
        reason = validate_tuning_row(
            row,
            expected_schema_version=COLDSTART_SCHEMA_VERSION,
            seen_input_digests=seen_inputs,
            max_input_bytes=MAX_INPUT_BYTES,
        )
        example_id = row.get("example_id")
        if (
            reason is not None
            or not isinstance(example_id, str)
            or example_id in seen_example_ids
        ):
            raise ColdStartConfigurationError(
                "Cold-start tuning row fails construction-evidence eligibility."
            )
        seen_example_ids.add(example_id)


__all__ = ["assert_optimizer_eligible", "generate_eval_set"]
