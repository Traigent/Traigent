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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast

from traigent import __version__ as _SDK_VERSION
from traigent.contract import validate_evaluation_contract
from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.base import Dataset, EvaluationExample

from .contracts import (
    COLDSTART_SCHEMA_VERSION,
    CandidateState,
    ColdStartConfigurationError,
    ColdStartInputContractError,
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
    ScoringContract,
    SystemSpec,
)
from .evidence import (
    MAX_INPUT_BYTES,
    EvidenceAdmission,
    _screen_candidate_inputs,
    admit_candidate,
    build_audit_row,
    build_tuning_row,
    validate_tuning_row,
)
from .generators import ContractGroundedGenerator
from .spec import extract_system_spec
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
    return cast(
        SystemSpec,
        extract_system_spec(func, repo_root=repo_root, options=options),
    )


def _default_generator() -> ScenarioGenerator:
    """Return the non-networked built-in input proposer."""
    return cast(ScenarioGenerator, ContractGroundedGenerator())


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
    if not isinstance(oracle.oracle_id, str) or not oracle.oracle_id.strip():
        raise ColdStartConfigurationError(
            "oracle.oracle_id must be a non-empty string."
        )
    if not isinstance(oracle.scoring_contract, ScoringContract):
        raise ColdStartConfigurationError(
            "oracle.scoring_contract must be a ScoringContract enum value."
        )
    if oracle.scoring_contract is not ScoringContract.EXACT_MATCH:
        raise ColdStartConfigurationError(
            "oracle.scoring_contract must be ScoringContract.EXACT_MATCH."
        )
    return oracle


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
        "unsupported_expected_output": QuarantineReason.CONTRACT_MISMATCH,
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
        "inspection_truncated": system.inspection_truncated,
        "skipped_file_count": system.skipped_file_count,
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


def _validate_generate_arguments(
    *,
    func: Callable[..., Any],
    options: ColdStartOptions | None,
    budget: ExecutionBudget | None,
) -> ColdStartOptions:
    """Validate the public inputs that must fail before artifact creation."""
    if not callable(func):
        raise ColdStartConfigurationError("func must be callable.")
    if options is not None and not isinstance(options, ColdStartOptions):
        raise ColdStartConfigurationError("options must be ColdStartOptions or None.")
    if budget is not None and not isinstance(budget, ExecutionBudget):
        raise ColdStartConfigurationError("budget must be ExecutionBudget or None.")
    return options or ColdStartOptions()


def _inspect_or_discovery_only(
    *,
    func: Callable[..., Any],
    repo_root: str | Path,
    options: ColdStartOptions,
    output_dir: str | Path,
    oracle: Oracle,
    audit_rows: list[dict[str, Any]],
    counts: Counter[CandidateState],
) -> tuple[SystemSpec | None, ColdStartResult | None]:
    """Run static inspection or persist its truthful discovery-only outcome."""
    try:
        return _inspect_system(func=func, repo_root=repo_root, options=options), None
    except ColdStartInputContractError as exc:
        return None, _discovery_only(
            output_dir=output_dir,
            system=None,
            generator=None,
            oracle=oracle,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(exc.gap,),
        )
    except ColdStartConfigurationError:
        return None, _discovery_only(
            output_dir=output_dir,
            system=None,
            generator=None,
            oracle=oracle,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(DiscoveryGap.STATIC_INSPECTION_FAILED,),
        )


def _proposals_or_discovery_only(
    *,
    generator: ScenarioGenerator | None,
    system: SystemSpec,
    options: ColdStartOptions,
    budget: ExecutionBudget | None,
    output_dir: str | Path,
    oracle: Oracle,
    audit_rows: list[dict[str, Any]],
    counts: Counter[CandidateState],
) -> tuple[ScenarioGenerator, Sequence[ScenarioCandidate], ColdStartResult | None]:
    """Propose bounded scenarios or record the distinct empty-proposal outcome."""
    selected_generator = _require_generator(generator)
    proposed = selected_generator.propose(system, options.num_candidates, options.seed)
    if not isinstance(proposed, Sequence) or len(proposed) > options.num_candidates:
        raise ColdStartConfigurationError(
            "generator.propose must return at most count ScenarioCandidate values."
        )
    if budget is not None:
        budget.record_external(cost_usd=None, examples=len(proposed))
    if proposed:
        return selected_generator, proposed, None
    return (
        selected_generator,
        proposed,
        _discovery_only(
            output_dir=output_dir,
            system=system,
            generator=selected_generator,
            oracle=oracle,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(DiscoveryGap.NO_SCENARIOS_PROPOSED,),
        ),
    )


def _record_oracle_quarantine(
    *,
    candidate: ScenarioCandidate,
    reason: QuarantineReason,
    audit_rows: list[dict[str, Any]],
    counts: Counter[CandidateState],
) -> None:
    """Add one opaque oracle failure to the audit trail and counters."""
    counts[CandidateState.QUARANTINED] += 1
    audit_rows.append(
        build_audit_row(
            candidate=candidate,
            admission=EvidenceAdmission(False, "", reason.value),
            schema_version=COLDSTART_SCHEMA_VERSION,
        )
    )


def _record_input_quarantine(
    *,
    candidate: ScenarioCandidate,
    admission: EvidenceAdmission,
    audit_rows: list[dict[str, Any]],
    counts: Counter[CandidateState],
) -> None:
    """Record a proposal rejected by concrete screening before oracle use."""
    audit = build_audit_row(
        candidate=candidate,
        admission=admission,
        schema_version=COLDSTART_SCHEMA_VERSION,
    )
    audit["quarantine_reason"] = _quarantine_reason(admission.quarantine_reason).value
    audit_rows.append(audit)
    counts[CandidateState.QUARANTINED] += 1


def _oracle_truth_or_quarantine_reason(
    candidate: ScenarioCandidate, oracle: Oracle
) -> tuple[GroundTruth | None, QuarantineReason | None, int]:
    """Call the injected oracle and reject anything outside its strict contract."""
    oracle_call_count = 1
    try:
        oracle_truth = oracle.ground_truth(candidate.inputs)
    except Exception:
        return None, QuarantineReason.ORACLE_ERROR, oracle_call_count
    if oracle_truth is None:
        return None, QuarantineReason.NO_GOLD, oracle_call_count
    if (
        not isinstance(oracle_truth, GroundTruth)
        or oracle_truth.source is not GroundTruthSource.ORACLE_COMPUTED
        or oracle_truth.scoring_contract is not ScoringContract.EXACT_MATCH
    ):
        return None, QuarantineReason.CONTRACT_MISMATCH, oracle_call_count
    return oracle_truth, None, oracle_call_count


def _ground_candidate(
    *,
    candidate: ScenarioCandidate,
    oracle: Oracle,
    generator: ScenarioGenerator,
    options: ColdStartOptions,
    system: SystemSpec,
    seen_proposal_inputs: set[str],
    seen_admitted_inputs: set[str],
    audit_rows: list[dict[str, Any]],
    counts: Counter[CandidateState],
) -> tuple[dict[str, Any] | None, bool, int]:
    """Oracle-ground one proposal and return an eligible row when admissible."""
    counts[CandidateState.PROPOSED] += 1
    _, input_admission = _screen_candidate_inputs(
        candidate,
        seen_input_digests=seen_proposal_inputs,
        max_input_bytes=MAX_INPUT_BYTES,
    )
    assert input_admission is not None
    if not input_admission.admitted:
        _record_input_quarantine(
            candidate=candidate,
            admission=input_admission,
            audit_rows=audit_rows,
            counts=counts,
        )
        return None, True, 0
    seen_proposal_inputs.add(input_admission.input_digest)

    oracle_truth, quarantine_reason, oracle_call_count = (
        _oracle_truth_or_quarantine_reason(candidate, oracle)
    )
    if quarantine_reason is not None:
        _record_oracle_quarantine(
            candidate=candidate,
            reason=quarantine_reason,
            audit_rows=audit_rows,
            counts=counts,
        )
        return None, False, oracle_call_count
    assert oracle_truth is not None
    grounded = ScenarioCandidate(
        candidate_id=candidate.candidate_id,
        inputs=candidate.inputs,
        ground_truth=oracle_truth,
        state=CandidateState.GROUNDED,
    )
    counts[CandidateState.GROUNDED] += 1
    admission = admit_candidate(
        grounded,
        seen_input_digests=seen_admitted_inputs,
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
        counts[CandidateState.QUARANTINED] += 1
        return (
            None,
            admission.quarantine_reason != "missing_expected_output",
            oracle_call_count,
        )
    audit_rows.append(audit)
    counts[CandidateState.ELIGIBLE] += 1
    return (
        build_tuning_row(
            candidate=grounded,
            schema_version=COLDSTART_SCHEMA_VERSION,
            oracle_id=oracle.oracle_id,
            generator_id=generator.technique_id,
            seed=options.seed,
            system_fingerprint=system.fingerprint,
        ),
        True,
        oracle_call_count,
    )


def _ground_proposals(
    *,
    proposed: Sequence[ScenarioCandidate],
    oracle: Oracle,
    generator: ScenarioGenerator,
    options: ColdStartOptions,
    system: SystemSpec,
    audit_rows: list[dict[str, Any]],
    counts: Counter[CandidateState],
) -> tuple[list[dict[str, Any]], int, int]:
    """Ground every valid proposal and report actual oracle invocation count."""
    seen_proposal_inputs: set[str] = set()
    seen_admitted_inputs: set[str] = set()
    eligible_rows: list[dict[str, Any]] = []
    non_gold_blockers = 0
    oracle_call_count = 0
    for candidate in proposed:
        if not isinstance(candidate, ScenarioCandidate):
            raise ColdStartConfigurationError(
                "generator.propose must return ScenarioCandidate values."
            )
        row, has_non_gold_blocker, candidate_oracle_calls = _ground_candidate(
            candidate=candidate,
            oracle=oracle,
            generator=generator,
            options=options,
            system=system,
            seen_proposal_inputs=seen_proposal_inputs,
            seen_admitted_inputs=seen_admitted_inputs,
            audit_rows=audit_rows,
            counts=counts,
        )
        non_gold_blockers += int(has_non_gold_blocker)
        oracle_call_count += candidate_oracle_calls
        if row is not None:
            eligible_rows.append(row)
    return eligible_rows, non_gold_blockers, oracle_call_count


def _evaluation_dataset(
    rows: Sequence[Mapping[str, Any]], options: ColdStartOptions
) -> Dataset:
    """Translate fixed persisted row shapes into the evaluator's in-memory type."""
    return Dataset(
        examples=[
            EvaluationExample(
                input_data=dict(row["input"]),
                expected_output=row["expected_output"],
                metadata={
                    "example_id": row["example_id"],
                    "traigent_coldstart": row["traigent_coldstart"],
                },
            )
            for row in rows
        ],
        name=options.dataset_name or "coldstart_tuning",
    )


def _append_evaluation_contract_audit(audit_rows: list[dict[str, Any]]) -> None:
    """Record a value-free audit item when final call-shape validation fails."""
    audit_rows.append(
        {
            "artifact": "coldstart_audit",
            "schema_version": COLDSTART_SCHEMA_VERSION,
            "state": CandidateState.QUARANTINED.value,
            "quarantine_reason": QuarantineReason.CONTRACT_MISMATCH.value,
        }
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
    selected_options = _validate_generate_arguments(
        func=func, options=options, budget=budget
    )
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

    system, discovery_result = _inspect_or_discovery_only(
        func=func,
        repo_root=repo_root,
        options=selected_options,
        output_dir=output_dir,
        oracle=selected_oracle,
        audit_rows=audit_rows,
        counts=counts,
    )
    if discovery_result is not None:
        return discovery_result
    assert system is not None
    selected_generator, proposed, discovery_result = _proposals_or_discovery_only(
        generator=generator,
        system=system,
        options=selected_options,
        budget=budget,
        output_dir=output_dir,
        oracle=selected_oracle,
        audit_rows=audit_rows,
        counts=counts,
    )
    if discovery_result is not None:
        return discovery_result

    eligible_rows, non_gold_blockers, oracle_call_count = _ground_proposals(
        proposed=proposed,
        oracle=selected_oracle,
        generator=selected_generator,
        options=selected_options,
        system=system,
        audit_rows=audit_rows,
        counts=counts,
    )
    if budget is not None and oracle_call_count:
        budget.record_external(cost_usd=None, examples=oracle_call_count)

    if not eligible_rows:
        gap = (
            DiscoveryGap.NO_GOLD_DERIVABLE
            if non_gold_blockers == 0
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

    dataset = _evaluation_dataset(eligible_rows, selected_options)
    report = validate_evaluation_contract(func=func, dataset=dataset)
    if not report.ok:
        _append_evaluation_contract_audit(audit_rows)
        return _discovery_only(
            output_dir=output_dir,
            system=system,
            generator=selected_generator,
            oracle=selected_oracle,
            audit_rows=audit_rows,
            counts=counts,
            gaps=(DiscoveryGap.EVALUATION_CONTRACT_MISMATCH,),
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


@dataclass(frozen=True, slots=True)
class _ManifestDescriptors:
    """Manifest values that each optimizer-eligible row must reproduce."""

    system_fingerprint: str
    generator_id: str
    oracle_id: str
    scoring_contract: str


def _integrity_paths(tuning_path: str | Path) -> tuple[Path, Path]:
    """Return the fixed dataset and manifest paths after symlink checks."""
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
    return dataset_path, manifest_path


def _manifest_matches_dataset(manifest: Mapping[str, Any], raw_dataset: bytes) -> bool:
    """Check the immutable envelope that binds a manifest to dataset bytes."""
    return (
        manifest.get("schema_version") == COLDSTART_SCHEMA_VERSION
        and manifest.get("outcome") == ColdStartOutcome.EVAL_SET.value
        and manifest.get("holdout_prohibited") is True
        and manifest.get("dataset_path") == _TUNING_FILENAME
        and manifest.get("dataset_sha256") == sha256_bytes(raw_dataset)
    )


def _read_integrity_artifacts(
    manifest_path: Path, dataset_path: Path
) -> tuple[bytes, Mapping[str, Any]]:
    """Read the local artifact pair and validate their shared manifest envelope."""
    try:
        raw_dataset = dataset_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColdStartConfigurationError(
            "Cold-start dataset and manifest must be readable local JSON artifacts."
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ColdStartConfigurationError("Cold-start manifest must be a JSON object.")
    if not _manifest_matches_dataset(manifest, raw_dataset):
        raise ColdStartConfigurationError("Cold-start manifest integrity check failed.")
    return raw_dataset, manifest


def _nonempty_manifest_descriptor(
    descriptor: Mapping[str, Any], field: str
) -> str | None:
    """Return one non-empty string descriptor without coercing untrusted values."""
    value = descriptor.get(field)
    return value if isinstance(value, str) and value else None


def _manifest_descriptors(
    manifest: Mapping[str, Any],
) -> tuple[_ManifestDescriptors, Mapping[str, Any]]:
    """Extract row-binding manifest descriptors or fail before loading the dataset."""
    system = manifest.get("system")
    generator = manifest.get("generator")
    oracle = manifest.get("oracle")
    counts = manifest.get("counts")
    if not all(
        isinstance(item, Mapping) for item in (system, generator, oracle, counts)
    ):
        raise ColdStartConfigurationError(
            "Cold-start manifest descriptors fail construction-evidence eligibility."
        )
    assert isinstance(system, Mapping)
    assert isinstance(generator, Mapping)
    assert isinstance(oracle, Mapping)
    assert isinstance(counts, Mapping)
    fingerprint = _nonempty_manifest_descriptor(system, "fingerprint")
    generator_id = _nonempty_manifest_descriptor(generator, "technique_id")
    oracle_id = _nonempty_manifest_descriptor(oracle, "oracle_id")
    scoring_contract = manifest.get("scoring_contract")
    if (
        fingerprint is None
        or generator_id is None
        or oracle_id is None
        or oracle.get("scoring_contract") != ScoringContract.EXACT_MATCH.value
        or scoring_contract != ScoringContract.EXACT_MATCH.value
    ):
        raise ColdStartConfigurationError(
            "Cold-start manifest descriptors fail construction-evidence eligibility."
        )
    return (
        _ManifestDescriptors(
            system_fingerprint=fingerprint,
            generator_id=generator_id,
            oracle_id=oracle_id,
            scoring_contract=scoring_contract,
        ),
        counts,
    )


def _raise_dataset_loader_error(exc: Exception) -> NoReturn:
    """Keep trusted-root diagnostics intact while normalizing other loader errors."""
    message = str(exc)
    if "Dataset path must reside under" in message:
        raise ColdStartConfigurationError(message) from exc
    raise ColdStartConfigurationError(
        "Cold-start tuning data is not Dataset.from_jsonl-compatible."
    ) from exc


def _load_tuning_rows(
    dataset_path: Path, raw_dataset: bytes
) -> list[Mapping[str, Any]]:
    """Use the normal Dataset loader and decode the exact persisted rows."""
    try:
        Dataset.from_jsonl(str(dataset_path))
        parsed_rows = [
            json.loads(line)
            for line in raw_dataset.decode("utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        _raise_dataset_loader_error(exc)
    if not parsed_rows or not all(isinstance(row, Mapping) for row in parsed_rows):
        raise ColdStartConfigurationError(
            "Cold-start tuning dataset is empty or malformed."
        )
    return [cast(Mapping[str, Any], row) for row in parsed_rows]


def _provenance_matches_manifest(
    provenance: Mapping[str, Any], descriptors: _ManifestDescriptors
) -> bool:
    """Compare every row provenance field to its manifest descriptor."""
    return (
        provenance.get("oracle_id"),
        provenance.get("generator_id"),
        provenance.get("system_fingerprint"),
        provenance.get("scoring_contract"),
    ) == (
        descriptors.oracle_id,
        descriptors.generator_id,
        descriptors.system_fingerprint,
        descriptors.scoring_contract,
    )


def _assert_row_eligible(
    *,
    row: Mapping[str, Any],
    descriptors: _ManifestDescriptors,
    seen_inputs: set[str],
    seen_example_ids: set[str],
) -> None:
    """Re-derive one row's evidence and cross-check its manifest provenance."""
    reason = validate_tuning_row(
        row,
        expected_schema_version=COLDSTART_SCHEMA_VERSION,
        seen_input_digests=seen_inputs,
        max_input_bytes=MAX_INPUT_BYTES,
    )
    if reason is not None:
        raise ColdStartConfigurationError(
            "Cold-start tuning row fails construction-evidence eligibility."
        )
    example_id = row.get("example_id")
    if not isinstance(example_id, str) or example_id in seen_example_ids:
        raise ColdStartConfigurationError(
            "Cold-start tuning row fails construction-evidence eligibility."
        )
    provenance = row.get("traigent_coldstart")
    assert isinstance(provenance, Mapping)
    if not _provenance_matches_manifest(provenance, descriptors):
        raise ColdStartConfigurationError(
            "Cold-start tuning row fails construction-evidence eligibility."
        )
    seen_example_ids.add(example_id)


def _assert_eligible_count(counts: Mapping[str, Any], row_count: int) -> None:
    """Bind the manifest's declared eligible count to the decoded row count."""
    eligible_count = counts.get(CandidateState.ELIGIBLE.value)
    if (
        not isinstance(eligible_count, int)
        or isinstance(eligible_count, bool)
        or eligible_count != row_count
    ):
        raise ColdStartConfigurationError(
            "Cold-start manifest eligible count does not match the tuning dataset."
        )


def assert_optimizer_eligible(tuning_path: str | Path) -> None:
    """Check local cold-start artifact integrity before passing it to optimize.

    This verifies accidental alteration and construction-policy eligibility.  It
    is not authentication and does not prove the semantic correctness of an
    oracle or prevent a party who can edit both dataset and manifest from
    recomputing hashes.
    """
    dataset_path, manifest_path = _integrity_paths(tuning_path)
    raw_dataset, manifest = _read_integrity_artifacts(manifest_path, dataset_path)
    descriptors, manifest_counts = _manifest_descriptors(manifest)
    parsed_rows = _load_tuning_rows(dataset_path, raw_dataset)
    seen_inputs: set[str] = set()
    seen_example_ids: set[str] = set()
    for row in parsed_rows:
        _assert_row_eligible(
            row=row,
            descriptors=descriptors,
            seen_inputs=seen_inputs,
            seen_example_ids=seen_example_ids,
        )
    _assert_eligible_count(manifest_counts, len(parsed_rows))


__all__ = ["assert_optimizer_eligible", "generate_eval_set"]
