"""Offline, keyless certificate verification commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from traigent_schema.certification import (
        AgentQualityVerificationContext,
        DatasetRecordVerificationContext,
        EvaluatorQualityVerificationContext,
        ProcessRecordVerificationContext,
        TrustAnchorKeyV1,
    )

#: Stable refusal code for `--kind evaluator-quality` without
#: `--development-integration` (see `_evaluator_quality_eligibility_policy`
#: below for the disposition this pins).
EVALUATOR_QUALITY_DEV_INTEGRATION_ONLY = "EVALUATOR_QUALITY_DEV_INTEGRATION_ONLY"

# --- Evaluator-quality eligibility policy -----------------------------------
#
# This SDK release never reports an evaluator-quality certificate as eligible
# for unrestricted acceptance (`certification_eligible` is always `false`),
# regardless of the bundle's own technical verdict. Two independent reasons,
# both open:
#
# 1. `EvaluatorQualityVerificationContext.expected_evaluator_commitment_ref`
#    must, per that dataclass's own docstring, come from a relying party's
#    OWN prior call to `verify_process_record_certificate` -- a cross-bundle
#    composition this CLI command does not perform (`verify` takes exactly
#    one bundle per invocation for evaluator-quality; see
#    `--process-record`'s dataset-record-only scope below). Nothing here
#    proves the ref the caller supplied was ever independently verified, so
#    the CLI must not synthesize that proof and must not report eligibility
#    that rests on it.
# 2. `EVQ6` (efficiency) and `EVQ7` (agreement-per-cost frontier) are
#    registered claim ids the v1 Schema deliberately makes unrepresentable
#    (`claim_material` is a fixed 5-tuple of EVQ1..EVQ5 by JSON-Schema
#    position) -- a bundle can never claim them; a bundle that tries is
#    refused by the real verifier's own schema validation, reported as
#    `EVALUATOR_SCHEMA` (there is no separate "unsupported" disposition).
#
# Until a relying-party composition step is added (verifying the referenced
# process record in the same invocation and rejecting a mismatched
# commitment ref), `--kind evaluator-quality` is development-integration
# only: pass `--development-integration` to run technical verification and
# see its verdicts, but never treat the result as a certification decision.
#
# This disposition is PINNED to an exact (Schema pin, verifier revision)
# pair, recorded as data (not just prose) in
# `EVALUATOR_QUALITY_POLICY_DISPOSITION` below.
# `tests/unit/cli/test_certify_evaluator_quality.py::
# test_eligibility_policy_pin_matches_installed_schema` fails closed if
# either the Schema pin or the hashed verifier artifacts drift from that
# record, so a Schema pin bump or a `traigent_schema` release that changes
# `evaluator_quality_verifier.py`, the evaluator-quality JSON schema, or any
# of the three evaluator registry digest files forces a human to revisit
# this comment and `EVALUATOR_QUALITY_POLICY_DISPOSITION` in the same PR --
# that dict IS the reviewed disposition record.
# Re-recorded 2026-10-20 at Schema 31e0ddf: verifier_revision changed only because
# Schema aacd5db (#563) moved `return Draft7Validator(...)` out of the try block in
# the verifier module (behaviour-preserving). Owner card
# feature-pick-2026-10-20-sdk-verifier-revision-rerecord, answered A.
# Pin moved to d5d0cb6 (commitment-linkage reader, C1 step 1) with verifier_revision
# UNCHANGED for both evaluator-quality and agent-quality (recomputed on both installs).
# Pin moved to 7104839 (= PyPI traigent-schema 7.0.0, tag v7.0.0): evaluator-quality
# verifier_revision UNCHANGED (recomputed); agent-quality verifier_revision changed
# because the agent-quality non-claim catalog wording changed ("measured" -> "reported
# by the customer", same catalog version 1.0.0, new digest).
EVALUATOR_QUALITY_POLICY_DISPOSITION = {
    "schema_pin": "7104839ae6d748e852b1cb584312cf86d1f0f2db",
    "verifier_revision": "887114f799459df27269463cb73e1bc1ff9c4a948d73dddb83db4b88cc3e831f",
    "reviewed_by": "maintainers",
    "date": "2026-10-20",
}


def _evaluator_quality_artifact_paths() -> list[Path]:
    """The installed evaluator-quality verification artifacts hashed by
    `evaluator_quality_verifier_revision`."""
    import traigent_schema
    from traigent_schema.certification import evaluator_quality_verifier as module

    root = Path(traigent_schema.__file__).parent
    return [
        Path(module.__file__),
        root / "schemas" / "certification" / "evaluator_quality_v1_schema.json",
        root / "data" / "certification" / "evaluator_assertion_templates.digest.json",
        root / "data" / "certification" / "evaluator_measurement_registry.digest.json",
        root / "data" / "certification" / "evaluator_perturbation_set.digest.json",
    ]


def evaluator_quality_verifier_revision(files: list[Path] | None = None) -> str:
    """SHA-256 over the installed evaluator-quality verification artifacts.

    Covers `evaluator_quality_verifier.py`, the evaluator-quality JSON
    schema, and the three evaluator registry digest files -- not just the
    verifier module -- so a Schema release that changes emittable-claim
    semantics or registry bindings without touching the verifier's own
    source still moves this hash. Used only as a guard-test comparison key
    against `EVALUATOR_QUALITY_POLICY_DISPOSITION["verifier_revision"]` --
    never at runtime by the CLI itself, so an absent `traigent_schema`
    install never breaks anything other than that one test.

    `files` overrides the artifact list -- tests use this to hash tmp
    copies of the installed artifacts instead of mutating the install.
    """
    if files is None:
        files = _evaluator_quality_artifact_paths()
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda p: p.name):
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


# --- Agent-quality (pillar 1) eligibility policy ----------------------------
#
# Unlike evaluator-quality (pillar 3), `--kind agent-quality` composes the
# process-record verification INSIDE the real verifier
# (`verify_agent_quality_certificate` verifies `--process-record` in full
# before checking anything else), so there is no cross-bundle composition gap
# here -- see the module-level comment above `verify` for the full contrast.
# `certification_eligible` is conditional only on the one thing the result
# type itself cannot express: whether the embedded process record's base
# status was independently checked against a trust anchor
# (`process_record_base_status=checked`) or merely accepted unchecked
# (`=not_checked`). A VERIFIED result over an unchecked process record is
# still printed, but never eligible.
#
# Same pinned-disposition discipline as evaluator-quality:
# `tests/unit/cli/test_certify_agent_quality.py::
# test_eligibility_policy_pin_matches_installed_schema` fails closed if
# either the Schema pin or the hashed verifier artifacts drift from
# `AGENT_QUALITY_POLICY_DISPOSITION` below -- a Schema pin bump or a
# `traigent_schema` release that changes `agent_quality_verifier.py`, the
# agent-quality JSON schema, or any of the four registry documents (and their
# digest sidecars) forces a human to revisit this comment and that dict in
# the same PR.
# Re-recorded 2026-10-20 at Schema 31e0ddf: verifier_revision changed only because
# Schema aacd5db (#563) moved `return Draft7Validator(...)` out of the try block in
# the verifier module (behaviour-preserving). Owner card
# feature-pick-2026-10-20-sdk-verifier-revision-rerecord, answered A.
AGENT_QUALITY_POLICY_DISPOSITION = {
    "schema_pin": "7104839ae6d748e852b1cb584312cf86d1f0f2db",
    "verifier_revision": "37d4c5e891283f85a8b17be385b81ae1eef0c980092fb0a2b2555b0627be24b1",
    "reviewed_by": "maintainers",
    "date": "2026-10-23",
}


def _agent_quality_artifact_paths() -> list[Path]:
    """The installed agent-quality verification artifacts hashed by
    `agent_quality_verifier_revision`: the verifier module, the
    agent-quality JSON schema, the three schemas it `$ref`s (`certification_
    common_v0`, `certificate_audit_report_v0`, `certificate_evidence_refs_v0`
    -- the verifier resolves these at validation time, so a change to any of
    them changes verified behavior without touching agent_quality_v1_schema.
    json's own bytes), and the four registry documents
    (`_load_agent_quality_document`'s "objective_registry",
    "aggregation_policy", "non_claim_catalog", "quantile_table" stems) each
    with its pinned `.digest.json` sidecar."""
    import traigent_schema
    from traigent_schema.certification import agent_quality_verifier as module

    root = Path(traigent_schema.__file__).parent
    schema_dir = root / "schemas" / "certification"
    data_dir = root / "data" / "certification"
    stems = (
        "objective_registry",
        "aggregation_policy",
        "non_claim_catalog",
        "quantile_table",
    )
    paths = [
        Path(module.__file__),
        schema_dir / "agent_quality_v1_schema.json",
        schema_dir / "certification_common_v0_schema.json",
        schema_dir / "certificate_audit_report_v0_schema.json",
        schema_dir / "certificate_evidence_refs_v0_schema.json",
    ]
    for stem in stems:
        paths.append(data_dir / f"agent_quality_{stem}.json")
        paths.append(data_dir / f"agent_quality_{stem}.digest.json")
    return paths


def agent_quality_verifier_revision(files: list[Path] | None = None) -> str:
    """SHA-256 over the installed agent-quality verification artifacts.

    Mirrors `evaluator_quality_verifier_revision` exactly: used only as a
    guard-test comparison key against
    `AGENT_QUALITY_POLICY_DISPOSITION["verifier_revision"]`, never at runtime
    by the CLI itself. `files` overrides the artifact list -- tests use this
    to hash tmp copies of the installed artifacts instead of mutating the
    install.
    """
    if files is None:
        files = _agent_quality_artifact_paths()
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda p: p.name):
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _load_certification_schema() -> SimpleNamespace:
    """Lazily import traigent_schema.certification, called only inside `verify`.

    traigent-schema is a declared core dependency (exact PyPI pin), but it is
    still imported lazily so `traigent --help` and the other subcommands do not
    pay the import cost or break in an environment where it was removed.
    """
    try:
        from traigent_schema.certification import (
            AGENT_QUALITY_CLAIM_ABSTAINED,
            AGENT_QUALITY_VERIFIED,
            AgentQualityVerificationContext,
            DatasetRecordVerificationContext,
            EvaluatorQualityVerificationContext,
            ProcessRecordVerificationContext,
            TrustAnchorKeyV1,
            verify_agent_quality_certificate,
            verify_dataset_record_certificate,
            verify_evaluator_quality_certificate,
            verify_process_record_certificate,
        )
        from traigent_schema.certification.agent_quality_verifier import (
            AgentQualityVerificationError,
        )
        from traigent_schema.certification.dataset_record_verifier import (
            DatasetRecordVerificationError,
        )
        from traigent_schema.certification.evaluator_quality_verifier import (
            EvaluatorQualityVerificationError,
        )
        from traigent_schema.certification.process_record_verifier import (
            ProcessRecordVerificationError,
        )
        from traigent_schema.certification.relying_party_verifier import (
            VerificationContext,
        )
    except ImportError as exc:
        raise click.UsageError(
            "traigent-schema is not installed; the offline verifier needs the "
            "pinned TraigentSchema package (see scripts/ci/schema-pin.txt)"
        ) from exc
    return SimpleNamespace(
        AGENT_QUALITY_CLAIM_ABSTAINED=AGENT_QUALITY_CLAIM_ABSTAINED,
        AGENT_QUALITY_VERIFIED=AGENT_QUALITY_VERIFIED,
        AgentQualityVerificationContext=AgentQualityVerificationContext,
        DatasetRecordVerificationContext=DatasetRecordVerificationContext,
        EvaluatorQualityVerificationContext=EvaluatorQualityVerificationContext,
        ProcessRecordVerificationContext=ProcessRecordVerificationContext,
        TrustAnchorKeyV1=TrustAnchorKeyV1,
        verify_agent_quality_certificate=verify_agent_quality_certificate,
        verify_dataset_record_certificate=verify_dataset_record_certificate,
        verify_evaluator_quality_certificate=verify_evaluator_quality_certificate,
        verify_process_record_certificate=verify_process_record_certificate,
        AgentQualityVerificationError=AgentQualityVerificationError,
        DatasetRecordVerificationError=DatasetRecordVerificationError,
        EvaluatorQualityVerificationError=EvaluatorQualityVerificationError,
        ProcessRecordVerificationError=ProcessRecordVerificationError,
        VerificationContext=VerificationContext,
    )


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        failed = True
    else:
        failed = False
    # Raised OUTSIDE the except block: JSONDecodeError.doc holds the file text
    # (which may be a commitments bundle), so it must not stay as __context__.
    if failed:
        raise click.UsageError(f"could not read {description}")
    if not isinstance(value, dict):
        raise click.UsageError(f"{description} must contain a JSON object")
    return value


def _required(payload: dict[str, Any], name: str) -> Any:
    if name not in payload:
        raise click.UsageError(f"context is missing required field: {name}")
    return payload[name]


def _anchor(payload: dict[str, Any], schema: SimpleNamespace) -> TrustAnchorKeyV1:
    try:
        return schema.TrustAnchorKeyV1(
            key_ref=_required(payload, "key_ref"),
            algorithm=_required(payload, "algorithm"),
            public_key_der_b64=_required(payload, "public_key_der_b64"),
            public_key_digest=_required(payload, "public_key_digest"),
        )
    except (TypeError, ValueError) as exc:
        raise click.UsageError("anchor is invalid") from exc


_CONTEXT_IS_INVALID = "context is invalid"


def _process_context(
    payload: dict[str, Any], anchor: TrustAnchorKeyV1, schema: SimpleNamespace
) -> ProcessRecordVerificationContext:
    base = _required(payload, "base_context")
    if not isinstance(base, dict):
        raise click.UsageError("context.base_context must be a JSON object")
    base_context = schema.VerificationContext(
        expected_nonce=_required(base, "expected_nonce"),
        expected_build_session_ref=_required(base, "expected_build_session_ref"),
        expected_issuer_key_ref=_required(base, "expected_issuer_key_ref"),
        expected_issuer_algorithm=_required(base, "expected_issuer_algorithm"),
        expected_trust_ring_ref=_required(base, "expected_trust_ring_ref"),
        expected_project_ref=_required(base, "expected_project_ref"),
        expected_client_key_ref=_required(base, "expected_client_key_ref"),
        expected_client_algorithm=_required(base, "expected_client_algorithm"),
        client_public_key=_required(base, "client_public_key"),
    )
    allow_unchecked = _required(payload, "allow_unchecked_base_status")
    try:
        return schema.ProcessRecordVerificationContext(
            expected_materials_digest=_required(payload, "expected_materials_digest"),
            certificate_ref=_required(payload, "certificate_ref"),
            base_context=base_context,
            expected_project_ref=_required(payload, "expected_project_ref"),
            expected_build_session_ref=_required(payload, "expected_build_session_ref"),
            expected_agent_commitment_ref=_required(
                payload, "expected_agent_commitment_ref"
            ),
            expected_dataset_commitment_ref=_required(
                payload, "expected_dataset_commitment_ref"
            ),
            expected_evaluator_commitment_ref=_required(
                payload, "expected_evaluator_commitment_ref"
            ),
            expected_build_definition_commitment_ref=_required(
                payload, "expected_build_definition_commitment_ref"
            ),
            allow_unchecked_base_status=allow_unchecked,
            verification_time=_required(payload, "verification_time"),
            trust_anchor=None if allow_unchecked else anchor,
        )
    except (TypeError, ValueError) as exc:
        raise click.UsageError(_CONTEXT_IS_INVALID) from exc


def _dataset_context(
    payload: dict[str, Any], anchor: TrustAnchorKeyV1, schema: SimpleNamespace
) -> DatasetRecordVerificationContext:
    process_payload = _required(payload, "process_context")
    if not isinstance(process_payload, dict):
        raise click.UsageError("context.process_context must be a JSON object")
    process_context = _process_context(process_payload, anchor, schema)
    try:
        return schema.DatasetRecordVerificationContext(
            process_context=process_context,
            expected_project_ref=_required(payload, "expected_project_ref"),
            expected_build_session_ref=_required(payload, "expected_build_session_ref"),
            expected_dataset_commitment_ref=_required(
                payload, "expected_dataset_commitment_ref"
            ),
            expected_dataset_identity_root=payload.get(
                "expected_dataset_identity_root"
            ),
            expected_identity_blind_epoch=_required(
                payload, "expected_identity_blind_epoch"
            ),
            expected_leakage_scope_ref=_required(payload, "expected_leakage_scope_ref"),
            expected_prereg_digest=payload.get("expected_prereg_digest"),
            require_leaf_lists=_required(payload, "require_leaf_lists"),
        )
    except (TypeError, ValueError) as exc:
        raise click.UsageError(_CONTEXT_IS_INVALID) from exc


def _evaluator_quality_context(
    payload: dict[str, Any], anchor: TrustAnchorKeyV1, schema: SimpleNamespace
) -> EvaluatorQualityVerificationContext:
    allow_unchecked = _required(payload, "allow_unchecked_trust_status")
    try:
        return schema.EvaluatorQualityVerificationContext(
            expected_project_ref=_required(payload, "expected_project_ref"),
            expected_evaluator_commitment_ref=_required(
                payload, "expected_evaluator_commitment_ref"
            ),
            allow_unchecked_trust_status=allow_unchecked,
            verification_time=_required(payload, "verification_time"),
            trust_anchor=None if allow_unchecked else anchor,
        )
    except (TypeError, ValueError) as exc:
        raise click.UsageError(_CONTEXT_IS_INVALID) from exc


def _agent_quality_context(
    payload: dict[str, Any],
    anchor: TrustAnchorKeyV1,
    schema: SimpleNamespace,
    trust_status_payload: dict[str, Any] | None,
) -> AgentQualityVerificationContext:
    process_record_payload = _required(payload, "process_record_context")
    if not isinstance(process_record_payload, dict):
        raise click.UsageError("context.process_record_context must be a JSON object")
    _check_context_anchor(process_record_payload, anchor)
    process_record_context = _process_context(process_record_payload, anchor, schema)
    try:
        return schema.AgentQualityVerificationContext(
            process_record_context=process_record_context,
            expected_project_ref=_required(payload, "expected_project_ref"),
            expected_build_session_ref=_required(payload, "expected_build_session_ref"),
            expected_agent_commitment_ref=_required(
                payload, "expected_agent_commitment_ref"
            ),
            expected_dataset_commitment_ref=_required(
                payload, "expected_dataset_commitment_ref"
            ),
            expected_evaluator_commitment_ref=_required(
                payload, "expected_evaluator_commitment_ref"
            ),
            expected_build_definition_commitment_ref=_required(
                payload, "expected_build_definition_commitment_ref"
            ),
            expected_measurement_contract_ref=_required(
                payload, "expected_measurement_contract_ref"
            ),
            expected_measurement_contract_record_digest=_required(
                payload, "expected_measurement_contract_record_digest"
            ),
            accept_abstained_bundle=_required(payload, "accept_abstained_bundle"),
            expected_declared_plan_digest=payload.get("expected_declared_plan_digest"),
            trust_status=trust_status_payload,
        )
    except schema.AgentQualityVerificationError:
        raise
    except (TypeError, ValueError) as exc:
        raise click.UsageError(_CONTEXT_IS_INVALID) from exc


def _resolve_kind(kind: str | None, context_payload: dict[str, Any]) -> str:
    """Pick the verification family, refusing to guess evaluator-quality or
    agent-quality.

    `--kind evaluator-quality` is never auto-detected. Its context marker is
    `allow_unchecked_trust_status` (the S10 opt-in unique to
    `EvaluatorQualityVerificationContext`) -- NOT
    `expected_evaluator_commitment_ref`, which a legitimate process-record
    context also carries (see `_process_context`), so that field cannot be
    used to distinguish the families. A context carrying the
    evaluator-quality marker without an explicit `--kind` is refused
    outright rather than routed into the wrong verifier.

    `--kind agent-quality` is likewise never auto-detected. Its markers are
    `accept_abstained_bundle` and `process_record_context` -- both unique to
    `AgentQualityVerificationContext` (a legitimate dataset-record context's
    nested key is `process_context`, singular and without "_record", and a
    process-record context has neither key at its own top level).
    """
    if kind is not None:
        return kind
    if "allow_unchecked_trust_status" in context_payload:
        raise click.UsageError(
            "context contains allow_unchecked_trust_status (an "
            "evaluator-quality marker); pass --kind evaluator-quality "
            "explicitly -- it is never auto-detected"
        )
    if (
        "accept_abstained_bundle" in context_payload
        or "process_record_context" in context_payload
    ):
        raise click.UsageError(
            "context contains accept_abstained_bundle or process_record_context "
            "(an agent-quality marker); pass --kind agent-quality explicitly "
            "-- it is never auto-detected"
        )
    if "process_context" in context_payload:
        return "dataset-record"
    return "process-record"


def _check_context_anchor(payload: dict[str, Any], anchor: TrustAnchorKeyV1) -> None:
    embedded = payload.get("trust_anchor")
    if embedded is not None and embedded != {
        "key_ref": anchor.key_ref,
        "algorithm": anchor.algorithm,
        "public_key_der_b64": anchor.public_key_der_b64,
        "public_key_digest": anchor.public_key_digest,
    }:
        raise click.UsageError("context trust_anchor does not match --anchor")


@click.group()
def certify() -> None:
    """Verify public certificate artifacts offline."""


@certify.command()
@click.argument(
    "bundle",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--context",
    "context_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--anchor",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--trust-status",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--process-record",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--kind",
    type=click.Choice(
        ["dataset-record", "process-record", "evaluator-quality", "agent-quality"]
    ),
    default=None,
    help=(
        "Verification family. Required for evaluator-quality and "
        "agent-quality (never guessed)."
    ),
)
@click.option(
    "--development-integration",
    is_flag=True,
    default=False,
    help=(
        "Required to run --kind evaluator-quality at all. Technical "
        "verification runs and prints its verdicts, but the result is "
        "never reported certification_eligible=true (see "
        "EVALUATOR_QUALITY_POLICY_DISPOSITION's comment). Exit codes for "
        "--kind evaluator-quality: 1 = refused or verification failed, "
        "2 = usage error, 3 = technically valid, not certification-eligible. "
        "--kind agent-quality does not use this flag; its exit codes are "
        "0 = verified and eligible, 1 = refused or verification failed, "
        "2 = usage error, 3 = technically valid, not eligible."
    ),
)
@click.option(
    "--open-commitments",
    "open_commitments",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=None,
    help=(
        "Client bundle written by `certify finalize --link-commitments`. After "
        "the process record verifies, open the dataset and config_space "
        "commitments in its sidecar and print COMMITMENT_OPENED_MATCH, "
        "_MISMATCH or COMMITMENT_UNAVAILABLE per role (MISMATCH exits 1). "
        "Only valid with --kind process-record. POSIX-only for now: on other "
        "platforms it refuses with LINKAGE_UNSUPPORTED_PLATFORM."
    ),
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit machine-readable JSON instead of human-readable lines.",
)
def verify(
    bundle: Path,
    context_path: Path,
    anchor: Path,
    trust_status: Path | None,
    process_record: Path | None,
    kind: str | None,
    development_integration: bool,
    open_commitments: Path | None,
    as_json: bool,
) -> None:
    """Verify BUNDLE using independently supplied context and public anchor."""
    if open_commitments is not None:
        from traigent.cloud import process_record_linkage as _prl

        try:
            _prl.require_posix_platform()
        except _prl.CommitmentLinkageClientError as err:
            click.echo(err.code)
            raise click.exceptions.Exit(1) from None
    schema = _load_certification_schema()
    bundle_payload = _read_json(bundle, "bundle")
    context_payload = _read_json(context_path, "context")
    anchor_payload = _read_json(anchor, "anchor")
    anchor_value = _anchor(anchor_payload, schema)
    _check_context_anchor(context_payload, anchor_value)
    trust_status_payload = (
        _read_json(trust_status, "trust status") if trust_status is not None else None
    )

    resolved_kind = _resolve_kind(kind, context_payload)
    if open_commitments is not None and resolved_kind != "process-record":
        raise click.UsageError("--open-commitments requires --kind process-record")

    if resolved_kind == "evaluator-quality":
        if process_record is not None:
            raise click.UsageError(
                "--kind evaluator-quality does not accept --process-record "
                "(this CLI performs no relying-party composition step; see "
                "EVALUATOR_QUALITY_POLICY_DISPOSITION's comment)"
            )
        _verify_evaluator_quality(
            schema,
            bundle_payload,
            context_payload,
            anchor_value,
            trust_status_payload,
            development_integration=development_integration,
            as_json=as_json,
        )
        return

    if resolved_kind == "agent-quality":
        if process_record is None:
            raise click.UsageError("--kind agent-quality requires --process-record")
        if development_integration:
            raise click.UsageError(
                "--kind agent-quality does not use --development-integration"
            )
        _verify_agent_quality(
            schema,
            bundle_payload,
            context_payload,
            _read_json(process_record, "process record"),
            anchor_value,
            trust_status_payload,
            as_json=as_json,
        )
        return

    if resolved_kind == "dataset-record" and "process_context" not in context_payload:
        raise click.UsageError("--kind dataset-record requires context.process_context")
    if resolved_kind == "process-record" and "process_context" in context_payload:
        raise click.UsageError(
            "--kind process-record does not accept context.process_context "
            "(that shape is dataset-record)"
        )

    try:
        if resolved_kind == "dataset-record":
            if process_record is None:
                raise click.UsageError(
                    "dataset-record verification requires --process-record"
                )
            result = schema.verify_dataset_record_certificate(
                bundle_payload,
                context=_dataset_context(context_payload, anchor_value, schema),
                process_record_bundle=_read_json(process_record, "process record"),
                leaf_lists=context_payload.get("leaf_lists"),
            )
        else:
            if (
                context_payload.get("allow_unchecked_base_status") is True
                and trust_status_payload is not None
            ):
                raise click.UsageError(
                    "--trust-status cannot be used with "
                    "allow_unchecked_base_status=true"
                )
            result = schema.verify_process_record_certificate(
                bundle_payload,
                context=_process_context(context_payload, anchor_value, schema),
                trust_status=trust_status_payload,
            )
    except click.UsageError:
        raise
    except (
        schema.ProcessRecordVerificationError,
        schema.DatasetRecordVerificationError,
    ) as exc:
        click.echo(exc.code)
        raise click.exceptions.Exit(1) from None

    if not result.valid:
        click.echo(result.code)
        raise click.exceptions.Exit(1)
    # "VERIFIED" alone could be read as "quality verified": it covers signature and structure
    # only, and any quality numbers in the certificate are reported by the customer.
    click.echo(
        "signature_and_structure=VERIFIED "
        "(quality numbers are customer-reported; see the certificate non-claims)"
    )
    if hasattr(result, "trust_status_evidence"):
        click.echo(f"trust_status_evidence={result.trust_status_evidence}")
        click.echo(f"trust_status_effective_time={result.trust_status_effective_time}")
    if open_commitments is not None:
        _open_commitments(bundle_payload, open_commitments)


def _open_commitments(record_bundle: dict[str, Any], commitments_path: Path) -> None:
    """Open the dataset/config_space commitments of an already-verified record.

    The sidecar is verified against the record's own digest, project and
    session refs and the ISSUER KEY COPIED FROM THAT VERIFIED RECORD
    (``verification_materials_v0.issuer``), never from the sidecar. Openings
    and fingerprint digests come from the client bundle; they are never
    printed. A role with no opening or no local fp1 digest is reported
    COMMITMENT_UNAVAILABLE rather than guessed.
    """
    from traigent.cloud.process_record_linkage import (
        BUNDLE_SCHEMA,
        DISCLOSURE_LINE,
        OPENING_ROLES,
    )

    try:
        from traigent_schema.certification import commitment_linkage as cl
    except ImportError as exc:
        raise click.UsageError(
            "the pinned traigent-schema does not provide commitment linkage "
            "(need the pin in scripts/ci/schema-pin.txt)"
        ) from exc

    from traigent.cloud.process_record_linkage import (
        CommitmentLinkageClientError,
        read_private_json,
    )

    refused: str | None = None
    payload: dict[str, Any] = {}
    try:
        payload = read_private_json(commitments_path)
    except CommitmentLinkageClientError as err:
        refused = err.code
    if refused is not None:
        click.echo(f"COMMITMENT_BUNDLE_REFUSED {refused}")
        raise click.exceptions.Exit(1)
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise click.UsageError("commitments bundle has an unsupported schema")
    linkage = payload.get("commitment_linkage")
    if not isinstance(linkage, dict):
        raise click.UsageError("commitments bundle has no commitment_linkage")
    raw_openings = payload.get("openings")
    raw_openings = raw_openings if isinstance(raw_openings, dict) else {}
    fp_digests = payload.get("fp_digests")
    fp_digests = fp_digests if isinstance(fp_digests, dict) else {}

    try:
        manifest = record_bundle["unsigned_manifest"]
        issuer = record_bundle["verification_materials_v0"]["issuer"]
        context = cl.CommitmentLinkageVerificationContext(
            expected_process_record_digest=record_bundle["signature"][
                "unsigned_manifest_digest"
            ],
            expected_project_ref=manifest["project_ref"],
            expected_build_session_ref=manifest["build_session_ref"],
            issuer_key=cl.LinkageIssuerKeyV1(
                key_ref=issuer["key_ref"],
                trust_ring_ref=issuer["trust_ring_ref"],
                algorithm=issuer["algorithm"],
                public_key_der_b64=issuer["public_key_der_b64"],
                public_key_digest=issuer["public_key_digest"],
            ),
        )
    except (KeyError, TypeError, cl.CommitmentLinkageError):
        raise click.UsageError(
            "record bundle lacks the fields needed to open commitments"
        ) from None

    import base64

    # Authenticate the sidecar against the verified record FIRST, so a forged
    # or foreign sidecar is rejected even when no local fp/opening is present.
    invalid: str | None = None
    try:
        cl.verify_commitment_linkage(linkage, context=context)
    except cl.CommitmentLinkageError as err:
        invalid = err.code
    if invalid is not None:
        click.echo(f"COMMITMENT_LINKAGE_INVALID {invalid}")
        raise click.exceptions.Exit(1)

    results: list[tuple[str, str]] = []
    mismatch = False
    for role in OPENING_ROLES:
        fp = fp_digests.get(role)
        opening_b64 = raw_openings.get(role)
        if not isinstance(fp, str) or not isinstance(opening_b64, str):
            results.append((role, cl.COMMITMENT_UNAVAILABLE))
            continue
        outcome = ""
        invalid = None
        try:
            opening = base64.b64decode(opening_b64, validate=True)
            outcome = cl.verify_and_open_commitment(
                linkage, context=context, role=role, fp_digest=fp, opening=opening
            )
        except cl.CommitmentLinkageError as err:
            invalid = err.code
        except ValueError:
            invalid = "OPENING_SHAPE"
        if invalid is not None:
            click.echo(f"COMMITMENT_LINKAGE_INVALID {invalid}")
            raise click.exceptions.Exit(1)
        mismatch = mismatch or outcome == cl.COMMITMENT_OPENED_MISMATCH
        results.append((role, outcome))
    for role, outcome in results:
        click.echo(f"{role}: {outcome}")
    click.echo(DISCLOSURE_LINE)
    if mismatch:
        raise click.exceptions.Exit(1)


def _make_backend_client() -> Any:
    from traigent.cloud.backend_client import BackendIntegratedClient

    return BackendIntegratedClient()


@certify.command()
@click.argument("build_session_id")
@click.option(
    "--request",
    "request_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="JSON finalize payload (refs, base certificate, verification materials).",
)
@click.option(
    "--bundle",
    "bundle_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Client bundle file (mode 0600) that receives the record and sidecar.",
)
@click.option(
    "--link-commitments",
    is_flag=True,
    default=False,
    help=(
        "Opt in: also request the commitment-linkage sidecar for the dataset "
        "and config_space declarations. Fresh openings are written to --bundle "
        "BEFORE the request is sent and reused on retry. Default off. The command "
        "as a whole is POSIX-only for now: on other platforms it refuses with "
        "LINKAGE_UNSUPPORTED_PLATFORM."
    ),
)
@click.option(
    "--fp-digest",
    "fp_digest",
    multiple=True,
    metavar="ROLE=sha256:HEX",
    help="Locally known fp1 digest for dataset or config_space (stored in the bundle).",
)
def finalize(
    build_session_id: str,
    request_path: Path,
    bundle_path: Path,
    link_commitments: bool,
    fp_digest: tuple[str, ...],
) -> None:
    """Finalize BUILD_SESSION_ID's process record on the backend (opt-in client).

    POSIX-only for now: every invocation writes a client bundle, so on other
    platforms it refuses with LINKAGE_UNSUPPORTED_PLATFORM before doing any work.
    """
    # finalize always writes a client bundle (--bundle is mandatory), so the
    # POSIX guard applies to every invocation, not only --link-commitments.
    from traigent.cloud.process_record_linkage import (
        CommitmentLinkageClientError,
        finalize_process_record,
        require_posix_platform,
    )

    try:
        require_posix_platform()
    except CommitmentLinkageClientError as err:
        click.echo(f"FINALIZE_REFUSED {err.code}", err=True)
        raise click.exceptions.Exit(1) from None
    digests: dict[str, str] = {}
    for item in fp_digest:
        role, sep, value = item.partition("=")
        if not sep:
            raise click.UsageError("--fp-digest must look like ROLE=sha256:HEX")
        digests[role] = value
    payload = _read_json(request_path, "request")
    try:
        outcome = finalize_process_record(
            _make_backend_client(),
            build_session_id,
            payload,
            link_commitments=link_commitments,
            bundle_path=bundle_path,
            fp_digests=digests or None,
        )
    except CommitmentLinkageClientError as err:
        click.echo(f"FINALIZE_REFUSED {err.code}", err=True)
        if err.code == "SCHEMA_PACKAGE_TOO_OLD":
            click.echo(str(err), err=True)
        raise click.exceptions.Exit(1) from None
    click.echo("FINALIZED")
    click.echo(
        f"commitment_linkage={'present' if outcome['commitment_linkage'] else 'absent'}"
    )


def _verify_evaluator_quality(
    schema: SimpleNamespace,
    bundle_payload: dict[str, Any],
    context_payload: dict[str, Any],
    anchor_value: TrustAnchorKeyV1,
    trust_status_payload: dict[str, Any] | None,
    *,
    development_integration: bool,
    as_json: bool,
) -> None:
    """Run (or refuse) evaluator-quality verification under the fixed policy.

    Never reports `certification_eligible: true` -- see
    `EVALUATOR_QUALITY_POLICY_DISPOSITION`'s comment for why. Exit codes are
    part of the contract: 1 for outright refusal or a technical verification
    failure, 3 for a technically-valid bundle that is still not eligible
    (the only reachable outcome today), 2 (Click's default) for a usage
    error raised before either of those.
    """
    if not development_integration:
        _emit_evaluator_quality(
            as_json,
            {
                "code": EVALUATOR_QUALITY_DEV_INTEGRATION_ONLY,
                "certification_eligible": False,
            },
            [EVALUATOR_QUALITY_DEV_INTEGRATION_ONLY, "certification_eligible=false"],
        )
        raise click.exceptions.Exit(1)

    context = _evaluator_quality_context(context_payload, anchor_value, schema)
    try:
        result = schema.verify_evaluator_quality_certificate(
            bundle_payload,
            context=context,
            trust_status=trust_status_payload,
        )
    except schema.EvaluatorQualityVerificationError as exc:
        _emit_evaluator_quality(
            as_json,
            {"code": exc.code, "certification_eligible": False},
            [exc.code, "certification_eligible=false"],
        )
        raise click.exceptions.Exit(1) from None

    payload = {
        "valid": result.valid,
        "code": result.code,
        "instrument_adequacy": result.instrument_adequacy,
        "overall_verdict": result.overall_verdict,
        "overall_quality_ppm": result.overall_quality_ppm,
        "trust_status_evidence": result.trust_status_evidence,
        "certification_eligible": False,
    }
    lines = [
        result.code,
        f"instrument_adequacy={result.instrument_adequacy}",
        f"overall_verdict={result.overall_verdict}",
        f"trust_status_evidence={result.trust_status_evidence}",
        "certification_eligible=false",
    ]
    _emit_evaluator_quality(as_json, payload, lines)
    raise click.exceptions.Exit(3)


def _emit_evaluator_quality(
    as_json: bool, payload: dict[str, Any], lines: list[str]
) -> None:
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True))
    else:
        for line in lines:
            click.echo(line)


def _verify_agent_quality(
    schema: SimpleNamespace,
    bundle_payload: dict[str, Any],
    context_payload: dict[str, Any],
    process_record_payload: dict[str, Any],
    anchor_value: TrustAnchorKeyV1,
    trust_status_payload: dict[str, Any] | None,
    *,
    as_json: bool,
) -> None:
    """Run agent-quality (pillar 1) verification.

    Unlike evaluator-quality, this composes the process-record verification
    INSIDE the real verifier (`verify_agent_quality_certificate` verifies
    `--process-record` in full, first, before checking anything else) -- see
    `AGENT_QUALITY_POLICY_DISPOSITION`'s comment for why that removes the
    composition gap evaluator-quality has. `certification_eligible` is true
    only for a VERIFIED result over a process record whose base status was
    independently checked (`context.process_record_context.
    allow_unchecked_base_status is False`) -- an
    `AgentQualityVerificationResult` carries no trust-status evidence of its
    own, so a VERIFIED result looks identical whether or not that check ran;
    the eligibility gate supplies the missing condition.
    """
    process_record_context_payload = _required(
        context_payload, "process_record_context"
    )
    if not isinstance(process_record_context_payload, dict):
        raise click.UsageError("context.process_record_context must be a JSON object")
    if (
        process_record_context_payload.get("allow_unchecked_base_status") is True
        and trust_status_payload is not None
    ):
        raise click.UsageError(
            "--trust-status cannot be used with "
            "context.process_record_context.allow_unchecked_base_status=true"
        )

    try:
        context = _agent_quality_context(
            context_payload, anchor_value, schema, trust_status_payload
        )
        result = schema.verify_agent_quality_certificate(
            bundle_payload,
            context=context,
            process_record_bundle=process_record_payload,
        )
    except schema.ProcessRecordVerificationError as exc:
        _emit_agent_quality(
            as_json,
            {"code": exc.code, "certification_eligible": False},
            [exc.code, "certification_eligible=false"],
        )
        raise click.exceptions.Exit(1) from None
    except schema.AgentQualityVerificationError as exc:
        _emit_agent_quality(
            as_json,
            {"code": exc.code, "certification_eligible": False},
            [exc.code, "certification_eligible=false"],
        )
        raise click.exceptions.Exit(1) from None

    base_status_checked = (
        context.process_record_context.allow_unchecked_base_status is False
    )
    process_record_base_status = "checked" if base_status_checked else "not_checked"
    certification_eligible = (
        result.code == schema.AGENT_QUALITY_VERIFIED and base_status_checked
    )

    payload = {
        "code": result.code,
        "primary_objective_id": result.primary_objective_id,
        "nominal_coverage_ppm": result.nominal_coverage_ppm,
        "holdout_item_count": result.holdout_item_count,
        "holdout_item_count_source": "client_reported",
        "interval_verification_level": result.interval_verification_level,
        "split_verification_level": result.split_verification_level,
        "dataset_condition_code": result.dataset_condition_code,
        "evaluator_condition_code": result.evaluator_condition_code,
        "plan_ordering_note": result.plan_ordering_note,
        "process_record_base_status": process_record_base_status,
        "certification_eligible": certification_eligible,
    }
    # Machine code above stays stable; the scope line says what was verified.
    # An accepted abstained bundle has no claims, so no interval arithmetic ran.
    if result.code == schema.AGENT_QUALITY_CLAIM_ABSTAINED:
        verified_scope = (
            "verified_scope=signature_only (abstained: no quality claim; "
            "interval arithmetic not applicable)"
        )
    else:
        verified_scope = (
            "verified_scope=signature_and_arithmetic (quality numbers are "
            "reported by the customer; Traigent checked their arithmetic "
            "consistency but has not independently measured the quality or "
            "verified the underlying observations)"
        )
    lines = [
        result.code,
        verified_scope,
        f"primary_objective_id={result.primary_objective_id}",
        f"nominal_coverage_ppm={result.nominal_coverage_ppm}",
        f"holdout_item_count={result.holdout_item_count}",
        "holdout_item_count_source=client_reported",
        f"interval_verification_level={result.interval_verification_level}",
        f"split_verification_level={result.split_verification_level}",
        f"dataset_condition_code={result.dataset_condition_code}",
        f"evaluator_condition_code={result.evaluator_condition_code}",
        f"plan_ordering_note={result.plan_ordering_note}",
        f"process_record_base_status={process_record_base_status}",
        f"certification_eligible={'true' if certification_eligible else 'false'}",
    ]
    _emit_agent_quality(as_json, payload, lines)
    raise click.exceptions.Exit(0 if certification_eligible else 3)


def _emit_agent_quality(
    as_json: bool, payload: dict[str, Any], lines: list[str]
) -> None:
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True))
    else:
        for line in lines:
            click.echo(line)


# --- G1 client commands (keygen, pin, enroll, bind, abandon) ----------------
#
# All output is closed: fixed tokens, server-issued references and local paths.
# Nothing from a Backend response body or a local exception message is echoed.


def _g1_transport() -> Any:
    """The authenticated, egress-gated Backend transport (replaced in tests)."""
    from traigent.certification.transport import BackendG1Transport

    return BackendG1Transport(_make_backend_client())


def _g1_fail(message: str) -> Any:
    click.echo(message)
    raise click.exceptions.Exit(1)


@certify.command("keygen")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Key file to create (default: ~/.traigent/certification/keys/).",
)
def keygen_command(out_path: Path | None) -> None:
    """Generate a client signing key; prints the file path and public key only."""
    from traigent.certification.client_keys import ClientKeyError, generate_client_key

    try:
        key = generate_client_key(out_path)
    except ClientKeyError as err:
        _g1_fail(f"KEYGEN_REFUSED {err.code}")
        return
    click.echo(f"key_file={key.path}")
    click.echo(f"public_key_der_b64={key.public_key_der_b64}")


@certify.command("pin")
@click.option(
    "--from",
    "source",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Trusted pin document (issuer key, trust ring, project ref, ...).",
)
@click.option(
    "--pins",
    "pins_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Pin file to write (default: ~/.traigent/certification/pins.json).",
)
def pin_command(source: Path, pins_path: Path | None) -> None:
    """Install the trusted local pin file that also carries the project ref."""
    from traigent.certification.pins import PinError, write_pins

    try:
        pins = write_pins(source, pins_path)
    except PinError as err:
        _g1_fail(err.code)
        return
    click.echo(f"PINNED project_ref={pins.project_ref}")


@certify.command("enroll")
@click.option(
    "--key",
    "key_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--pins",
    "pins_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
def enroll_command(key_path: Path, pins_path: Path | None) -> None:
    """Enrol the client public key for the pinned project and pin its reference."""
    from traigent.certification.client_keys import (
        ClientKeyError,
        enroll_client_key,
        load_client_key,
    )
    from traigent.certification.pins import PinError, load_pins

    try:
        project_ref = load_pins(pins_path).project_ref
        key = load_client_key(key_path)
        ref = enroll_client_key(_g1_transport(), key, project_ref)
    except PinError as err:
        _g1_fail(err.code)
        return
    except ClientKeyError as err:
        _g1_fail(f"ENROLL_REFUSED {err.code}")
        return
    click.echo(f"ENROLLED client_key_ref={ref}")


@certify.command("bind")
@click.argument("root")
@click.option(
    "--key", "key_path", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
@click.option("--run-id", required=True)
@click.option(
    "--store-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Local snapshot store holding the tes1 snapshot.",
)
@click.option(
    "--pins", "pins_path", type=click.Path(dir_okay=False, path_type=Path), default=None
)
@click.option(
    "--evidence-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Local client-evidence directory (default: ~/.traigent/certification/evidence).",
)
@click.option("--sdk-ref", default=None, help="40-hex git SHA of this SDK build.")
@click.option(
    "--declared-mode",
    type=click.Choice(["offline", "current_online"]),
    default="current_online",
    show_default=True,
)
def bind_command(
    root: str,
    key_path: Path,
    run_id: str,
    store_dir: Path,
    pins_path: Path | None,
    evidence_dir: Path | None,
    sdk_ref: str | None,
    declared_mode: str,
) -> None:
    """Bind retained client evidence (ROOT is sha256:...) and obtain the certificate."""
    from traigent.certification import bind as bind_module
    from traigent.certification.client_evidence import (
        ClientEvidenceError,
        LocalFileCustody,
        reconstruct_and_compare,
    )
    from traigent.certification.client_keys import ClientKeyError, load_client_key
    from traigent.certification.pins import PinError, load_pins
    from traigent.snapshot.backends import LocalDirStore

    # Local reconstruction first: a failure here makes no network call.
    try:
        evidence = reconstruct_and_compare(
            root,
            custody=LocalFileCustody(evidence_dir),
            store=LocalDirStore(store_dir),
        )
    except ClientEvidenceError as err:
        _g1_fail(f"BIND_REFUSED {err.code}")
        return
    except Exception:
        _g1_fail("BIND_REFUSED EVIDENCE_STALE_OR_CORRUPT")
        return
    try:
        transport = _g1_transport()
        bind_module.check_g1_readiness(transport)
        resolved_ref, sdk_version = bind_module.resolve_sdk_identity(sdk_ref)
        pins = load_pins(pins_path)
        key = load_client_key(key_path)
        result = bind_module.bind_g1(
            transport,
            evidence,
            key,
            pins,
            run_id=run_id,
            sdk_ref=resolved_ref,
            sdk_version=sdk_version,
            declared_mode=declared_mode,
        )
    except bind_module.BindError as err:
        click.echo(f"BIND_REFUSED {err.code}")
        session_ref = getattr(err, "session_ref", None)
        if session_ref:
            click.echo(f"session_ref={session_ref}")
        raise click.exceptions.Exit(1) from None
    except (PinError, ClientKeyError) as err:
        _g1_fail(f"BIND_REFUSED {err.code}")
        return
    click.echo(f"BOUND certificate_ref={result.certificate_ref or 'none'}")


@certify.command("abandon")
@click.argument("build_session_ref", required=False)
@click.option(
    "--run",
    "run_id",
    default=None,
    help="Abandon this run's G1 intent when no build session exists yet.",
)
def abandon_command(build_session_ref: str | None, run_id: str | None) -> None:
    """Abandon a build session (its bsn: reference), or a run's G1 intent with --run."""
    from traigent.certification import bind as bind_module

    if (build_session_ref is None) == (run_id is None):
        _g1_fail("ABANDON_REFUSED")
        return
    try:
        if run_id is not None:
            bind_module.abandon_run(_g1_transport(), run_id)
        else:
            assert build_session_ref is not None
            bind_module.abandon_session(_g1_transport(), build_session_ref)
    except bind_module.BindError as err:
        _g1_fail(err.code)
        return
    if run_id is not None:
        click.echo(f"ABANDONED run {run_id}")
    else:
        click.echo(f"ABANDONED {build_session_ref}")
