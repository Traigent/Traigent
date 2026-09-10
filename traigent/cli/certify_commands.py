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
EVALUATOR_QUALITY_POLICY_DISPOSITION = {
    "schema_pin": "4b3373925cee6bd57071980285c58044165c90a4",
    "verifier_revision": "a2fd2e4e1222dc9e1589d26adcbb12b40dcbd5473e5c23cfb1e8e01ea39e333c",
    "reviewed_by": "spine-trail:st_46ecde7dd286",
    "date": "2026-09-10",
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


def _load_certification_schema() -> SimpleNamespace:
    """Lazily import traigent_schema.certification, called only inside `verify`.

    traigent-schema is deliberately not a declared dependency (PyPI rejects a
    direct-URL git dependency; see scripts/ci/schema-pin.txt), so importing it
    at module import time would break `traigent --help` for anyone who
    installed the SDK the supported way (plain `pip install`).
    """
    try:
        from traigent_schema.certification import (
            DatasetRecordVerificationContext,
            EvaluatorQualityVerificationContext,
            ProcessRecordVerificationContext,
            TrustAnchorKeyV1,
            verify_dataset_record_certificate,
            verify_evaluator_quality_certificate,
            verify_process_record_certificate,
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
        DatasetRecordVerificationContext=DatasetRecordVerificationContext,
        EvaluatorQualityVerificationContext=EvaluatorQualityVerificationContext,
        ProcessRecordVerificationContext=ProcessRecordVerificationContext,
        TrustAnchorKeyV1=TrustAnchorKeyV1,
        verify_dataset_record_certificate=verify_dataset_record_certificate,
        verify_evaluator_quality_certificate=verify_evaluator_quality_certificate,
        verify_process_record_certificate=verify_process_record_certificate,
        DatasetRecordVerificationError=DatasetRecordVerificationError,
        EvaluatorQualityVerificationError=EvaluatorQualityVerificationError,
        ProcessRecordVerificationError=ProcessRecordVerificationError,
        VerificationContext=VerificationContext,
    )


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise click.UsageError(f"could not read {description}: {exc}") from exc
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
        raise click.UsageError("context is invalid") from exc


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
        raise click.UsageError("context is invalid") from exc


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
        raise click.UsageError("context is invalid") from exc


def _resolve_kind(kind: str | None, context_payload: dict[str, Any]) -> str:
    """Pick the verification family, refusing to guess evaluator-quality.

    `--kind evaluator-quality` is never auto-detected. Its context marker is
    `allow_unchecked_trust_status` (the S10 opt-in unique to
    `EvaluatorQualityVerificationContext`) -- NOT
    `expected_evaluator_commitment_ref`, which a legitimate process-record
    context also carries (see `_process_context`), so that field cannot be
    used to distinguish the families. A context carrying the
    evaluator-quality marker without an explicit `--kind` is refused
    outright rather than routed into the wrong verifier.
    """
    if kind is not None:
        return kind
    if "allow_unchecked_trust_status" in context_payload:
        raise click.UsageError(
            "context contains allow_unchecked_trust_status (an "
            "evaluator-quality marker); pass --kind evaluator-quality "
            "explicitly -- it is never auto-detected"
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
    type=click.Choice(["dataset-record", "process-record", "evaluator-quality"]),
    default=None,
    help="Verification family. Required for evaluator-quality (never guessed).",
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
        "2 = usage error, 3 = technically valid, not certification-eligible."
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
    as_json: bool,
) -> None:
    """Verify BUNDLE using independently supplied context and public anchor."""
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
    click.echo("VERIFIED")
    if hasattr(result, "trust_status_evidence"):
        click.echo(f"trust_status_evidence={result.trust_status_evidence}")
        click.echo(f"trust_status_effective_time={result.trust_status_effective_time}")


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
