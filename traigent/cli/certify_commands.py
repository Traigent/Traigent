"""Offline, keyless certificate verification commands."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from traigent_schema.certification import (
        DatasetRecordVerificationContext,
        ProcessRecordVerificationContext,
        TrustAnchorKeyV1,
    )


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
            ProcessRecordVerificationContext,
            TrustAnchorKeyV1,
            verify_dataset_record_certificate,
            verify_process_record_certificate,
        )
        from traigent_schema.certification.dataset_record_verifier import (
            DatasetRecordVerificationError,
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
        ProcessRecordVerificationContext=ProcessRecordVerificationContext,
        TrustAnchorKeyV1=TrustAnchorKeyV1,
        verify_dataset_record_certificate=verify_dataset_record_certificate,
        verify_process_record_certificate=verify_process_record_certificate,
        DatasetRecordVerificationError=DatasetRecordVerificationError,
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
def verify(
    bundle: Path,
    context_path: Path,
    anchor: Path,
    trust_status: Path | None,
    process_record: Path | None,
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

    try:
        if "process_context" in context_payload:
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
