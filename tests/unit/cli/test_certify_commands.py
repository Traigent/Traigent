from __future__ import annotations

import copy
import json
from pathlib import Path

from click.testing import CliRunner
from traigent_schema.certification import (
    PROCESS_RECORD_ERROR_CODES,
    ProcessRecordVerificationContext,
    TrustAnchorKeyV1,
    verify_process_record_certificate,
)
from traigent_schema.certification.process_record_verifier import (
    ProcessRecordVerificationError,
)
from traigent_schema.certification.relying_party_verifier import VerificationContext

from traigent.cli.certify_commands import certify


FIXTURES = Path(__file__).parents[2] / "fixtures" / "certification"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _context(payload: dict, anchor: dict) -> ProcessRecordVerificationContext:
    return ProcessRecordVerificationContext(
        expected_materials_digest=payload["expected_materials_digest"],
        certificate_ref=payload["certificate_ref"],
        base_context=VerificationContext(**payload["base_context"]),
        expected_project_ref=payload["expected_project_ref"],
        expected_build_session_ref=payload["expected_build_session_ref"],
        expected_agent_commitment_ref=payload["expected_agent_commitment_ref"],
        expected_dataset_commitment_ref=payload["expected_dataset_commitment_ref"],
        expected_evaluator_commitment_ref=payload["expected_evaluator_commitment_ref"],
        expected_build_definition_commitment_ref=payload[
            "expected_build_definition_commitment_ref"
        ],
        allow_unchecked_base_status=payload["allow_unchecked_base_status"],
        verification_time=payload["verification_time"],
        trust_anchor=TrustAnchorKeyV1(**anchor),
    )


def test_verify_fixture_reports_checked_active() -> None:
    result = CliRunner().invoke(
        certify,
        [
            "verify",
            str(FIXTURES / "bundle.json"),
            "--context",
            str(FIXTURES / "context.json"),
            "--anchor",
            str(FIXTURES / "anchor.json"),
            "--trust-status",
            str(FIXTURES / "trust_status.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "VERIFIED",
        "trust_status_evidence=checked_active",
        "trust_status_effective_time=2026-09-08T10:10:00Z",
    ]


def test_verify_requires_process_record_for_dataset_context(tmp_path: Path) -> None:
    context = _load("context.json")
    context["process_context"] = context.copy()
    context.pop("base_context")
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    result = CliRunner().invoke(
        certify,
        [
            "verify",
            str(FIXTURES / "bundle.json"),
            "--context",
            str(context_path),
            "--anchor",
            str(FIXTURES / "anchor.json"),
        ],
    )
    assert result.exit_code == 2


def test_tamper_controls_have_discriminating_closed_codes() -> None:
    bundle = _load("bundle.json")
    context_payload = _load("context.json")
    anchor = _load("anchor.json")
    trust_status = _load("trust_status.json")
    context = _context(context_payload, anchor)

    tampered_bundle = copy.deepcopy(bundle)
    tampered_bundle["report"]["rows"][0]["registered_values"]["status_code"] = "failed"
    try:
        verify_process_record_certificate(
            tampered_bundle, context=context, trust_status=trust_status
        )
    except ProcessRecordVerificationError as exc:
        assert exc.code == "REPORT_MISMATCH"
        assert exc.code in PROCESS_RECORD_ERROR_CODES
    else:
        raise AssertionError("tampered report unexpectedly verified")

    tampered_context = copy.deepcopy(context_payload)
    tampered_context["expected_materials_digest"] = "sha256:" + "0" * 64
    tampered = _context(tampered_context, anchor)
    try:
        verify_process_record_certificate(
            bundle, context=tampered, trust_status=trust_status
        )
    except ProcessRecordVerificationError as exc:
        assert exc.code in PROCESS_RECORD_ERROR_CODES
        assert exc.code != "REPORT_MISMATCH"
    else:
        raise AssertionError("tampered context unexpectedly verified")
