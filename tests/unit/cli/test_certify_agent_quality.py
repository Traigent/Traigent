"""`traigent certify verify --kind agent-quality`.

Unlike evaluator-quality (pillar 3), this pillar composes the process-record
verification INSIDE the real verifier -- `verify_agent_quality_certificate`
verifies `--process-record` in full, first, so there is no cross-bundle
composition gap and no development-integration-only restriction. See the
policy comment above `AGENT_QUALITY_POLICY_DISPOSITION` in
`traigent/cli/certify_commands.py` for the full contrast with pillar 3.

These tests drive the REAL `traigent_schema` verifier end to end against
real, signed, digest-closed bundles generated from Schema's own test
builders (see `tests/fixtures/certification/PROVENANCE.md`) -- no mocking of
`_load_certification_schema` or of `verify_agent_quality_certificate`
anywhere in this file except the one place that maps a verifier exception to
an exit code, which is exercised with the real exception type.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli import certify_commands
from traigent.cli.certify_commands import certify

try:
    from traigent_schema.certification.agent_quality_verifier import (
        AgentQualityVerificationError,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    _missing = exc.name or ""
    if _missing != "traigent_schema" and not _missing.startswith("traigent_schema."):
        raise
    _SCHEMA_AVAILABLE = False
else:
    _SCHEMA_AVAILABLE = True

requires_schema = pytest.mark.skipif(
    not _SCHEMA_AVAILABLE,
    reason="traigent_schema (scripts/ci/schema-pin.txt) is not installed in this lane",
)


FIXTURES = Path(__file__).parents[2] / "fixtures" / "certification"
_PIN_RE = re.compile(
    r"^traigent-schema\s*@\s*git\+https://github\.com/Traigent/TraigentSchema\.git@([0-9a-f]{40})\s*$",
    re.MULTILINE,
)


def _pinned_schema_sha() -> str:
    pin_file = Path(__file__).parents[3] / "scripts" / "ci" / "schema-pin.txt"
    matches = _PIN_RE.findall(pin_file.read_text(encoding="utf-8"))
    assert len(matches) == 1
    return matches[0]


def _invoke(args: list[str]) -> object:
    return CliRunner().invoke(certify, args)


def _load(name: str) -> dict:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_args(
    *,
    bundle: Path | None = None,
    context: Path | None = None,
    anchor: Path | None = None,
    trust_status: Path | None = None,
    process_record: Path | None = None,
    kind: str | None,
    as_json: bool,
) -> list[str]:
    args = [
        "verify",
        str(bundle if bundle is not None else FIXTURES / "agent_quality_bundle.json"),
        "--context",
        str(
            context
            if context is not None
            else FIXTURES / "agent_quality_context_unchecked.json"
        ),
        "--anchor",
        str(anchor if anchor is not None else FIXTURES / "agent_quality_anchor.json"),
    ]
    if trust_status is not None:
        args += ["--trust-status", str(trust_status)]
    args += [
        "--process-record",
        str(
            process_record
            if process_record is not None
            else FIXTURES / "agent_quality_process_record.json"
        ),
    ]
    if kind is not None:
        args += ["--kind", kind]
    if as_json:
        args.append("--json")
    return args


@requires_schema
def test_no_kind_agent_quality_context_is_never_auto_detected() -> None:
    result = _invoke(
        [
            "verify",
            str(FIXTURES / "agent_quality_bundle.json"),
            "--context",
            str(FIXTURES / "agent_quality_context_unchecked.json"),
            "--anchor",
            str(FIXTURES / "agent_quality_anchor.json"),
            "--process-record",
            str(FIXTURES / "agent_quality_process_record.json"),
        ]
    )
    assert result.exit_code == 2, result.output
    assert "agent-quality" in result.output


@requires_schema
def test_kind_agent_quality_without_process_record_is_usage_error() -> None:
    result = _invoke(
        [
            "verify",
            str(FIXTURES / "agent_quality_bundle.json"),
            "--context",
            str(FIXTURES / "agent_quality_context_unchecked.json"),
            "--anchor",
            str(FIXTURES / "agent_quality_anchor.json"),
            "--kind",
            "agent-quality",
        ]
    )
    assert result.exit_code == 2, result.output
    assert "--process-record" in result.output


@requires_schema
def test_unchecked_golden_bundle_is_verified_but_not_eligible() -> None:
    """The golden bundle over an unchecked (allow_unchecked_base_status=true)
    process record verifies, but certification_eligible is false and the
    exit code is 3 -- see AGENT_QUALITY_POLICY_DISPOSITION's comment."""
    result = _invoke(_base_args(kind="agent-quality", as_json=True))
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "AGENT_QUALITY_VERIFIED"
    assert payload["process_record_base_status"] == "not_checked"
    assert payload["certification_eligible"] is False
    assert payload["primary_objective_id"] == "obj.accuracy.exact_match.v1"
    assert payload["nominal_coverage_ppm"] == 950000
    assert payload["holdout_item_count"] == 200
    assert payload["interval_verification_level"] == "construction_recomputed_v1"
    assert payload["split_verification_level"] == "issuer_attested_v1"


@requires_schema
def test_unchecked_golden_bundle_human_output() -> None:
    result = _invoke(_base_args(kind="agent-quality", as_json=False))
    assert result.exit_code == 3, result.output
    lines = result.output.splitlines()
    assert lines[0] == "AGENT_QUALITY_VERIFIED"
    assert "process_record_base_status=not_checked" in lines
    assert "certification_eligible=false" in lines


@requires_schema
def test_checked_golden_bundle_is_verified_and_eligible() -> None:
    """The SAME golden agent-quality bundle, over a process record whose
    base status WAS independently checked (a fresh active trust-status
    snapshot over the anchor that actually signs it) -- the only path this
    CLI ever reports certification_eligible=true, and exit 0."""
    result = _invoke(
        _base_args(
            context=FIXTURES / "agent_quality_context_checked.json",
            anchor=FIXTURES / "agent_quality_trust_anchor.json",
            trust_status=FIXTURES / "agent_quality_trust_status_active.json",
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "AGENT_QUALITY_VERIFIED"
    assert payload["process_record_base_status"] == "checked"
    assert payload["certification_eligible"] is True


@requires_schema
def test_abstained_bundle_accepted_is_not_eligible(tmp_path: Path) -> None:
    context = _load("agent_quality_context_unchecked.json")
    context["accept_abstained_bundle"] = True
    context_path = _write(tmp_path, "context.json", context)

    result = _invoke(
        _base_args(
            bundle=FIXTURES / "agent_quality_abstained_bundle.json",
            context=context_path,
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "AGENT_QUALITY_CLAIM_ABSTAINED"
    assert payload["certification_eligible"] is False


@requires_schema
def test_abstained_bundle_not_accepted_is_claim_not_verified(tmp_path: Path) -> None:
    context = _load("agent_quality_context_unchecked.json")
    context["accept_abstained_bundle"] = False
    context_path = _write(tmp_path, "context.json", context)

    result = _invoke(
        _base_args(
            bundle=FIXTURES / "agent_quality_abstained_bundle.json",
            context=context_path,
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "CLAIM_NOT_VERIFIED"
    assert payload["certification_eligible"] is False


@requires_schema
def test_tamper_signature_byte_flip_is_issuer_signature_invalid() -> None:
    result = _invoke(
        _base_args(
            bundle=FIXTURES / "agent_quality_bundle_tamper_signature.json",
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "ISSUER_SIGNATURE_INVALID"
    assert payload["certification_eligible"] is False


@requires_schema
def test_tamper_signature_byte_flip_is_actually_tampered() -> None:
    """Not vacuous: the tamper fixture's signature differs from the golden
    bundle's own -- proves the negative above exercises a real mutation."""
    golden = _load("agent_quality_bundle.json")
    tampered = _load("agent_quality_bundle_tamper_signature.json")
    assert golden["signature"]["signature"] != tampered["signature"]["signature"]
    golden_copy = copy.deepcopy(golden)
    golden_copy["signature"] = tampered["signature"]
    assert golden_copy != golden


@requires_schema
def test_tamper_wrong_commitment_ref_is_commitment_ref_mismatch() -> None:
    result = _invoke(
        _base_args(
            context=FIXTURES / "agent_quality_context_tamper_commitment_ref.json",
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "COMMITMENT_REF_MISMATCH"
    assert payload["certification_eligible"] is False


@requires_schema
def test_tamper_wrong_build_session_ref_is_scope_mismatch() -> None:
    result = _invoke(
        _base_args(
            context=FIXTURES / "agent_quality_context_tamper_scope.json",
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "SCOPE_MISMATCH"
    assert payload["certification_eligible"] is False


@requires_schema
@pytest.mark.parametrize(
    "fixture_name",
    [
        "agent_quality_bundle_tamper_interval_plus1.json",
        "agent_quality_bundle_tamper_interval_minus1.json",
    ],
)
def test_tamper_measured_claim_interval_off_by_one_ppm_is_interval_mismatch(
    fixture_name: str,
) -> None:
    result = _invoke(
        _base_args(
            bundle=FIXTURES / fixture_name,
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "INTERVAL_RECOMPUTATION_MISMATCH"
    assert payload["certification_eligible"] is False


@requires_schema
def test_tampered_process_record_surfaces_its_own_code_unrelabeled() -> None:
    """A tampered process record (one receipt digest flipped) is refused
    with the PROCESS RECORD's own code, verified directly first against
    `verify_process_record_certificate`, then asserted to be exactly what
    the CLI printed -- and that code must not be any member of
    AGENT_QUALITY_ERROR_CODES (it is not this module's vocabulary)."""
    from traigent_schema.certification import (
        AGENT_QUALITY_ERROR_CODES,
        ProcessRecordVerificationError,
        verify_process_record_certificate,
    )

    tampered_process_record = _load("agent_quality_process_record_tamper_receipt.json")

    context_payload = _load("agent_quality_context_unchecked.json")
    pr_context_payload = context_payload["process_record_context"]

    real_schema = certify_commands._load_certification_schema()
    anchor_value = certify_commands._anchor(
        _load("agent_quality_anchor.json"), real_schema
    )
    pr_context = certify_commands._process_context(
        pr_context_payload, anchor_value, real_schema
    )

    with pytest.raises(ProcessRecordVerificationError) as caught:
        verify_process_record_certificate(
            tampered_process_record, context=pr_context, trust_status=None
        )
    direct_code = caught.value.code
    assert direct_code not in AGENT_QUALITY_ERROR_CODES

    result = _invoke(
        _base_args(
            process_record=FIXTURES / "agent_quality_process_record_tamper_receipt.json",
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == direct_code
    assert payload["certification_eligible"] is False


@requires_schema
def test_trust_status_rejected_when_allow_unchecked_base_status_true(
    tmp_path: Path,
) -> None:
    result = _invoke(
        _base_args(
            trust_status=FIXTURES / "agent_quality_trust_status_active.json",
            kind="agent-quality",
            as_json=True,
        )
    )
    assert result.exit_code == 2, result.output
    assert "--trust-status" in result.output
    assert "allow_unchecked_base_status" in result.output


@requires_schema
def test_kind_agent_quality_technical_failure_reports_code_and_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one mocked test in this file: maps a real
    AgentQualityVerificationError to exit 1 and its code, without depending
    on which particular bundle condition triggers it."""
    real_schema = certify_commands._load_certification_schema()

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise AgentQualityVerificationError("BUNDLE_SHAPE", "bundle")

    patched = copy.copy(real_schema)
    patched.verify_agent_quality_certificate = _raise
    monkeypatch.setattr(certify_commands, "_load_certification_schema", lambda: patched)
    result = _invoke(_base_args(kind="agent-quality", as_json=True))
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "BUNDLE_SHAPE"
    assert payload["certification_eligible"] is False


def _installed_schema_commit() -> str | None:
    """The commit id pip recorded for the installed `traigent-schema`
    dist, read from dist-info `direct_url.json` -- the authority for what
    is actually on disk, independent of any pin file or constant."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        direct_url_text = distribution("traigent-schema").read_text("direct_url.json")
    except PackageNotFoundError:  # pragma: no cover - only if uninstalled
        return None
    if direct_url_text is None:
        return None
    direct_url = json.loads(direct_url_text)
    return direct_url.get("vcs_info", {}).get("commit_id")


@requires_schema
def test_eligibility_policy_pin_matches_installed_schema() -> None:
    """Guard test (mirrors evaluator-quality's). Fails closed if the Schema
    pin, the installed agent_quality_verifier module source, the
    agent-quality JSON schema, or any of the four registry documents (and
    their digest sidecars) drift from the reviewed disposition recorded in
    AGENT_QUALITY_POLICY_DISPOSITION in traigent/cli/certify_commands.py --
    forcing that record to be revisited (and, if the disposition still
    holds, updated) in the same PR that moves any of them."""
    disposition = certify_commands.AGENT_QUALITY_POLICY_DISPOSITION
    assert disposition["schema_pin"] == _pinned_schema_sha()
    assert (
        disposition["verifier_revision"]
        == certify_commands.agent_quality_verifier_revision()
    )

    installed_commit = _installed_schema_commit()
    if installed_commit is not None:
        assert installed_commit == disposition["schema_pin"]


@requires_schema
def test_eligibility_policy_pin_mutation_alone_fails_the_guard() -> None:
    mutated_pin = "1111111111111111111111111111111111111111"
    assert mutated_pin != _pinned_schema_sha()
    installed_commit = _installed_schema_commit()
    if installed_commit is not None:
        assert mutated_pin != installed_commit

    mutated_revision = "0" * 64
    assert mutated_revision != certify_commands.agent_quality_verifier_revision()


@requires_schema
def test_verifier_revision_covers_schema_json_not_only_verifier_module(
    tmp_path: Path,
) -> None:
    """The hash must move if the agent-quality JSON schema changes, not
    only if agent_quality_verifier.py changes. Hashes tmp copies of the
    installed artifacts -- never writes to the installed package."""
    installed_files = certify_commands._agent_quality_artifact_paths()
    baseline = certify_commands.agent_quality_verifier_revision()

    copies = [tmp_path / file.name for file in installed_files]
    for source, dest in zip(installed_files, copies, strict=True):
        dest.write_bytes(source.read_bytes())
    unaltered = certify_commands.agent_quality_verifier_revision(copies)
    assert unaltered == baseline

    schema_copy = tmp_path / "agent_quality_v1_schema.json"
    schema_copy.write_bytes(schema_copy.read_bytes() + b" ")
    mutated = certify_commands.agent_quality_verifier_revision(copies)
    assert mutated != unaltered


def test_verify_help_documents_agent_quality_exit_codes() -> None:
    result = CliRunner().invoke(certify, ["verify", "--help"])
    assert result.exit_code == 0, result.output
    assert "agent-quality" in result.output
