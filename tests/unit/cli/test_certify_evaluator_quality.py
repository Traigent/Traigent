"""`traigent certify verify --kind evaluator-quality`.

This SDK release never reports an evaluator-quality certificate as eligible
for unrestricted acceptance -- see the policy comment above
``EVALUATOR_QUALITY_POLICY_DISPOSITION`` in ``traigent/cli/certify_commands.py``
for why (no relying-party composition proves the caller's
``expected_evaluator_commitment_ref`` traces to an independently-verified
process record, and this CLI does not synthesize that proof). These tests
drive the REAL ``traigent_schema`` verifier end to end against a real,
signed, digest-closed bundle generated from Schema's own test builder (see
``tests/fixtures/certification/PROVENANCE.md``) -- no mocking of
``_load_certification_schema`` or of ``verify_evaluator_quality_certificate``
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
    from traigent_schema.certification.evaluator_quality_verifier import (
        EvaluatorQualityVerificationError,
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
    development_integration: bool,
    as_json: bool,
) -> list[str]:
    args = [
        "verify",
        str(
            bundle if bundle is not None else FIXTURES / "evaluator_quality_bundle.json"
        ),
        "--context",
        str(
            context
            if context is not None
            else FIXTURES / "evaluator_quality_context.json"
        ),
        "--anchor",
        str(
            anchor if anchor is not None else FIXTURES / "evaluator_quality_anchor.json"
        ),
    ]
    if trust_status is not None:
        args += ["--trust-status", str(trust_status)]
    if process_record is not None:
        args += ["--process-record", str(process_record)]
    if kind is not None:
        args += ["--kind", kind]
    if development_integration:
        args.append("--development-integration")
    if as_json:
        args.append("--json")
    return args


@requires_schema
def test_no_kind_evaluator_quality_context_is_never_auto_detected() -> None:
    """context.json (process-record) legitimately carries
    expected_evaluator_commitment_ref too, so that field must NOT be the
    ambiguity marker -- only allow_unchecked_trust_status is unique to
    evaluator-quality."""
    result = _invoke(
        [
            "verify",
            str(FIXTURES / "evaluator_quality_bundle.json"),
            "--context",
            str(FIXTURES / "evaluator_quality_context.json"),
            "--anchor",
            str(FIXTURES / "evaluator_quality_anchor.json"),
        ]
    )
    assert result.exit_code == 2, result.output
    assert "evaluator-quality" in result.output


@requires_schema
def test_kind_evaluator_quality_without_development_integration_refuses() -> None:
    result = _invoke(
        _base_args(
            kind="evaluator-quality", development_integration=False, as_json=False
        )
    )
    assert result.exit_code == 1, result.output
    assert result.output.splitlines()[0] == "EVALUATOR_QUALITY_DEV_INTEGRATION_ONLY"
    assert "certification_eligible=false" in result.output


@requires_schema
def test_kind_evaluator_quality_without_development_integration_json() -> None:
    result = _invoke(
        _base_args(
            kind="evaluator-quality", development_integration=False, as_json=True
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_QUALITY_DEV_INTEGRATION_ONLY"
    assert payload["certification_eligible"] is False


@requires_schema
def test_kind_evaluator_quality_development_integration_runs_real_verifier_human() -> (
    None
):
    """The real, signed, digest-closed fixture bundle verifies fully
    (unchecked trust status) but is still never reported eligible -- the
    process-record composition this contract requires is not performed by
    this CLI (see module docstring)."""
    result = _invoke(
        _base_args(
            kind="evaluator-quality", development_integration=True, as_json=False
        )
    )
    assert result.exit_code == 3, result.output
    lines = result.output.splitlines()
    assert lines[0] == "EVALUATOR_QUALITY_VERIFIED"
    assert "instrument_adequacy=passed" in lines
    assert "overall_verdict=passed" in lines
    assert "trust_status_evidence=not_checked" in lines
    assert "certification_eligible=false" in lines


@requires_schema
def test_kind_evaluator_quality_development_integration_json_never_eligible() -> None:
    result = _invoke(
        _base_args(kind="evaluator-quality", development_integration=True, as_json=True)
    )
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["certification_eligible"] is False
    assert payload["code"] == "EVALUATOR_QUALITY_VERIFIED"
    assert payload["valid"] is True
    assert payload["instrument_adequacy"] == "passed"
    assert payload["overall_verdict"] == "passed"
    assert payload["trust_status_evidence"] == "not_checked"


@requires_schema
def test_checked_trust_status_active_reports_checked_evidence(tmp_path: Path) -> None:
    """The checked happy path: allow_unchecked_trust_status=false, an ACTIVE
    trust-status envelope, and the anchor that actually signs it."""
    context = _load("evaluator_quality_context.json")
    context["allow_unchecked_trust_status"] = False
    context_path = _write(tmp_path, "context.json", context)

    result = _invoke(
        _base_args(
            context=context_path,
            anchor=FIXTURES / "evaluator_quality_trust_anchor.json",
            trust_status=FIXTURES / "evaluator_quality_trust_status_active.json",
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_QUALITY_VERIFIED"
    assert payload["trust_status_evidence"] == "checked_active"
    assert payload["certification_eligible"] is False


@requires_schema
def test_tamper_signature_byte_flip_is_evaluator_issuer_signature_invalid(
    tmp_path: Path,
) -> None:
    bundle = _load("evaluator_quality_bundle.json")
    sig = bytearray(base64.b64decode(bundle["signature"]["signature"]))
    sig[0] ^= 0xFF
    bundle["signature"]["signature"] = base64.b64encode(bytes(sig)).decode("ascii")
    bundle_path = _write(tmp_path, "bundle.json", bundle)

    result = _invoke(
        _base_args(
            bundle=bundle_path,
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_ISSUER_SIGNATURE_INVALID"
    assert payload["certification_eligible"] is False


@requires_schema
def test_tamper_wrong_commitment_ref_is_evaluator_commitment_mismatch(
    tmp_path: Path,
) -> None:
    context = _load("evaluator_quality_context.json")
    context["expected_evaluator_commitment_ref"] = "sha256:" + "d" * 64
    context_path = _write(tmp_path, "context.json", context)

    result = _invoke(
        _base_args(
            context=context_path,
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_COMMITMENT_MISMATCH"
    assert payload["certification_eligible"] is False


@requires_schema
def test_revoked_trust_status_checked_is_key_revoked(tmp_path: Path) -> None:
    context = _load("evaluator_quality_context.json")
    context["allow_unchecked_trust_status"] = False
    context_path = _write(tmp_path, "context.json", context)

    result = _invoke(
        _base_args(
            context=context_path,
            anchor=FIXTURES / "evaluator_quality_trust_anchor.json",
            trust_status=FIXTURES / "evaluator_quality_trust_status_revoked.json",
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "KEY_REVOKED"
    assert payload["certification_eligible"] is False


@requires_schema
def test_stale_trust_status_snapshot_is_revocation_status_unavailable(
    tmp_path: Path,
) -> None:
    """verification_time far in the future of the active snapshot's
    effective_time makes the snapshot stale under the trust policy's
    max_age_seconds -- a checked-but-unusable trust status, distinct from a
    revoked one."""
    context = _load("evaluator_quality_context.json")
    context["allow_unchecked_trust_status"] = False
    context["verification_time"] = "2031-01-01T00:00:00Z"
    context_path = _write(tmp_path, "context.json", context)

    result = _invoke(
        _base_args(
            context=context_path,
            anchor=FIXTURES / "evaluator_quality_trust_anchor.json",
            trust_status=FIXTURES / "evaluator_quality_trust_status_active.json",
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "REVOCATION_STATUS_UNAVAILABLE"
    assert payload["certification_eligible"] is False


@requires_schema
def test_process_record_bundle_under_evaluator_quality_kind_is_bundle_shape() -> None:
    """A process-record-family bundle (wrong schema_version) passed under
    --kind evaluator-quality is refused as a shape error, not silently
    mis-verified as if it were an evaluator-quality bundle."""
    result = _invoke(
        _base_args(
            bundle=FIXTURES / "bundle.json",
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_BUNDLE_SHAPE"
    assert payload["certification_eligible"] is False


@requires_schema
def test_kind_evaluator_quality_technical_failure_reports_code_and_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one mocked test in this file: maps a real
    EvaluatorQualityVerificationError to exit 1 and its code, without
    depending on which particular bundle condition triggers it."""

    real_schema = certify_commands._load_certification_schema()

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise EvaluatorQualityVerificationError("EVALUATOR_SCHEMA", "bundle")

    patched = copy.copy(real_schema)
    patched.verify_evaluator_quality_certificate = _raise
    monkeypatch.setattr(certify_commands, "_load_certification_schema", lambda: patched)
    result = _invoke(
        _base_args(kind="evaluator-quality", development_integration=True, as_json=True)
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_SCHEMA"
    assert payload["certification_eligible"] is False


@requires_schema
def test_eligibility_policy_pin_matches_installed_schema() -> None:
    """Guard test (acceptance row: policy binding). Fails closed if the
    Schema pin, the installed evaluator_quality_verifier module source, the
    evaluator-quality JSON schema, or any of the three evaluator registry
    digest files drift from the reviewed disposition recorded in
    EVALUATOR_QUALITY_POLICY_DISPOSITION in
    traigent/cli/certify_commands.py -- forcing that record to be revisited
    (and, if the disposition still holds, updated) in the same PR that moves
    any of them. Also checks the *installed* distribution, not just the pin
    file, against the same pin -- two copies of the pin agreeing is not
    proof the install matches; the dist-info direct_url.json is the
    authority for what is actually on disk."""
    disposition = certify_commands.EVALUATOR_QUALITY_POLICY_DISPOSITION
    assert disposition["schema_pin"] == _pinned_schema_sha()
    assert (
        disposition["verifier_revision"]
        == certify_commands.evaluator_quality_verifier_revision()
    )

    from importlib.metadata import PackageNotFoundError, distribution

    try:
        direct_url_text = distribution("traigent-schema").read_text("direct_url.json")
    except PackageNotFoundError:  # pragma: no cover - only if uninstalled
        direct_url_text = None
    if direct_url_text is not None:
        direct_url = json.loads(direct_url_text)
        installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
        assert installed_commit == disposition["schema_pin"]


@requires_schema
def test_eligibility_policy_pin_mutation_alone_fails_the_guard(tmp_path: Path) -> None:
    """Proves the guard actually bites: a disposition whose schema_pin no
    longer matches the pin file must fail the same comparison the guard test
    above makes -- without needing a real scratch venv, since the guard
    reads the pin file and the installed artifacts, not the disposition
    dict's own internal consistency."""
    mutated = dict(certify_commands.EVALUATOR_QUALITY_POLICY_DISPOSITION)
    mutated["schema_pin"] = "1111111111111111111111111111111111111111"
    assert mutated["schema_pin"] != _pinned_schema_sha()

    mutated_revision = dict(certify_commands.EVALUATOR_QUALITY_POLICY_DISPOSITION)
    mutated_revision["verifier_revision"] = "0" * 64
    assert (
        mutated_revision["verifier_revision"]
        != certify_commands.evaluator_quality_verifier_revision()
    )


@requires_schema
def test_eligibility_policy_record_metadata_edit_alone_does_not_fail_the_guard() -> (
    None
):
    """Editing only reviewed_by/date (not schema_pin or verifier_revision)
    must not fail the guard -- the guard checks the technical pins against
    real installed artifacts, not the review metadata."""
    disposition = certify_commands.EVALUATOR_QUALITY_POLICY_DISPOSITION
    edited = dict(disposition)
    edited["reviewed_by"] = "someone-else"
    edited["date"] = "2099-01-01"
    assert edited["schema_pin"] == _pinned_schema_sha()
    assert (
        edited["verifier_revision"]
        == certify_commands.evaluator_quality_verifier_revision()
    )


@requires_schema
def test_verifier_revision_covers_schema_json_not_only_verifier_module() -> None:
    """P2-3: the hash must move if the evaluator-quality JSON schema changes,
    not only if evaluator_quality_verifier.py changes -- otherwise a Schema
    release that re-widens the emittable claim-id enum without touching the
    verifier's own source would leave the pinned revision valid."""
    import traigent_schema

    schema_path = (
        Path(traigent_schema.__file__).parent
        / "schemas"
        / "certification"
        / "evaluator_quality_v1_schema.json"
    )
    original = schema_path.read_bytes()
    baseline = certify_commands.evaluator_quality_verifier_revision()
    try:
        schema_path.write_bytes(original + b" ")
        mutated = certify_commands.evaluator_quality_verifier_revision()
    finally:
        schema_path.write_bytes(original)
    assert mutated != baseline


@requires_schema
def test_evq6_evq7_registered_but_not_emittable() -> None:
    """EVQ6 (efficiency) and EVQ7 (frontier) remain in the full claim-id
    vocabulary but are structurally excluded from
    EmittableEvaluatorQualityClaimIdV1 -- the enum every claim_material
    entry's claim_id is validated against -- so no v1 bundle can ever
    construct them. Exercised directly against the installed pinned Schema's
    JSON, not a hand-built bundle."""
    import traigent_schema

    schema_dir = Path(traigent_schema.__file__).parent / "schemas" / "certification"
    with open(
        schema_dir / "evaluator_quality_v1_schema.json", encoding="utf-8"
    ) as handle:
        document = json.load(handle)

    full_vocabulary = document["definitions"]["EvaluatorQualityClaimIdV1"]["enum"]
    assert {"EVQ6", "EVQ7"}.issubset(full_vocabulary)

    emittable = document["definitions"]["EmittableEvaluatorQualityClaimIdV1"]
    assert emittable["enum"] == ["EVQ1", "EVQ2", "EVQ3", "EVQ4", "EVQ5"]
    assert "EVQ6" not in emittable["enum"]
    assert "EVQ7" not in emittable["enum"]

    import jsonschema

    validator = jsonschema.Draft7Validator(emittable)
    assert validator.is_valid("EVQ5")
    assert not validator.is_valid("EVQ6")
    assert not validator.is_valid("EVQ7")


@requires_schema
def test_evq6_claim_id_through_real_cli_is_evaluator_schema(tmp_path: Path) -> None:
    """The real refusal for a bundle claiming EVQ6 is EVALUATOR_SCHEMA
    (schema rejection at the emittable-claim-id enum), asserted through the
    CLI against a mutated real bundle -- not just against the JSON schema
    document in isolation (see test_evq6_evq7_registered_but_not_emittable)."""
    bundle = _load("evaluator_quality_bundle.json")
    bundle["claim_material"][0]["claim_id"] = "EVQ6"
    bundle_path = _write(tmp_path, "bundle.json", bundle)

    result = _invoke(
        _base_args(
            bundle=bundle_path,
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_SCHEMA"


@requires_schema
def test_evq7_claim_id_through_real_cli_is_evaluator_schema(tmp_path: Path) -> None:
    bundle = _load("evaluator_quality_bundle.json")
    bundle["claim_material"][0]["claim_id"] = "EVQ7"
    bundle_path = _write(tmp_path, "bundle.json", bundle)

    result = _invoke(
        _base_args(
            bundle=bundle_path,
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_SCHEMA"


def test_evaluator_quality_verify_never_synthesizes_a_process_record() -> None:
    """`_verify_evaluator_quality` takes no process-record input (no
    `--process-record` composition) -- S must not synthesize the verified
    process record `EvaluatorQualityVerificationContext.
    expected_evaluator_commitment_ref` is documented as requiring; that is
    why eligibility is unconditionally false rather than conditional on one
    being supplied."""
    import inspect

    params = inspect.signature(certify_commands._verify_evaluator_quality).parameters
    assert "process_record" not in params
    assert "process_record_bundle" not in params


@requires_schema
def test_process_record_flag_rejected_under_evaluator_quality_kind() -> None:
    """P3-3: --process-record supplied alongside --kind evaluator-quality is
    a usage error through the actual CLI invocation, not merely a signature
    check on the implementation function -- so this still catches a future
    regression that reads the flag via click.get_current_context() instead
    of a function parameter."""
    result = _invoke(
        _base_args(
            process_record=FIXTURES / "bundle.json",
            kind="evaluator-quality",
            development_integration=True,
            as_json=True,
        )
    )
    assert result.exit_code == 2, result.output
    assert "--process-record" in result.output


def test_verify_help_documents_exit_codes() -> None:
    result = CliRunner().invoke(certify, ["verify", "--help"])
    assert result.exit_code == 0, result.output
    assert "3 = technically valid" in result.output
