"""`traigent certify verify --kind evaluator-quality`.

This SDK release never reports an evaluator-quality certificate as eligible
for unrestricted acceptance -- see the policy comment above
``EVALUATOR_QUALITY_POLICY_SCHEMA_PIN`` in ``traigent/cli/certify_commands.py``
for why (no relying-party composition proves the caller's
``expected_evaluator_commitment_ref`` traces to an independently-verified
process record, and this CLI does not synthesize that proof). These tests
exercise the CLI-side policy gate the way an integrator will hit it: real
``traigent_schema`` dataclasses and error types, with the cryptographic
verifier itself mocked at the one function boundary
(``verify_evaluator_quality_certificate``) so the tests do not depend on
hand-constructing a fully signed bundle -- the thing this packet adds is the
policy gate around that call, not the cryptography inside it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli import certify_commands
from traigent.cli.certify_commands import certify

try:
    from traigent_schema.certification import (
        EvaluatorQualityVerificationContext,
        EvaluatorQualityVerificationResult,
    )
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


def _base_args(
    *, kind: str | None, development_integration: bool, as_json: bool
) -> list[str]:
    args = [
        "verify",
        str(FIXTURES / "bundle.json"),
        "--context",
        str(FIXTURES / "evaluator_quality_context.json"),
        "--anchor",
        str(FIXTURES / "anchor.json"),
    ]
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
            str(FIXTURES / "bundle.json"),
            "--context",
            str(FIXTURES / "evaluator_quality_context.json"),
            "--anchor",
            str(FIXTURES / "anchor.json"),
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
def test_kind_evaluator_quality_development_integration_runs_but_stays_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a bundle that verifies fully (the default, all-passed result) is
    never reported eligible -- the process-record composition this contract
    requires is not performed by this CLI (see module docstring)."""
    monkeypatch.setattr(
        certify_commands,
        "_load_certification_schema",
        lambda: _patched_schema(verify_result=EvaluatorQualityVerificationResult()),
    )
    result = _invoke(
        _base_args(
            kind="evaluator-quality", development_integration=True, as_json=False
        )
    )
    assert result.exit_code == 3, result.output
    lines = result.output.splitlines()
    assert lines[0] == "EVALUATOR_QUALITY_VERIFIED"
    assert "certification_eligible=false" in lines


@requires_schema
def test_kind_evaluator_quality_development_integration_json_never_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        certify_commands,
        "_load_certification_schema",
        lambda: _patched_schema(verify_result=EvaluatorQualityVerificationResult()),
    )
    result = _invoke(
        _base_args(kind="evaluator-quality", development_integration=True, as_json=True)
    )
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["certification_eligible"] is False
    assert payload["code"] == "EVALUATOR_QUALITY_VERIFIED"
    assert payload["valid"] is True


@requires_schema
def test_kind_evaluator_quality_technical_failure_reports_code_and_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise EvaluatorQualityVerificationError("EVALUATOR_SCHEMA", "bundle")

    monkeypatch.setattr(
        certify_commands,
        "_load_certification_schema",
        lambda: _patched_schema(verify_side_effect=_raise),
    )
    result = _invoke(
        _base_args(kind="evaluator-quality", development_integration=True, as_json=True)
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "EVALUATOR_SCHEMA"
    assert payload["certification_eligible"] is False


def _patched_schema(
    *,
    verify_result: object = None,
    verify_side_effect: object = None,
) -> object:
    """A stand-in for `_load_certification_schema()` that uses the REAL
    context/error/anchor types (so `EvaluatorQualityVerificationContext`'s own
    validation still runs) but replaces only
    `verify_evaluator_quality_certificate` -- the one call this packet does
    not need to exercise cryptographically."""
    from types import SimpleNamespace

    from traigent_schema.certification import TrustAnchorKeyV1

    def _verify(*_args: object, **_kwargs: object) -> object:
        if verify_side_effect is not None:
            return verify_side_effect(*_args, **_kwargs)
        return verify_result

    return SimpleNamespace(
        TrustAnchorKeyV1=TrustAnchorKeyV1,
        EvaluatorQualityVerificationContext=EvaluatorQualityVerificationContext,
        EvaluatorQualityVerificationError=EvaluatorQualityVerificationError,
        verify_evaluator_quality_certificate=_verify,
    )


@requires_schema
def test_eligibility_policy_pin_matches_installed_schema() -> None:
    """Guard test (acceptance row: policy binding). Fails closed if the
    Schema pin or the installed evaluator_quality_verifier module source
    drifts from the reviewed disposition recorded as the constants and
    comment above them in traigent/cli/certify_commands.py -- forcing that
    comment to be revisited (and, if the disposition still holds, the two
    constants bumped) in the same PR that moves either."""
    assert certify_commands.EVALUATOR_QUALITY_POLICY_SCHEMA_PIN == _pinned_schema_sha()
    assert (
        certify_commands.EVALUATOR_QUALITY_POLICY_VERIFIER_REVISION
        == certify_commands.evaluator_quality_verifier_revision()
    )


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
