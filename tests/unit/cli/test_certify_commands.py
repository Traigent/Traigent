from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.certify_commands import certify

# ``traigent_schema`` is the git-pinned internal contract package (see
# scripts/ci/schema-pin.txt); it is deliberately not a dev extra, so "absent
# locally and in the collection lane, present in the unit lane" is the normal
# state. Same guard shape as tests/unit/economics/test_schema.py: only an error
# whose ``name`` identifies ``traigent_schema`` itself is eligible for the skip
# (a broken install of the package must still propagate), and the skip is a
# test-level marker so collection of this module never depends on the pin.
try:
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

# Installed as a sys.meta_path finder in a subprocess to simulate a user
# environment where the SDK was installed the supported way (plain
# `pip install`) and traigent-schema (an internal-only pinned dependency,
# see scripts/ci/schema-pin.txt) is absent.
_BLOCK_TRAIGENT_SCHEMA = """
import sys


class _Blocker:
    def find_spec(self, name, path, target=None):
        if name == "traigent_schema" or name.startswith("traigent_schema."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, _Blocker())
"""


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


@requires_schema
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


@requires_schema
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


@requires_schema
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


def test_cli_help_works_without_traigent_schema_installed() -> None:
    """traigent-schema is not a declared dependency (scripts/ci/schema-pin.txt);
    the whole CLI must stay usable when it is absent, not just `certify`."""
    script = (
        _BLOCK_TRAIGENT_SCHEMA
        + """
from click.testing import CliRunner
from traigent.cli.main import cli

runner = CliRunner()

top_level = runner.invoke(cli, ["--help"])
assert top_level.exit_code == 0, top_level.output

certify_help = runner.invoke(cli, ["certify", "--help"])
assert certify_help.exit_code == 0, certify_help.output
assert "verify" in certify_help.output

print("PROBE-OK")
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PROBE-OK" in completed.stdout


def test_verify_without_traigent_schema_installed_exits_with_usage_error() -> None:
    script = (
        _BLOCK_TRAIGENT_SCHEMA
        + f"""
import sys

sys.argv = [
    "traigent",
    "certify",
    "verify",
    {str(FIXTURES / "bundle.json")!r},
    "--context",
    {str(FIXTURES / "context.json")!r},
    "--anchor",
    {str(FIXTURES / "anchor.json")!r},
    "--trust-status",
    {str(FIXTURES / "trust_status.json")!r},
]

from traigent.cli.main import cli

cli()
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2, (
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )
    assert "Traceback" not in completed.stderr
    assert "traigent-schema is not installed" in completed.stderr
    assert "scripts/ci/schema-pin.txt" in completed.stderr
