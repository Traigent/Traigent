"""`certify verify --open-commitments` and `certify finalize` (C1 SDK step; mocked backend)."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

pytest.importorskip("traigent_schema.certification.commitment_linkage")

import traigent_schema.certification.commitment_linkage as cl  # noqa: E402
import traigent_schema.fp2 as fp2  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402
from traigent_schema.certification.relying_party_verifier import (  # noqa: E402
    _ISSUER_SPKI_DOMAIN,
)

from traigent.cli import certify_commands as cc  # noqa: E402
from traigent.cli.certify_commands import certify  # noqa: E402
from traigent.cloud.process_record_linkage import (  # noqa: E402
    BUNDLE_SCHEMA,
    DISCLOSURE_LINE,
    ProcessRecordRequestError,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "certification"
FP_DATASET = "sha256:" + "ab" * 32
FP_CONFIG = "sha256:" + "cd" * 32
OPEN_DATASET = bytes(range(32))
OPEN_CONFIG = bytes(range(32, 64))
SESSION = "bsess_" + "0" * 32

_PRIVATE = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(64, 96)))
_DER = _PRIVATE.public_key().public_bytes(
    serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
)


def _record() -> dict[str, Any]:
    """The fixture record, re-keyed to a test issuer key (verification is stubbed)."""
    record = json.loads((FIXTURES / "bundle.json").read_text())
    issuer = record["verification_materials_v0"]["issuer"]
    issuer.update(
        algorithm="ed25519",
        public_key_der_b64=base64.b64encode(_DER).decode("ascii"),
        public_key_digest="sha256:"
        + hashlib.sha256(_ISSUER_SPKI_DOMAIN + b"\x00" + _DER).hexdigest(),
    )
    return record


def _sidecar(
    record: dict[str, Any], *, config_available: bool = True
) -> dict[str, Any]:
    issuer = record["verification_materials_v0"]["issuer"]
    entries = [
        {
            "role": "dataset",
            "basis": "order_free_multiset",
            "attestation": "client_declared",
            "commitment": cl.compute_commitment(
                role="dataset",
                basis="order_free_multiset",
                fp_digest=FP_DATASET,
                opening=OPEN_DATASET,
            ),
        }
    ]
    if config_available:
        entries.append(
            {
                "role": "config_space",
                "basis": "config_space",
                "attestation": "client_declared",
                "commitment": cl.compute_commitment(
                    role="config_space",
                    basis="config_space",
                    fp_digest=FP_CONFIG,
                    opening=OPEN_CONFIG,
                ),
            }
        )
    else:
        entries.append(
            {
                "role": "config_space",
                "basis": "unavailable",
                "attestation": "client_declared",
            }
        )
    unsigned = {
        "schema_version": "traigent.commitment_linkage.unsigned.v1",
        "process_record_digest": record["signature"]["unsigned_manifest_digest"],
        "project_ref": record["unsigned_manifest"]["project_ref"],
        "build_session_ref": record["unsigned_manifest"]["build_session_ref"],
        "entries": entries,
    }
    canonical = fp2.canonicalize(unsigned).encode("utf-8")
    return {
        "schema_version": "traigent.commitment_linkage.v1",
        "unsigned_linkage": unsigned,
        "signature": {
            "schema_version": "traigent.commitment_linkage.signature.v1",
            "algorithm": "ed25519",
            "issuer_key_ref": issuer["key_ref"],
            "trust_ring_ref": issuer["trust_ring_ref"],
            "signed_payload": "unsigned_commitment_linkage",
            "unsigned_linkage_digest": "sha256:"
            + hashlib.sha256(
                cl.UNSIGNED_LINKAGE_DOMAIN + b"\x00" + canonical
            ).hexdigest(),
            "signature": base64.b64encode(
                _PRIVATE.sign(cl.ISSUER_SIGNATURE_DOMAIN + b"\x00" + canonical)
            ).decode("ascii"),
        },
    }


def _bundle_file(
    tmp_path: Path,
    linkage: dict[str, Any],
    *,
    fp: dict[str, str] | None,
    openings: bool = True,
) -> Path:
    data: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "build_session_row_id": SESSION,
        "commitment_linkage": linkage,
    }
    if openings:
        data["openings"] = {
            "dataset": base64.b64encode(OPEN_DATASET).decode(),
            "config_space": base64.b64encode(OPEN_CONFIG).decode(),
        }
    if fp is not None:
        data["fp_digests"] = fp
    path = tmp_path / "commitments.json"
    path.write_text(json.dumps(data))
    os.chmod(path, 0o600)
    return path


@pytest.fixture
def stub_record_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    real = cc._load_certification_schema()
    patched = SimpleNamespace(**vars(real))
    patched.verify_process_record_certificate = lambda *_a, **_k: SimpleNamespace(
        valid=True,
        trust_status_evidence="checked_active",
        trust_status_effective_time="2026-09-08T10:10:00Z",
    )
    monkeypatch.setattr(cc, "_load_certification_schema", lambda: patched)


def _verify(tmp_path: Path, record: dict[str, Any], commitments: Path, *extra: str):
    rec = tmp_path / "record.json"
    rec.write_text(json.dumps(record))
    return CliRunner().invoke(
        certify,
        [
            "verify",
            str(rec),
            "--context",
            str(FIXTURES / "context.json"),
            "--anchor",
            str(FIXTURES / "anchor.json"),
            "--trust-status",
            str(FIXTURES / "trust_status.json"),
            "--open-commitments",
            str(commitments),
            *extra,
        ],
    )


FP_BOTH = {"dataset": FP_DATASET, "config_space": FP_CONFIG}


def test_open_commitments_match_prints_per_role_and_exact_disclosure(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    result = _verify(
        tmp_path, record, _bundle_file(tmp_path, _sidecar(record), fp=FP_BOTH)
    )
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "dataset: COMMITMENT_OPENED_MATCH" in lines
    assert "config_space: COMMITMENT_OPENED_MATCH" in lines
    assert lines[-1] == DISCLOSURE_LINE
    assert DISCLOSURE_LINE == (
        "Dataset and configuration-space declarations only; agent, evaluator and build "
        "definition are not covered. This is not execution evidence."
    )
    assert base64.b64encode(OPEN_DATASET).decode() not in result.output


def test_open_commitments_mismatch_exits_1(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    wrong = {"dataset": "sha256:" + "ee" * 32, "config_space": FP_CONFIG}
    result = _verify(
        tmp_path, record, _bundle_file(tmp_path, _sidecar(record), fp=wrong)
    )
    assert result.exit_code == 1
    assert "dataset: COMMITMENT_OPENED_MISMATCH" in result.output
    assert "config_space: COMMITMENT_OPENED_MATCH" in result.output
    assert DISCLOSURE_LINE in result.output


def test_open_commitments_unavailable_without_local_fp_or_entry(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    # no fp digests known locally -> never guessed
    result = _verify(
        tmp_path, record, _bundle_file(tmp_path, _sidecar(record), fp=None)
    )
    assert result.exit_code == 0, result.output
    assert "dataset: COMMITMENT_UNAVAILABLE" in result.output
    assert "config_space: COMMITMENT_UNAVAILABLE" in result.output
    # issuer declared config_space unavailable -> schema says UNAVAILABLE
    sidecar = _sidecar(record, config_available=False)
    result = _verify(tmp_path, record, _bundle_file(tmp_path, sidecar, fp=FP_BOTH))
    assert "dataset: COMMITMENT_OPENED_MATCH" in result.output
    assert "config_space: COMMITMENT_UNAVAILABLE" in result.output
    assert result.output.splitlines()[-1] == DISCLOSURE_LINE


def test_sidecar_bound_to_another_record_is_invalid_not_a_match(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    sidecar = _sidecar(record)
    other = copy.deepcopy(record)
    other["signature"]["unsigned_manifest_digest"] = "sha256:" + "11" * 32
    result = _verify(tmp_path, other, _bundle_file(tmp_path, sidecar, fp=FP_BOTH))
    assert result.exit_code == 1
    assert "COMMITMENT_LINKAGE_INVALID RECORD_BINDING_MISMATCH" in result.output
    assert "MATCH" not in result.output.replace("MISMATCH", "")


def test_issuer_key_comes_from_record_not_from_sidecar(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    sidecar = _sidecar(record)
    record["verification_materials_v0"]["issuer"]["key_ref"] = "key:differentkey"
    result = _verify(tmp_path, record, _bundle_file(tmp_path, sidecar, fp=FP_BOTH))
    assert result.exit_code == 1
    assert "COMMITMENT_LINKAGE_INVALID KEY_MISMATCH" in result.output


def test_verify_refuses_group_readable_bundle(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    bundle = _bundle_file(tmp_path, _sidecar(record), fp=FP_BOTH)
    os.chmod(bundle, 0o644)
    result = _verify(tmp_path, record, bundle)
    assert result.exit_code == 1
    assert "BUNDLE_PERMISSIONS" in result.output
    assert "MATCH" not in result.output


def test_verify_refuses_non_0600_owner_only_bundle(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    bundle = _bundle_file(tmp_path, _sidecar(record), fp=FP_BOTH)
    os.chmod(bundle, 0o400)
    result = _verify(tmp_path, record, bundle)
    assert result.exit_code == 1
    assert "BUNDLE_PERMISSIONS" in result.output


def test_verify_refuses_bundle_owned_by_another_uid(
    tmp_path: Path, stub_record_verification: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record()
    bundle = _bundle_file(tmp_path, _sidecar(record), fp=FP_BOTH)
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(bundle).st_uid + 1)
    result = _verify(tmp_path, record, bundle)
    assert result.exit_code == 1
    assert "BUNDLE_PERMISSIONS" in result.output
    assert "MATCH" not in result.output


def test_invalid_record_result_without_raising_attempts_no_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = cc._load_certification_schema()
    patched = SimpleNamespace(**vars(real))
    patched.verify_process_record_certificate = lambda *_a, **_k: SimpleNamespace(
        valid=False, code="SIGNATURE_INVALID"
    )
    monkeypatch.setattr(cc, "_load_certification_schema", lambda: patched)
    called: list[str] = []
    monkeypatch.setattr(
        cl, "verify_and_open_commitment", lambda *a, **k: called.append("open")
    )
    monkeypatch.setattr(
        cl, "verify_commitment_linkage", lambda *a, **k: called.append("verify")
    )
    record = _record()
    result = _verify(
        tmp_path, record, _bundle_file(tmp_path, _sidecar(record), fp=FP_BOTH)
    )
    assert result.exit_code != 0
    assert called == []
    assert "COMMITMENT_OPENED" not in result.output


def test_open_commitments_refused_off_posix_before_reading_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.cloud import process_record_linkage as prl

    monkeypatch.setattr(prl, "_is_posix", lambda: False)
    monkeypatch.setattr(
        cc, "_read_json", lambda *a, **k: pytest.fail("read before platform check")
    )
    record = _record()
    result = _verify(
        tmp_path, record, _bundle_file(tmp_path, _sidecar(record), fp=FP_BOTH)
    )
    assert result.exit_code == 1
    assert "LINKAGE_UNSUPPORTED_PLATFORM" in result.output


def test_finalize_link_commitments_refused_off_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.cloud import process_record_linkage as prl

    monkeypatch.setattr(prl, "_is_posix", lambda: False)
    backend = _FakeBackend({"success": True, "data": {}})
    result = _finalize(
        tmp_path,
        monkeypatch,
        backend,
        "--link-commitments",
    )
    assert result.exit_code == 1
    assert "LINKAGE_UNSUPPORTED_PLATFORM" in result.output
    assert backend.sent == [] and not (tmp_path / "b.json").exists()


def test_malformed_bundle_on_verify_echoes_no_content(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    bundle = tmp_path / "commitments.json"
    bundle.write_text('{"openings": {"dataset": "OPENING-SECRET-GGGG"')
    os.chmod(bundle, 0o600)
    result = _verify(tmp_path, record, bundle)
    assert result.exit_code == 1
    assert "OPENING-SECRET" not in result.output


def test_unreadable_json_error_chain_holds_no_file_text(tmp_path: Path) -> None:
    bad = tmp_path / "x.json"
    bad.write_text('{"a": "OPENING-SECRET-HHHH"')
    with pytest.raises(Exception) as err:
        cc._read_json(bad, "thing")
    assert err.value.__cause__ is None and err.value.__context__ is None
    assert "OPENING-SECRET" not in repr(err.value) + str(err.value)


def test_invalid_signature_with_no_local_inputs_is_invalid_not_unavailable(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    sidecar = _sidecar(record)
    sidecar["signature"]["signature"] = base64.b64encode(b"\x00" * 64).decode()
    result = _verify(
        tmp_path,
        record,
        _bundle_file(tmp_path, sidecar, fp=None, openings=False),
    )
    assert result.exit_code == 1
    assert "COMMITMENT_LINKAGE_INVALID" in result.output
    assert "UNAVAILABLE" not in result.output


def test_foreign_record_sidecar_with_no_local_inputs_is_invalid(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    sidecar = _sidecar(record)
    other = copy.deepcopy(record)
    other["signature"]["unsigned_manifest_digest"] = "sha256:" + "11" * 32
    result = _verify(
        tmp_path, other, _bundle_file(tmp_path, sidecar, fp=None, openings=False)
    )
    assert result.exit_code == 1
    assert "COMMITMENT_LINKAGE_INVALID RECORD_BINDING_MISMATCH" in result.output
    assert "UNAVAILABLE" not in result.output


def test_arbitrary_sidecar_dict_is_not_reported_unavailable(
    tmp_path: Path, stub_record_verification: None
) -> None:
    record = _record()
    result = _verify(
        tmp_path,
        record,
        _bundle_file(tmp_path, {"anything": 1}, fp=None, openings=False),
    )
    assert result.exit_code == 1
    assert "COMMITMENT_LINKAGE_INVALID" in result.output


def test_failed_record_verification_attempts_no_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = cc._load_certification_schema()
    patched = SimpleNamespace(**vars(real))

    def failing(*_a: Any, **_k: Any) -> Any:
        raise real.ProcessRecordVerificationError("SIGNATURE_INVALID")

    patched.verify_process_record_certificate = failing
    monkeypatch.setattr(cc, "_load_certification_schema", lambda: patched)
    called: list[str] = []
    monkeypatch.setattr(
        cl, "verify_and_open_commitment", lambda *a, **k: called.append("open")
    )
    monkeypatch.setattr(
        cl, "verify_commitment_linkage", lambda *a, **k: called.append("verify")
    )
    record = _record()
    result = _verify(
        tmp_path, record, _bundle_file(tmp_path, _sidecar(record), fp=FP_BOTH)
    )
    assert result.exit_code != 0
    assert called == []
    assert "COMMITMENT_OPENED" not in result.output


def test_open_commitments_requires_process_record_kind(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        certify,
        [
            "verify",
            str(FIXTURES / "bundle.json"),
            "--context",
            str(FIXTURES / "context.json"),
            "--anchor",
            str(FIXTURES / "anchor.json"),
            "--kind",
            "agent-quality",
            "--process-record",
            str(FIXTURES / "bundle.json"),
            "--open-commitments",
            str(_bundle_file(tmp_path, {}, fp=None)),
        ],
    )
    assert result.exit_code == 2
    assert "--open-commitments requires --kind process-record" in result.output


# --- certify finalize -------------------------------------------------------------------


class _FakeBackend:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.sent: list[dict[str, Any]] = []

    def finalize_process_record_sync(
        self, sid: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.sent.append(json.loads(json.dumps(payload)))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def get_process_record_sync(self, sid: str) -> dict[str, Any]:
        raise AssertionError("not expected")


def _finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend, *args: str
):
    monkeypatch.setattr(cc, "_make_backend_client", lambda: backend)
    request = tmp_path / "req.json"
    request.write_text(json.dumps({"project_ref": "p"}))
    return CliRunner().invoke(
        certify,
        [
            "finalize",
            SESSION,
            "--request",
            str(request),
            "--bundle",
            str(tmp_path / "b.json"),
            *args,
        ],
    )


def test_finalize_default_off_requests_no_linkage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend({"data": {"r": 1}})
    result = _finalize(tmp_path, monkeypatch, backend)
    assert result.exit_code == 0, result.output
    assert "openings" not in backend.sent[0]
    assert "commitment_linkage=absent" in result.output


def test_finalize_link_commitments_writes_0600_bundle_and_never_prints_openings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend({"data": {"r": 1}, "commitment_linkage": {"l": 1}})
    result = _finalize(
        tmp_path,
        monkeypatch,
        backend,
        "--link-commitments",
        "--fp-digest",
        f"dataset={FP_DATASET}",
    )
    assert result.exit_code == 0, result.output
    bundle = tmp_path / "b.json"
    assert stat.S_IMODE(os.stat(bundle).st_mode) == 0o600
    stored = json.loads(bundle.read_text())
    assert stored["fp_digests"] == {"dataset": FP_DATASET}
    for value in stored["openings"].values():
        assert value not in result.output
    assert set(backend.sent[0]["openings"]) == {"dataset", "config_space"}


def test_finalize_backend_refusal_is_a_closed_code_with_no_echo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(
        ProcessRecordRequestError(400, "COMMITMENT_LINKAGE_DISABLED")
    )
    result = _finalize(tmp_path, monkeypatch, backend, "--link-commitments")
    assert result.exit_code == 1
    assert "FINALIZE_REFUSED COMMITMENT_LINKAGE_DISABLED" in result.output
    stored = json.loads(
        (tmp_path / "b.json").read_text()
    )  # bundle survived the failure
    for value in stored["openings"].values():
        assert value not in result.output
