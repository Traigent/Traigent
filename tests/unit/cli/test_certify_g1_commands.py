"""``traigent certify`` G1 commands: keygen, pin, enroll, bind, abandon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from tests.unit.certification.fakes import FakeG1Backend
from tests.unit.snapshot import helpers
from traigent.certification import bind as bind_module
from traigent.certification import client_evidence as ce
from traigent.certification.bind import BindError, BindResult
from traigent.certification.client_keys import load_client_key
from traigent.certification.pins import load_pins
from traigent.cli import certify_commands
from traigent.cli.certify_commands import certify
from traigent.snapshot import (
    EvaluatorInputs,
    build_execution_snapshot,
    publish_snapshot,
)
from traigent.snapshot.backends import LocalDirStore

_FIXTURE: dict[str, Any] = json.loads(
    (
        Path(__file__).parents[1] / "certification" / "data" / "g1_cosign_fixture.json"
    ).read_text()
)
_PROJECT_REF = _FIXTURE["pins"]["project_ref"]
_SDK_REF = "ab" * 20


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeG1Backend:
    fake = FakeG1Backend(project_ref=_PROJECT_REF)
    monkeypatch.setattr(certify_commands, "_g1_transport", lambda: fake)
    return fake


@pytest.fixture
def pins_file(tmp_path: Path) -> Path:
    source = tmp_path / "pins-source.json"
    source.write_text(json.dumps(_FIXTURE["pins"]))
    target = tmp_path / "pins" / "pins.json"
    result = CliRunner().invoke(
        certify, ["pin", "--from", str(source), "--pins", str(target)]
    )
    assert result.exit_code == 0, result.output
    return target


@pytest.fixture
def evidence_setup(tmp_path: Path) -> dict[str, Any]:
    store = LocalDirStore(tmp_path / "store")
    snapshot = build_execution_snapshot(
        dataset=[{"input": {"q": "a"}, "expected": "b"}],
        config={"temperature": 0.2},
        config_space={"model": ["x", "y"]},
        evaluator=EvaluatorInputs(metric_functions={"a": helpers.metric_a}),
    )
    publish_snapshot(store, snapshot)
    custody = ce.LocalFileCustody(tmp_path / "evidence")
    evidence = ce.build_client_evidence(snapshot, custody=custody)
    return {
        "store_dir": tmp_path / "store",
        "evidence_dir": tmp_path / "evidence",
        "evidence": evidence,
        "path": Path(custody.path_for(evidence.root_hex)),
    }


def _keygen(tmp_path: Path) -> Path:
    path = tmp_path / "keys" / "k.json"
    result = CliRunner().invoke(certify, ["keygen", "--out", str(path)])
    assert result.exit_code == 0, result.output
    return path


def _bind_args(setup: dict[str, Any], key: Path, pins: Path) -> list[str]:
    return [
        "bind",
        setup["evidence"].root,
        "--key",
        str(key),
        "--run-id",
        "run_1",
        "--store-dir",
        str(setup["store_dir"]),
        "--pins",
        str(pins),
        "--evidence-dir",
        str(setup["evidence_dir"]),
        "--sdk-ref",
        _SDK_REF,
    ]


def test_keygen_prints_path_and_public_key_only(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "k.json"
    result = CliRunner().invoke(certify, ["keygen", "--out", str(path)])
    assert result.exit_code == 0
    key = load_client_key(path)
    assert str(path) in result.output
    assert key.public_key_der_b64 in result.output
    document = json.loads(path.read_text())
    private = document["private_key_pkcs8_b64"]
    assert private not in result.output
    assert "private_key" not in result.output
    assert "PRIVATE" not in result.output


def test_pin_and_enroll_round_trip(
    tmp_path: Path, backend: FakeG1Backend, pins_file: Path
) -> None:
    assert load_pins(pins_file).project_ref == _PROJECT_REF
    key_path = _keygen(tmp_path)
    result = CliRunner().invoke(
        certify, ["enroll", "--key", str(key_path), "--pins", str(pins_file)]
    )
    assert result.exit_code == 0, result.output
    key = load_client_key(key_path)
    ref = key.derive_ref(_PROJECT_REF)
    assert ref in result.output
    assert key.pinned_ref(_PROJECT_REF) == ref
    assert [(m, p) for m, p, _ in backend.requests] == [
        ("POST", "/api/v1beta/certificate-client-keys")
    ]


def test_pin_invalid_file_is_closed_error(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text("{}")
    result = CliRunner().invoke(
        certify, ["pin", "--from", str(source), "--pins", str(tmp_path / "p.json")]
    )
    assert result.exit_code == 1
    assert "PIN_INVALID" in result.output


def test_missing_pin_file_is_closed_error(
    tmp_path: Path, backend: FakeG1Backend
) -> None:
    key_path = _keygen(tmp_path)
    result = CliRunner().invoke(
        certify,
        ["enroll", "--key", str(key_path), "--pins", str(tmp_path / "nope.json")],
    )
    assert result.exit_code == 1
    assert "PIN_MISSING" in result.output
    assert backend.requests == []


def test_bind_passes_reconstructed_evidence(
    tmp_path: Path,
    backend: FakeG1Backend,
    pins_file: Path,
    evidence_setup: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = _keygen(tmp_path)
    seen: dict[str, Any] = {}

    def stub(
        transport: Any, evidence: Any, key: Any, pins: Any, **kwargs: Any
    ) -> BindResult:
        seen.update(transport=transport, evidence=evidence, pins=pins, kwargs=kwargs)
        return BindResult("bsn:" + "a" * 43, "cert:" + "c" * 43, "n")

    monkeypatch.setattr(bind_module, "bind_g1", stub)
    result = CliRunner().invoke(
        certify,
        _bind_args(evidence_setup, key_path, pins_file)
        + ["--declared-mode", "offline"],
    )
    assert result.exit_code == 0, result.output
    assert "BOUND certificate_ref=cert:" + "c" * 43 in result.output
    assert seen["transport"] is backend
    assert seen["evidence"].root == evidence_setup["evidence"].root
    assert seen["evidence"].documents == evidence_setup["evidence"].documents
    assert seen["kwargs"]["run_id"] == "run_1"
    assert seen["kwargs"]["sdk_ref"] == _SDK_REF
    assert seen["kwargs"]["declared_mode"] == "offline"
    assert seen["pins"].project_ref == _PROJECT_REF


@pytest.mark.parametrize(
    "damage", ["tamper_evidence", "missing_evidence", "missing_store"]
)
def test_bind_reconstruction_failure_makes_no_network_call(
    tmp_path: Path,
    backend: FakeG1Backend,
    pins_file: Path,
    evidence_setup: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    key_path = _keygen(tmp_path)
    called: list[str] = []
    monkeypatch.setattr(bind_module, "bind_g1", lambda *a, **k: called.append("bind"))
    args = _bind_args(evidence_setup, key_path, pins_file)
    if damage == "tamper_evidence":
        path: Path = evidence_setup["path"]
        record = json.loads(path.read_text())
        record["documents"]["build_process_evidence"]["manifest_digest"] = (
            "sha256:" + "0" * 64
        )
        path.write_text(json.dumps(record))
        expected = "EVIDENCE_STALE_OR_CORRUPT"
    elif damage == "missing_evidence":
        evidence_setup["path"].unlink()
        expected = "EVIDENCE_NOT_FOUND"
    else:
        for item in sorted(Path(evidence_setup["store_dir"]).rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
        expected = "EVIDENCE_STALE_OR_CORRUPT"
    result = CliRunner().invoke(certify, args)
    assert result.exit_code == 1
    assert f"BIND_REFUSED {expected}" in result.output
    assert backend.requests == []
    assert called == []


def test_bind_error_prints_code_and_session_ref(
    tmp_path: Path,
    backend: FakeG1Backend,
    pins_file: Path,
    evidence_setup: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = _keygen(tmp_path)

    def refuse(*args: Any, **kwargs: Any) -> BindResult:
        raise BindError("FINALIZE_REFUSED", "bsn:" + "b" * 43)

    monkeypatch.setattr(bind_module, "bind_g1", refuse)
    result = CliRunner().invoke(
        certify, _bind_args(evidence_setup, key_path, pins_file)
    )
    assert result.exit_code == 1
    assert "BIND_REFUSED FINALIZE_REFUSED" in result.output
    assert "session_ref=bsn:" + "b" * 43 in result.output


def test_bind_readiness_unavailable_refuses(
    tmp_path: Path,
    backend: FakeG1Backend,
    pins_file: Path,
    evidence_setup: dict[str, Any],
) -> None:
    key_path = _keygen(tmp_path)
    backend.readiness = "unavailable"
    result = CliRunner().invoke(
        certify, _bind_args(evidence_setup, key_path, pins_file)
    )
    assert result.exit_code == 1
    assert "BIND_REFUSED READINESS_UNAVAILABLE" in result.output
    assert [p for _, p, _ in backend.requests] == [
        "/api/v1beta/certificate-build-sessions/readiness"
    ]


def test_bind_rejects_non_digest_root(tmp_path: Path, backend: FakeG1Backend) -> None:
    result = CliRunner().invoke(
        certify,
        [
            "bind",
            "nothex",
            "--key",
            str(tmp_path / "k"),
            "--run-id",
            "r",
            "--store-dir",
            str(tmp_path / "s"),
        ],
    )
    assert result.exit_code == 1
    assert "BIND_REFUSED EVIDENCE_STALE_OR_CORRUPT" in result.output
    assert backend.requests == []


def test_abandon(backend: FakeG1Backend) -> None:
    backend.fixture = _FIXTURE
    ref = _FIXTURE["session"]["build_session_ref"]
    result = CliRunner().invoke(certify, ["abandon", ref])
    assert result.exit_code == 0, result.output
    assert f"ABANDONED {ref}" in result.output
    backend.refuse_step = "abandon"
    result = CliRunner().invoke(certify, ["abandon", ref])
    assert result.exit_code == 1
    assert "ABANDON_REFUSED" in result.output
