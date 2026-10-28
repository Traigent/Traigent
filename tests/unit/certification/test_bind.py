"""Bind flow: issue -> link -> bind -> prepare -> co-sign -> finalize, against a recording fake."""

from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from traigent_schema.certification import (
    RelyingPartyPolicy,
    VerificationContext,
    verify_certificate,
)

from tests.unit.certification.fakes import FakeG1Backend
from traigent.certification import bind as bind_module
from traigent.certification import client_keys as ck
from traigent.certification import pins as pins_module
from traigent.certification.client_evidence import (
    SLOT_KINDS,
    ClientEvidence,
    manifest_root_digest,
    slot_commitment_digest,
)
from traigent.certification.transport import G1RequestError

_FIXTURE: dict[str, Any] = json.loads(
    (
        Path(__file__).parents[2]
        / "fixtures"
        / "certification"
        / "g1_cosign_fixture.json"
    ).read_text()
)


def _evidence() -> ClientEvidence:
    documents = _FIXTURE["evidence"]["documents"]
    blinds = {
        k: bytes.fromhex(v) for k, v in _FIXTURE["evidence"]["blinds_hex"].items()
    }
    commitments = {
        kind: slot_commitment_digest(blinds[kind], kind, documents[kind])
        for kind in SLOT_KINDS
    }
    root = manifest_root_digest(commitments)
    assert root == _FIXTURE["root"]
    return ClientEvidence(
        documents,
        blinds,
        commitments,
        root,
        documents["build_process_evidence"]["manifest_digest"],
    )


def _private_from_seed() -> ed25519.Ed25519PrivateKey:
    return ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(_FIXTURE["client_seed_hex"])
    )


@pytest.fixture
def key(tmp_path: Path) -> ck.ClientKey:
    key = ck.ClientKey(tmp_path / "k.json", _private_from_seed())
    key.pin(
        _FIXTURE["pins"]["project_ref"], key.derive_ref(_FIXTURE["pins"]["project_ref"])
    )
    return key


@pytest.fixture
def pins(tmp_path: Path) -> pins_module.Pins:
    source = tmp_path / "pins-source.json"
    source.write_text(json.dumps(_FIXTURE["pins"]))
    return pins_module.write_pins(source, tmp_path / "pins" / "pins.json")


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeG1Backend:
    fake = FakeG1Backend()
    fake.fixture = _FIXTURE
    monkeypatch.setattr(
        bind_module.secrets, "token_hex", lambda n=16: _FIXTURE["session"]["nonce"]
    )
    return fake


def _bind(
    backend: FakeG1Backend, key: ck.ClientKey, pins: pins_module.Pins
) -> bind_module.BindResult:
    session = _FIXTURE["session"]
    return bind_module.bind_g1(
        backend,
        _evidence(),
        key,
        pins,
        run_id="run_1",
        sdk_ref=session["sdk_ref"],
        sdk_version=session["sdk_version"],
        declared_mode=session["declared_mode"],
    )


def test_happy_path_cosigns_and_schema_verifier_accepts(
    backend: FakeG1Backend, key: ck.ClientKey, pins: pins_module.Pins
) -> None:
    result = _bind(backend, key, pins)
    assert result.build_session_ref == _FIXTURE["session"]["build_session_ref"]
    assert [(m, p.rsplit("/", 1)[-1]) for m, p, _ in backend.requests] == [
        ("POST", "certificate-build-sessions"),
        ("POST", "link-run"),
        ("POST", "bind-g1"),
        ("POST", "prepare"),
        ("POST", "finalize"),
    ]
    assert backend.finalized is not None
    assert backend.finalized["nonce"] == _FIXTURE["session"]["nonce"]
    assert backend.finalized["build_session_ref"] == result.build_session_ref
    co = backend.finalized["co_attestation"]
    # ed25519 is deterministic: the produced co-attestation is the fixture's.
    expected = _FIXTURE["certificate"]["signatures"]["co_attestation"]
    assert co == expected
    certificate = copy.deepcopy(_FIXTURE["prepare_response"])
    certificate["signatures"]["co_attestation"] = co
    session = _FIXTURE["session"]
    issuer = certificate["signatures"]["issuer_signature"]
    context = VerificationContext(
        expected_nonce=session["nonce"],
        expected_build_session_ref=session["build_session_ref"],
        expected_issuer_key_ref=issuer["issuer_key_ref"],
        expected_issuer_algorithm=issuer["algorithm"],
        expected_trust_ring_ref=issuer["trust_ring_ref"],
        expected_client_key_ref=co["client_key_ref"],
        expected_client_algorithm="ed25519",
        expected_project_ref=_FIXTURE["pins"]["project_ref"],
        client_public_key=key.public_key,
    )
    policy = RelyingPartyPolicy(
        pins.compiler_register_versions,
        (
            ("B1", "ver.cert.seal_signature", "0.1.0"),
            ("G1", "ver.cert.manifest_commitment", "0.1.0"),
        ),
    )
    verdict = verify_certificate(
        certificate,
        issuer_public_key=pins.issuer_public_key,
        context=context,
        policy=policy,
    )
    assert verdict.valid


def test_requests_carry_only_catalog_fields_and_no_secret(
    backend: FakeG1Backend, key: ck.ClientKey, pins: pins_module.Pins
) -> None:
    _bind(backend, key, pins)
    wire = json.dumps(backend.requests)
    evidence = _evidence()
    for blind in evidence.blinds.values():
        assert blind.hex() not in wire
    for kind in SLOT_KINDS:
        assert evidence.commitments[kind] not in wire
        assert json.dumps(evidence.documents[kind]) not in wire
    assert evidence.snapshot_manifest_digest not in wire
    private_b64 = json.loads(key.path.read_text())["private_key_pkcs8_b64"]
    assert private_b64 not in wire


@pytest.mark.parametrize(
    ("step", "code"),
    [
        ("issue", "SESSION_REFUSED"),
        ("link-run", "LINK_REFUSED"),
        ("bind-g1", "BIND_REFUSED"),
        ("prepare", "PREPARE_REFUSED"),
        ("finalize", "FINALIZE_REFUSED"),
    ],
)
def test_each_refused_step_maps_to_a_closed_code(
    backend: FakeG1Backend,
    key: ck.ClientKey,
    pins: pins_module.Pins,
    step: str,
    code: str,
) -> None:
    backend.refuse_step = step
    with pytest.raises(bind_module.BindError) as caught:
        _bind(backend, key, pins)
    assert caught.value.code == code
    assert (
        caught.value.__cause__ is None
        or isinstance(caught.value.__cause__, G1RequestError) is False
    )
    if step == "issue":
        assert caught.value.session_ref is None
    else:
        assert caught.value.session_ref == _FIXTURE["session"]["build_session_ref"]
    if step != "finalize":
        assert backend.finalized is None


def test_wrong_root_in_bind_response_refuses_before_prepare(
    backend: FakeG1Backend, key: ck.ClientKey, pins: pins_module.Pins
) -> None:
    backend.bind_root_override = "sha256:" + "9" * 64
    with pytest.raises(bind_module.BindError) as caught:
        _bind(backend, key, pins)
    assert caught.value.code == "BIND_RESPONSE_INVALID"
    assert not any(path.endswith("/prepare") for _, path, _ in backend.requests)


def test_prepare_that_differs_from_pins_is_never_signed(
    backend: FakeG1Backend, key: ck.ClientKey, tmp_path: Path
) -> None:
    other = dict(_FIXTURE["pins"])
    other["trust_ring_ref"] = "ring:" + "Z" * 20
    source = tmp_path / "other.json"
    source.write_text(json.dumps(other))
    pins = pins_module.write_pins(source, tmp_path / "p2" / "pins.json")
    with pytest.raises(bind_module.BindError) as caught:
        _bind(backend, key, pins)
    assert caught.value.code == "CO_CONTEXT"
    assert backend.finalized is None
    assert not any(path.endswith("/finalize") for _, path, _ in backend.requests)


def test_unpinned_client_key_refuses_before_any_request(
    backend: FakeG1Backend, tmp_path: Path, pins: pins_module.Pins
) -> None:
    unpinned = ck.ClientKey(tmp_path / "u.json", _private_from_seed())
    with pytest.raises(bind_module.BindError) as caught:
        _bind(backend, unpinned, pins)
    assert caught.value.code == "KEY_NOT_PINNED"
    assert backend.requests == []


def test_abandon_posts_no_body(backend: FakeG1Backend) -> None:
    bind_module.abandon_session(backend, _FIXTURE["session"]["build_session_ref"])
    ((method, path, body),) = backend.requests
    assert (method, path.rsplit("/", 1)[-1], body) == ("POST", "abandon", None)
    backend.refuse_step = "abandon"
    with pytest.raises(bind_module.BindError) as caught:
        bind_module.abandon_session(backend, _FIXTURE["session"]["build_session_ref"])
    assert caught.value.code == "ABANDON_REFUSED"


@pytest.mark.parametrize(
    ("state", "ok"), [("available", True), ("unavailable", False), ("absent", False)]
)
def test_readiness_probe(state: str, ok: bool) -> None:
    fake = FakeG1Backend()
    fake.readiness = state
    if ok:
        bind_module.check_g1_readiness(fake)
        return
    with pytest.raises(bind_module.BindError) as caught:
        bind_module.check_g1_readiness(fake)
    assert caught.value.code == "READINESS_UNAVAILABLE"


def test_readiness_wrong_version_or_shape_is_unavailable() -> None:
    class Odd:
        def __init__(self, body: Any) -> None:
            self.body = body

        def request(self, method: str, path: str, body: Any = None) -> Any:
            from traigent.certification.transport import G1Response

            return G1Response(200, self.body)

    for body in (
        {"readiness_version": 2, "g1_issuance": "available"},
        {"g1_issuance": "available"},
        ["available"],
        None,
    ):
        with pytest.raises(bind_module.BindError):
            bind_module.check_g1_readiness(Odd(body))  # type: ignore[arg-type]


def test_sdk_identity_requires_a_git_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIGENT_SDK_REF", raising=False)
    monkeypatch.setattr(
        bind_module.importlib.metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(
            bind_module.importlib.metadata.PackageNotFoundError()
        ),
    )
    with pytest.raises(bind_module.BindError) as caught:
        bind_module.resolve_sdk_identity()
    assert caught.value.code == "SDK_IDENTITY_UNAVAILABLE"
    ref, version = bind_module.resolve_sdk_identity("a" * 40)
    assert ref == "a" * 40
    assert version.count(".") == 2
    with pytest.raises(bind_module.BindError):
        bind_module.resolve_sdk_identity("not-a-sha")


def test_pins_file_modes_and_validation(tmp_path: Path) -> None:
    source = tmp_path / "s.json"
    source.write_text(json.dumps(_FIXTURE["pins"]))
    destination = tmp_path / "d" / "pins.json"
    pins_module.write_pins(source, destination)
    assert stat.S_IMODE(os.stat(destination).st_mode) == 0o600
    assert (
        pins_module.load_pins(destination).project_ref
        == _FIXTURE["pins"]["project_ref"]
    )
    os.chmod(destination, 0o644)
    with pytest.raises(pins_module.PinError) as caught:
        pins_module.load_pins(destination)
    assert caught.value.code == "PIN_FILE_UNSAFE"
    with pytest.raises(pins_module.PinError) as caught:
        pins_module.load_pins(tmp_path / "missing.json")
    assert caught.value.code == "PIN_MISSING"
    bad = dict(_FIXTURE["pins"])
    del bad["trust_ring_ref"]
    source.write_text(json.dumps(bad))
    with pytest.raises(pins_module.PinError) as caught:
        pins_module.write_pins(source, tmp_path / "e" / "pins.json")
    assert caught.value.code == "PIN_INVALID"
    ecdsa = dict(_FIXTURE["pins"], issuer_algorithm="ecdsa_p256_sha256")
    source.write_text(json.dumps(ecdsa))
    with pytest.raises(pins_module.PinError):
        pins_module.write_pins(source, tmp_path / "f" / "pins.json")
    assert not (tmp_path / "e").exists() or not (tmp_path / "e" / "pins.json").exists()
