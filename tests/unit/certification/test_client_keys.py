"""Client signing key: generation, storage, enrolment, pinning."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from traigent_schema.certification import derive_client_key_ref

from tests.unit.certification.fakes import PROJECT_REF, FakeG1Backend
from traigent.certification import client_keys as ck
from traigent.certification.transport import G1RequestError


def test_generate_stores_0600_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "k.json"
    key = ck.generate_client_key(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert [p.name for p in path.parent.iterdir()] == ["k.json"]
    with pytest.raises(ck.ClientKeyError) as caught:
        ck.generate_client_key(path)
    assert caught.value.code == "KEY_EXISTS"
    assert ck.load_client_key(path).public_key_der_b64 == key.public_key_der_b64


def test_load_refuses_loose_permissions_and_garbage(tmp_path: Path) -> None:
    key = ck.generate_client_key(tmp_path / "k.json")
    os.chmod(key.path, 0o644)
    with pytest.raises(ck.ClientKeyError) as caught:
        ck.load_client_key(key.path)
    assert caught.value.code == "KEY_FILE_UNSAFE"
    os.chmod(key.path, 0o600)
    key.path.write_text("not json")
    with pytest.raises(ck.ClientKeyError) as caught:
        ck.load_client_key(key.path)
    assert caught.value.code == "KEY_FILE_INVALID"


def test_signature_verifies_with_public_key(tmp_path: Path) -> None:
    import base64

    key = ck.generate_client_key(tmp_path / "k.json")
    signature = base64.b64decode(key.sign(b"message"))
    assert len(signature) == 64
    key.public_key.verify(signature, b"message")


def test_enroll_sends_only_public_key_and_pins_derived_ref(tmp_path: Path) -> None:
    key = ck.generate_client_key(tmp_path / "k.json")
    backend = FakeG1Backend()
    ref = ck.enroll_client_key(backend, key, PROJECT_REF)
    assert ref == derive_client_key_ref(PROJECT_REF, "ed25519", key.public_key)
    ((method, path, body),) = backend.requests
    assert (method, path) == ("POST", ck.CLIENT_KEYS_PATH)
    assert body == {
        "algorithm": "ed25519",
        "public_key_der_b64": key.public_key_der_b64,
    }
    private_b64 = json.loads(key.path.read_text())["private_key_pkcs8_b64"]
    assert private_b64 not in json.dumps(backend.requests)
    reloaded = ck.load_client_key(key.path)
    assert reloaded.pinned_ref(PROJECT_REF) == ref


def test_enroll_rejects_backend_ref_that_differs_from_derived(tmp_path: Path) -> None:
    key = ck.generate_client_key(tmp_path / "k.json")
    backend = FakeG1Backend()
    backend.enroll_ref_override = "ckr:" + "A" * 43
    with pytest.raises(ck.ClientKeyError) as caught:
        ck.enroll_client_key(backend, key, PROJECT_REF)
    assert caught.value.code == "KEY_REF_MISMATCH"
    assert ck.load_client_key(key.path).pins == {}


def test_enroll_refusal_is_closed(tmp_path: Path) -> None:
    key = ck.generate_client_key(tmp_path / "k.json")
    backend = FakeG1Backend()
    backend.fail_with = (403, "authorization_denied")
    with pytest.raises(ck.ClientKeyError) as caught:
        ck.enroll_client_key(backend, key, PROJECT_REF)
    assert caught.value.code == "ENROLL_REFUSED"
    assert isinstance(caught.value.__cause__, G1RequestError) is False


def test_unpinned_project_raises(tmp_path: Path) -> None:
    key = ck.generate_client_key(tmp_path / "k.json")
    with pytest.raises(ck.ClientKeyError) as caught:
        key.pinned_ref(PROJECT_REF)
    assert caught.value.code == "KEY_NOT_PINNED"


def test_pin_with_wrong_ref_refused(tmp_path: Path) -> None:
    key = ck.generate_client_key(tmp_path / "k.json")
    other = ed25519.Ed25519PrivateKey.generate().public_key()
    wrong = derive_client_key_ref(PROJECT_REF, "ed25519", other)
    with pytest.raises(ck.ClientKeyError) as caught:
        key.pin(PROJECT_REF, wrong)
    assert caught.value.code == "KEY_REF_MISMATCH"


def test_tampered_pin_in_key_file_is_not_trusted(tmp_path: Path) -> None:
    key = ck.generate_client_key(tmp_path / "k.json")
    ck.enroll_client_key(FakeG1Backend(), key, PROJECT_REF)
    document = json.loads(key.path.read_text())
    document["pins"][PROJECT_REF] = "ckr:" + "B" * 43
    key.path.write_text(json.dumps(document))
    with pytest.raises(ck.ClientKeyError) as caught:
        ck.load_client_key(key.path).pinned_ref(PROJECT_REF)
    assert caught.value.code == "KEY_NOT_PINNED"


def test_ecdsa_key_file_is_refused_closed(tmp_path: Path) -> None:
    key = ck.generate_client_key(tmp_path / "k.json")
    document = json.loads(key.path.read_text())
    document["algorithm"] = "ecdsa_p256_sha256"
    key.path.write_text(json.dumps(document))
    with pytest.raises(ck.ClientKeyError) as caught:
        ck.load_client_key(key.path)
    assert caught.value.code == "KEY_FILE_INVALID"
