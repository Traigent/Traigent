"""Negative cases for the envelope, trust anchor and nominal types."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from traigent.labels._codec import LabelCryptoError, b64u, unb64u
from traigent.labels.envelope import RevisionGuard, open_label, pad, seal_label, unpad
from traigent.labels.hpke import x25519_raw_public
from traigent.labels.trust_anchor import (
    TrustAnchor,
    TrustAnchorError,
    load_trust_anchor,
)
from traigent.labels.types import (
    CodeIdentifier,
    ContentClassError,
    Digest,
    EncryptedBlob,
    KeyMaterial,
    OpaqueRef,
)

REFS = {
    "tenant_ref": "tenant_7c0b2f6e5d1a4f3e9a410c2d8e6b1f01",
    "project_ref": "project_3a9e4d108b274c65b1f25e7a90c4d202",
}
OBJ = "agentref_" + "ab" * 16


def _env(
    sk: X25519PrivateKey, text: str = "name", revision: int = 1, epoch: int = 1
) -> dict:
    return seal_label(
        x25519_raw_public(sk),
        text,
        purpose="label",
        object_ref=OBJ,
        key_epoch=epoch,
        revision=revision,
        **REFS,
    )


def _open(sk: X25519PrivateKey, env: dict, revision: int = 1, **over):
    args = dict(purpose="label", object_ref=OBJ, revision=revision, **REFS)
    args.update(over)
    return open_label(sk, env, **args)


def test_round_trip_and_tamper() -> None:
    sk = X25519PrivateKey.generate()
    env = _env(sk)
    assert _open(sk, env) == "name"
    ct = bytearray(unb64u(env["ciphertext"]))
    ct[0] ^= 1
    with pytest.raises(LabelCryptoError):
        _open(sk, {**env, "ciphertext": b64u(bytes(ct))})
    with pytest.raises(LabelCryptoError):
        _open(X25519PrivateKey.generate(), env)


def test_wrong_epoch_purpose_object_and_scope_are_rejected() -> None:
    sk = X25519PrivateKey.generate()
    env = _env(sk)
    with pytest.raises(LabelCryptoError):
        _open(sk, {**env, "key_epoch": 2})
    with pytest.raises(LabelCryptoError):
        _open(sk, env, purpose="metric")
    with pytest.raises(LabelCryptoError):
        _open(sk, env, object_ref="knobref_" + "00" * 16)
    with pytest.raises(LabelCryptoError):
        _open(sk, env, project_ref="project_" + "00" * 16)
    with pytest.raises(LabelCryptoError):
        seal_label(
            x25519_raw_public(sk),
            "x",
            purpose="free_text",
            object_ref=OBJ,
            key_epoch=1,
            revision=1,
            **REFS,
        )


def test_bad_padding_and_oversize() -> None:
    with pytest.raises(LabelCryptoError):
        pad("x" * 256)
    with pytest.raises(LabelCryptoError):
        unpad(b"abc\x00" * 8 + b"\x00")  # not a bucket length
    with pytest.raises(LabelCryptoError):
        unpad(b"abc" + b"\x00" * 29)  # no 0x80 marker
    with pytest.raises(LabelCryptoError):
        unpad(b"a" * 5 + b"\x80" + b"\x00" * 58)  # non-minimal bucket
    with pytest.raises(LabelCryptoError):
        seal_label(
            x25519_raw_public(X25519PrivateKey.generate()),
            "x" * 256,
            purpose="label",
            object_ref=OBJ,
            key_epoch=1,
            revision=1,
            **REFS,
        )
    assert len(pad("é" * 100)) == 256


def test_stale_revision_is_rejected() -> None:
    guard = RevisionGuard()
    guard.accept(OBJ, 3)
    guard.accept(OBJ, 3)
    guard.accept(OBJ, 4)
    with pytest.raises(LabelCryptoError, match="stale"):
        guard.accept(OBJ, 3)
    with pytest.raises(LabelCryptoError, match="stale"):
        guard.check(OBJ, 1)
    guard.check("agentref_" + "cd" * 16, 1)  # other objects are independent
    sk = X25519PrivateKey.generate()
    env = _env(sk, revision=2)
    with pytest.raises(LabelCryptoError):  # replay under the older revision
        _open(sk, env, revision=1)


def test_error_messages_never_echo_content() -> None:
    sk = X25519PrivateKey.generate()
    env = _env(sk, text="Alice_has_diabetes")
    with pytest.raises(LabelCryptoError) as err:
        _open(sk, env, revision=9)
    assert "Alice" not in str(err.value) and env["ciphertext"] not in str(err.value)


# --- trust anchor ---------------------------------------------------------------------


def test_trust_anchor_loads_from_config(tmp_path: Path) -> None:
    cfg = tmp_path / "traigent.yml"
    cfg.write_text(
        "labels:\n  trust_anchor:\n"
        f'    root_fingerprint: "{"ab" * 32}"\n'
        f"    tenant_ref: {REFS['tenant_ref']}\n    project_ref: {REFS['project_ref']}\n"
    )
    assert load_trust_anchor(cfg) == TrustAnchor("ab" * 32, **REFS)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "labels: {}\n",
        "labels:\n  trust_anchor:\n    root_fingerprint: abc\n",
        "labels:\n  trust_anchor:\n    root_fingerprint: 'zz'\n    tenant_ref: t\n    project_ref: p\n",
    ],
)
def test_trust_anchor_missing_or_malformed_fails_closed(
    tmp_path: Path, text: str
) -> None:
    cfg = tmp_path / ".traigent.yml"
    cfg.write_text(text)
    with pytest.raises(TrustAnchorError):
        load_trust_anchor(cfg)


def test_no_config_found_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from traigent.labels import trust_anchor as ta

    monkeypatch.setattr(ta, "find_config", lambda start=None: None)
    with pytest.raises(TrustAnchorError):
        load_trust_anchor()


# --- nominal types ----------------------------------------------------------------------


def test_opaque_ref_digest_key_material() -> None:
    assert OpaqueRef(OBJ) == OBJ and OpaqueRef(REFS["tenant_ref"])
    assert OpaqueRef("00000000-0000-4000-8000-000000000001")
    for bad in ("Alice_has_diabetes", "knobref_xyz", "knobref_" + "AB" * 16, ""):
        with pytest.raises(ContentClassError):
            OpaqueRef(bad)
    assert Digest("sha256:" + "a" * 64) and Digest("b" * 64)
    with pytest.raises(ContentClassError):
        Digest("sha256:" + "a" * 63)
    assert KeyMaterial(b64u(b"\x01" * 32))
    for bad in ("has space", "a=", "A" * 513, ""):
        with pytest.raises(ContentClassError):
            KeyMaterial(bad)
    assert "Alice" not in repr(OpaqueRef(OBJ))


def test_code_identifier_is_verified_against_local_code(tmp_path: Path) -> None:
    ident = CodeIdentifier.verified("traigent.labels.envelope:pad")
    assert ident == "traigent.labels.envelope:pad"
    assert CodeIdentifier.of(pad) == ident
    assert CodeIdentifier.verified("traigent.labels.envelope:RevisionGuard.check")
    for bad in (
        "traigent.labels.envelope:nope",
        "traigent.no_such_module:pad",
        "not an identifier!",
        "Alice has diabetes",
    ):
        with pytest.raises(ContentClassError):
            CodeIdentifier.verified(bad)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("x = 1\n")
    assert CodeIdentifier.verified("pkg/m.py", root=tmp_path)
    for bad in ("pkg/missing.py", "../outside.py"):
        with pytest.raises(ContentClassError):
            CodeIdentifier.verified(bad, root=tmp_path)
    with pytest.raises(ContentClassError):
        CodeIdentifier.of(lambda: 1)


def test_encrypted_blob_accepts_only_envelopes() -> None:
    sk = X25519PrivateKey.generate()
    env = _env(sk)
    assert EncryptedBlob(env)["key_epoch"] == 1
    for bad in (
        {**env, "plaintext": "x"},
        {**env, "version": 2},
        {**env, "ciphertext": b64u(b"x" * 10)},
    ):
        with pytest.raises(ContentClassError):
            EncryptedBlob(bad)
    assert env["ciphertext"] not in repr(EncryptedBlob(env))
