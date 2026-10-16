"""Byte-for-byte conformance with the TraigentSchema label-service vectors."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from traigent.labels import hpke
from traigent.labels._codec import LabelCryptoError, b64u, jcs, unb64u
from traigent.labels.bindings import (
    key_id,
    make_binding,
    root_fingerprint,
    verify_chain,
)
from traigent.labels.envelope import (
    BUCKETS,
    label_aad,
    open_label,
    pack_envelope,
    pad,
    seal_label,
    unpad,
)
from traigent.labels.wrap import member_wrap_info, open_member_wrap, seal_member_wrap
from traigent.labels.writer import LabelWriter
from traigent.labels.trust_anchor import TrustAnchor

from .conftest import FIXTURES

SCHEMA_WT = Path(
    "/home/nimrod/TraigentProjects/core_project/worktrees/TraigentSchema/content-less"
)
BUNDLE = (
    Path(__file__).resolve().parents[3] / "traigent" / "labels" / "contract_schemas"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _x(hex_: str) -> X25519PrivateKey:
    return X25519PrivateKey.from_private_bytes(bytes.fromhex(hex_))


# --- pins against the Schema worktree ------------------------------------------------


@pytest.mark.skipif(not SCHEMA_WT.exists(), reason="TraigentSchema worktree absent")
def test_fixtures_are_pinned_to_the_schema_worktree() -> None:
    pairs = {
        "label_vectors_v0.json": SCHEMA_WT
        / "traigent_schema/data/label_vectors_v0.json",
        "rfc9180_x25519_base_vectors.json": SCHEMA_WT
        / "tests/data/rfc9180_x25519_base_vectors.json",
    }
    for name, source in pairs.items():
        assert _sha(FIXTURES / name) == _sha(source), name


@pytest.mark.skipif(not SCHEMA_WT.exists(), reason="TraigentSchema worktree absent")
def test_bundled_contract_schemas_are_pinned_to_the_schema_worktree() -> None:
    root = SCHEMA_WT / "traigent_schema/schemas"
    files = sorted(BUNDLE.rglob("*.json"))
    assert files
    for f in files:
        assert _sha(f) == _sha(root / f.relative_to(BUNDLE)), f.name


# --- RFC 9180 -----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["A.1", "A.2"])
def test_rfc9180_base_mode(rfc: dict, name: str) -> None:
    r = rfc[name]
    b = {
        k: bytes.fromhex(x)
        for k, x in r.items()
        if isinstance(x, str) and k != "source"
    }
    sk_e, pk_e = hpke.derive_keypair(b["ikmE"])
    assert pk_e == b["enc"]
    assert hpke.derive_keypair(b["ikmR"])[1] == b["pkRm"]
    shared = hpke._extract_and_expand(hpke._dh(sk_e, b["pkRm"]), b["enc"] + b["pkRm"])
    assert shared == b["shared_secret"]
    assert hpke._key_schedule(r["aead_id"], shared, b["info"]) == (
        b["key"],
        b["base_nonce"],
    )
    enc, ct = hpke.hpke_seal(
        b["pkRm"], b["info"], b["aad"], b["pt"], ikm_e=b["ikmE"], aead_id=r["aead_id"]
    )
    assert (enc, ct) == (b["enc"], b["ct"])
    assert (
        hpke.hpke_open(_x(r["skRm"]), enc, b["info"], b["aad"], ct, r["aead_id"])
        == b["pt"]
    )


def test_random_seal_differs_and_opens() -> None:
    sk = X25519PrivateKey.generate()
    pk = hpke.x25519_raw_public(sk)
    a = hpke.hpke_seal(pk, b"i", b"a", b"msg")
    b = hpke.hpke_seal(pk, b"i", b"a", b"msg")
    assert a != b
    assert hpke.hpke_open(sk, a[0], b"i", b"a", a[1]) == b"msg"


# --- envelopes ----------------------------------------------------------------------


def test_envelopes_reproduce_byte_for_byte(vectors: dict) -> None:
    refs, keys = vectors["refs"], vectors["keys"]
    pk = bytes.fromhex(keys["project_public_hex"])
    for entry in vectors["label_envelopes"]:
        assert (
            label_aad(
                refs["tenant_ref"],
                refs["project_ref"],
                entry["purpose"],
                entry["object_ref"],
                entry["envelope"]["key_epoch"],
                entry["revision"],
            ).decode()
            == entry["aad_jcs"]
        )
        env = seal_label(
            pk,
            entry["plaintext"],
            tenant_ref=refs["tenant_ref"],
            project_ref=refs["project_ref"],
            purpose=entry["purpose"],
            object_ref=entry["object_ref"],
            key_epoch=entry["envelope"]["key_epoch"],
            revision=entry["revision"],
            ikm_e=bytes.fromhex(entry["ephemeral_ikm_hex"]),
        )
        assert env == entry["envelope"]
        packed = pack_envelope(
            env["version"],
            env["key_epoch"],
            unb64u(env["enc"]),
            unb64u(env["ciphertext"]),
        )
        assert packed.hex() == entry["packed_hex"]
        assert (
            open_label(
                _x(keys["project_private_hex"]),
                env,
                tenant_ref=refs["tenant_ref"],
                project_ref=refs["project_ref"],
                purpose=entry["purpose"],
                object_ref=entry["object_ref"],
                revision=entry["revision"],
            )
            == entry["plaintext"]
        )


def test_padding_vectors(vectors: dict) -> None:
    for case in vectors["padding"]["ok"]:
        padded = pad(case["plaintext"])
        assert padded.hex() == case["padded_hex"]
        assert len(padded) == case["bucket"] and case["bucket"] in BUCKETS
        assert unpad(padded) == case["plaintext"]


def test_malformed_padding_is_refused(vectors: dict) -> None:
    for case in vectors["padding"]["malformed"]:
        with pytest.raises(LabelCryptoError):
            if case["operation"] == "pad":
                pad(case["plaintext"])
            else:
                unpad(bytes.fromhex(case["padded_hex"]))


# --- member wrap (flow b) -------------------------------------------------------------


def test_member_wrap_reproduces_and_open_is_opt_in(vectors: dict) -> None:
    refs, keys, mw = vectors["refs"], vectors["keys"], vectors["member_wrap"]
    kw = {"tenant_ref": refs["tenant_ref"], "project_ref": refs["project_ref"]}
    assert (
        member_wrap_info(
            refs["tenant_ref"], refs["project_ref"], 1, mw["wrap"]["recipient_key_id"]
        ).decode()
        == mw["info_jcs"]
    )
    wrap = seal_member_wrap(
        bytes.fromhex(keys["project_private_hex"]),
        bytes.fromhex(keys["member_public_hex"]),
        epoch=1,
        ikm_e=bytes.fromhex(mw["ephemeral_ikm_hex"]),
        **kw,
    )
    assert wrap == mw["wrap"]
    with pytest.raises(LabelCryptoError, match="disabled"):
        open_member_wrap(_x(keys["member_private_hex"]), wrap, **kw)
    sk = open_member_wrap(
        _x(keys["member_private_hex"]), wrap, allow_machine_member_open=True, **kw
    )
    assert sk.private_bytes_raw().hex() == keys["project_private_hex"]
    with pytest.raises(LabelCryptoError):
        open_member_wrap(
            _x(keys["project_private_hex"]), wrap, allow_machine_member_open=True, **kw
        )
    with pytest.raises(LabelCryptoError):
        open_member_wrap(
            _x(keys["member_private_hex"]),
            {**wrap, "epoch": 2},
            allow_machine_member_open=True,
            **kw,
        )


# --- bindings -----------------------------------------------------------------------


def _chain(vectors: dict, **tweak):
    chain = vectors["binding_chain"]
    return {
        "pinned_root_fingerprint": tweak.get(
            "pinned", vectors["keys"]["root_fingerprint_sha256"]
        ),
        "root_pub": bytes.fromhex(vectors["keys"]["root_public_hex"]),
        "admin_binding": tweak.get("admin", chain["admin_identity"]),
        "project_binding": tweak.get("project", chain["project_key"]),
        "member_binding": tweak.get("member", chain["member_key"]),
        "tenant_ref": vectors["refs"]["tenant_ref"],
        "project_ref": vectors["refs"]["project_ref"],
        "binding_list": tweak.get("blist", vectors["binding_list"]),
        "retained_sequence": tweak.get("retained", 1),
        "now": tweak.get("now", "2026-10-16T00:00:01Z"),
    }


def _root_sk(vectors: dict) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(vectors["keys"]["root_ed25519_seed_hex"])
    )


def _signed_list(vectors: dict, **fields):
    body = {
        k: x
        for k, x in vectors["binding_list"].items()
        if k not in ("signature", "signer_key_id")
    }
    return make_binding(
        _root_sk(vectors),
        bytes.fromhex(vectors["keys"]["root_public_hex"]),
        {**body, **fields},
    )


def test_fingerprints_and_key_ids(vectors: dict) -> None:
    k = vectors["keys"]
    assert (
        root_fingerprint(bytes.fromhex(k["root_public_hex"]))
        == k["root_fingerprint_sha256"]
    )
    chain = vectors["binding_chain"]
    assert (
        key_id(bytes.fromhex(k["member_public_hex"])) == chain["member_key"]["key_id"]
    )
    assert (
        key_id(bytes.fromhex(k["project_public_hex"])) == chain["project_key"]["key_id"]
    )


def test_signed_bindings_reproduce_byte_for_byte(vectors: dict) -> None:
    root_sk, root_pub = (
        _root_sk(vectors),
        bytes.fromhex(vectors["keys"]["root_public_hex"]),
    )
    admin_sk = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(vectors["keys"]["admin_ed25519_seed_hex"])
    )
    admin_pub = bytes.fromhex(vectors["keys"]["admin_public_hex"])

    def redo(binding: dict, sk, pub) -> dict:
        body = {
            k: x for k, x in binding.items() if k not in ("signature", "signer_key_id")
        }
        return make_binding(sk, pub, body)

    chain = vectors["binding_chain"]
    assert redo(chain["admin_identity"], root_sk, root_pub) == chain["admin_identity"]
    assert redo(chain["project_key"], admin_sk, admin_pub) == chain["project_key"]
    assert redo(chain["member_key"], admin_sk, admin_pub) == chain["member_key"]
    assert redo(vectors["binding_list"], root_sk, root_pub) == vectors["binding_list"]


def test_chain_verifies(vectors: dict) -> None:
    verify_chain(**_chain(vectors))


def test_chain_failures(vectors: dict) -> None:
    chain = vectors["binding_chain"]
    with pytest.raises(LabelCryptoError, match="pinned"):
        verify_chain(**_chain(vectors, pinned="0" * 64))
    with pytest.raises(LabelCryptoError):  # wrong epoch (tamper)
        verify_chain(**_chain(vectors, project={**chain["project_key"], "epoch": 2}))
    swapped = {**chain["member_key"], "public_key": chain["project_key"]["public_key"]}
    with pytest.raises(LabelCryptoError):  # server key substitution
        verify_chain(**_chain(vectors, member=swapped))
    with pytest.raises(LabelCryptoError):  # purposes swapped
        verify_chain(
            **_chain(vectors, project=chain["member_key"], member=chain["project_key"])
        )
    body = {
        k: x
        for k, x in chain["member_key"].items()
        if k not in ("signature", "signer_key_id")
    }
    forged = make_binding(
        _root_sk(vectors), bytes.fromhex(vectors["keys"]["root_public_hex"]), body
    )
    with pytest.raises(LabelCryptoError):  # signed by root instead of the admin
        verify_chain(**_chain(vectors, member=forged))
    args = _chain(vectors)
    args["tenant_ref"] = "tenant_" + "0" * 32
    with pytest.raises(LabelCryptoError):
        verify_chain(**args)


def test_revoked_signer_project_and_member_are_rejected(vectors: dict) -> None:
    chain = vectors["binding_chain"]
    for name in ("admin_identity", "project_key", "member_key"):
        revoked = _signed_list(
            vectors, revoked_key_ids=[chain[name]["key_id"]], sequence=2
        )
        with pytest.raises(LabelCryptoError, match="revoked"):
            verify_chain(**_chain(vectors, blist=revoked, retained=2))


def test_future_not_before_is_rejected(vectors: dict) -> None:
    chain = vectors["binding_chain"]
    body = {
        k: x
        for k, x in chain["admin_identity"].items()
        if k not in ("signature", "signer_key_id")
    }
    future = make_binding(
        _root_sk(vectors),
        bytes.fromhex(vectors["keys"]["root_public_hex"]),
        {**body, "not_before": "2099-01-01T00:00:00Z"},
    )
    with pytest.raises(LabelCryptoError, match="not yet active"):
        verify_chain(**_chain(vectors, admin=future))
    with pytest.raises(LabelCryptoError, match="not yet active"):
        verify_chain(**_chain(vectors, now="2026-10-15T23:59:59Z"))


def test_list_rollback_and_forgery_are_rejected(vectors: dict) -> None:
    with pytest.raises(LabelCryptoError, match="sequence"):
        verify_chain(**_chain(vectors, retained=2))
    with pytest.raises(LabelCryptoError):
        verify_chain(
            **_chain(
                vectors,
                blist={**vectors["binding_list"], "revoked_key_ids": ["A" * 22]},
            )
        )
    with pytest.raises(LabelCryptoError, match="tenant"):
        verify_chain(
            **_chain(
                vectors, blist=_signed_list(vectors, tenant_ref="tenant_" + "0" * 32)
            )
        )


def test_jcs_subset() -> None:
    assert jcs({"b": 1, "a": [True, None, "x"], "c": {"z": 0, "y": "é"}}) == (
        '{"a":[true,null,"x"],"b":1,"c":{"y":"é","z":0}}'.encode()
    )
    with pytest.raises(LabelCryptoError):
        jcs({"a": 1.5})
    assert b64u(b"\xfb\xff") == "-_8"


# --- writer: verify-before-encrypt --------------------------------------------------------


def _writer(vectors: dict, **tweak) -> LabelWriter:
    chain = vectors["binding_chain"]
    anchor = TrustAnchor(
        tweak.get("pinned", vectors["keys"]["root_fingerprint_sha256"]),
        vectors["refs"]["tenant_ref"],
        vectors["refs"]["project_ref"],
    )
    return LabelWriter(
        anchor,
        root_public_key=bytes.fromhex(vectors["keys"]["root_public_hex"]),
        admin_binding=tweak.get("admin", chain["admin_identity"]),
        project_binding=tweak.get("project", chain["project_key"]),
        binding_list=tweak.get("blist", vectors["binding_list"]),
        retained_sequence=tweak.get("retained", 1),
        now=tweak.get("now", "2026-10-16T00:00:01Z"),
    )


def test_writer_seals_to_the_verified_project_key_and_bumps_the_revision(
    vectors: dict,
) -> None:
    w = _writer(vectors)
    assert w.epoch == 1
    ref = vectors["label_envelopes"][0]["object_ref"]
    env = w.seal("Checkout flow", purpose="label", object_ref=ref, base_revision=0)
    kw = {
        "tenant_ref": vectors["refs"]["tenant_ref"],
        "project_ref": vectors["refs"]["project_ref"],
        "purpose": "label",
        "object_ref": ref,
    }
    sk = _x(vectors["keys"]["project_private_hex"])
    assert open_label(sk, env, revision=1, **kw) == "Checkout flow"
    for wrong in (0, 2):  # revision is bound into the AAD
        with pytest.raises(LabelCryptoError):
            open_label(sk, env, revision=wrong, **kw)


def test_writer_refuses_before_encrypting(vectors: dict) -> None:
    chain = vectors["binding_chain"]
    with pytest.raises(LabelCryptoError, match="pinned"):
        _writer(vectors, pinned="1" * 64)
    with pytest.raises(LabelCryptoError):
        _writer(vectors, project={**chain["project_key"], "epoch": 2})
    revoked = _signed_list(
        vectors, revoked_key_ids=[chain["project_key"]["key_id"]], sequence=2
    )
    with pytest.raises(LabelCryptoError, match="revoked"):
        _writer(vectors, blist=revoked, retained=2)
    with pytest.raises(LabelCryptoError, match="not yet active"):
        _writer(vectors, now="2026-10-15T00:00:00Z")
    with pytest.raises(LabelCryptoError, match="sequence"):
        _writer(vectors, retained=5)
