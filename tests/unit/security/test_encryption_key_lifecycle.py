"""Key reuse and decrypt-after-rotation behaviour for EncryptionManager."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from traigent.security.encryption import (
    EncryptionKeyNotFoundError,
    EncryptionManager,
    KeyManager,
)

PASSWORD = "store-pass-for-tests"  # pragma: allowlist secret


def test_encrypt_reuses_active_key_instead_of_minting_per_call() -> None:
    km = KeyManager()
    em = EncryptionManager(km)
    results = [em.encrypt(f"payload-{i}") for i in range(5)]
    assert len(km.keys) == 1
    assert {r["key_id"] for r in results} == set(km.keys)
    assert len({r["iv"] for r in results}) == 5


def test_persistent_store_does_not_grow_per_encrypt(tmp_path) -> None:
    store = tmp_path / "keys.json"
    km = KeyManager(storage_path=store, master_password=PASSWORD)
    em = EncryptionManager(km)
    em.encrypt("a")
    size = store.stat().st_size
    em.encrypt("b")
    em.encrypt("c")
    assert store.stat().st_size == size
    assert len(KeyManager(storage_path=store, master_password=PASSWORD).keys) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["keys.json"]


def test_round_trip_across_rotation() -> None:
    km = KeyManager()
    em = EncryptionManager(km)
    old = em.encrypt(b"before rotation")
    new_key_id = km.rotate_key(old["key_id"])
    assert km.get_key(old["key_id"]) is None  # no longer usable for encryption
    assert em.decrypt(old) == b"before rotation"
    fresh = em.encrypt(b"after rotation")
    assert fresh["key_id"] == new_key_id
    assert em.decrypt(fresh) == b"after rotation"
    assert len(km.keys) == 2


def test_round_trip_after_key_expiry_creates_new_active_key() -> None:
    km = KeyManager()
    em = EncryptionManager(km)
    old = em.encrypt(b"expiring")
    km.key_metadata[old["key_id"]].expires_at = datetime.now(UTC) - timedelta(days=1)
    assert em.decrypt(old) == b"expiring"
    fresh = em.encrypt(b"new")
    assert fresh["key_id"] != old["key_id"]
    assert em.decrypt(fresh) == b"new"


def test_deleted_key_still_fails_closed() -> None:
    km = KeyManager()
    em = EncryptionManager(km)
    enc = em.encrypt(b"x")
    km.delete_key(enc["key_id"])
    with pytest.raises(EncryptionKeyNotFoundError):
        em.decrypt(enc)
