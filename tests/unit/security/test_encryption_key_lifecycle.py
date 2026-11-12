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
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        ".keys.json.lock",
        "keys.json",
    ]


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


def test_two_managers_rotating_concurrently_lose_no_keys(tmp_path) -> None:
    import threading

    store = tmp_path / "shared.json"
    managers = [
        KeyManager(storage_path=store, master_password=PASSWORD) for _ in range(2)
    ]
    ems = [EncryptionManager(m) for m in managers]
    results: list[dict] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def worker(idx: int) -> None:
        try:
            for i in range(6):
                enc = ems[idx].encrypt(f"w{idx}-{i}".encode())
                with results_lock:
                    results.append(enc)
                managers[idx].rotate_key(enc["key_id"])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 12

    fresh = EncryptionManager(KeyManager(storage_path=store, master_password=PASSWORD))
    for enc in results:
        assert fresh.decrypt(enc).startswith(b"w")


def test_stale_manager_does_not_overwrite_newer_store(tmp_path) -> None:
    store = tmp_path / "shared.json"
    a = KeyManager(storage_path=store, master_password=PASSWORD)
    b = KeyManager(storage_path=store, master_password=PASSWORD)
    key_a = a.generate_key("AES-256")
    key_b = b.generate_key("AES-256")  # b never saw key_a in memory
    reloaded = KeyManager(storage_path=store, master_password=PASSWORD)
    assert {key_a, key_b} <= set(reloaded.keys)
