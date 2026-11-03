"""Publish / verify / load / idempotent artifact store, over the local backend."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tests.unit.snapshot import helpers
from traigent.snapshot import (
    CreateResult,
    EvaluatorInputs,
    ExecutionSnapshot,
    SnapshotArtifactStore,
    SnapshotErrorCode,
    SnapshotStoreError,
    build_execution_snapshot,
    load_snapshot,
    publish_snapshot,
    verify_snapshot,
)
from traigent.snapshot.backends import LocalDirStore


class CountingStore:
    """Wraps a store and counts objects actually created."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.created: list[str] = []
        self.max_object_bytes = inner.max_object_bytes

    def create_if_absent(self, key: str, data: bytes) -> CreateResult:
        result = self.inner.create_if_absent(key, data)
        if result is CreateResult.CREATED:
            self.created.append(key)
        return result

    def read(self, key: str) -> bytes | None:
        return self.inner.read(key)

    def locator(self) -> str:
        return self.inner.locator()


@pytest.fixture
def snapshot() -> ExecutionSnapshot:
    return build_execution_snapshot(
        dataset=[{"input": {"q": "a"}, "expected": "b"}],
        config={"temperature": 0.2},
        config_space={"model": ["x", "y"]},
        evaluator=EvaluatorInputs(metric_functions={"a": helpers.metric_a}),
    )


@pytest.fixture
def store(tmp_path: Path) -> LocalDirStore:
    return LocalDirStore(tmp_path / "store")


def blob_path(store: LocalDirStore, snapshot: ExecutionSnapshot, part: str) -> Path:
    sha = snapshot.manifest["parts"][part]["sha256"][len("sha256:") :]
    return Path(store.locator()[len("file://") :]) / "blobs" / "sha256" / sha


def test_publish_then_verify_then_load(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    ref = publish_snapshot(store, snapshot)
    assert ref.manifest_digest == snapshot.manifest_digest
    report = verify_snapshot(store, ref.manifest_digest)
    assert report.integrity == "ok" and report.ok
    loaded = load_snapshot(store, ref.manifest_digest)
    assert loaded.manifest_bytes == snapshot.manifest_bytes
    assert dict(loaded.parts) == dict(snapshot.parts)


class OrderRecordingStore(CountingStore):
    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.order: list[str] = []

    def create_if_absent(self, key: str, data: bytes) -> CreateResult:
        self.order.append(key)
        return super().create_if_absent(key, data)


def test_manifest_is_written_last(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    recording = OrderRecordingStore(store)
    publish_snapshot(recording, snapshot)
    order = recording.order
    assert order[-1].endswith("manifest.json") and len(order) > 1
    assert not any(key.endswith("manifest.json") for key in order[:-1])


def test_republish_same_snapshot_creates_zero_new_objects(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    counting = CountingStore(store)
    publish_snapshot(counting, snapshot)
    first = len(counting.created)
    assert first > 0
    publish_snapshot(counting, snapshot)
    assert len(counting.created) == first


def test_rebuild_gets_new_manifest_but_blobs_dedupe(store: LocalDirStore) -> None:
    kwargs: dict[str, Any] = {
        "dataset": [{"input": 1, "expected": 1}],
        "config": {"a": 1},
    }
    first = build_execution_snapshot(**kwargs)
    counting = CountingStore(store)
    publish_snapshot(counting, first)
    before = len(counting.created)
    import datetime as dt

    later = build_execution_snapshot(
        **kwargs, clock=lambda: dt.datetime(2030, 1, 1, tzinfo=dt.UTC)
    )
    assert later.manifest_digest != first.manifest_digest
    publish_snapshot(counting, later)
    assert len(counting.created) == before + 1  # only the new manifest


def test_tampered_blob_is_part_corrupt(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    ref = publish_snapshot(store, snapshot)
    blob_path(store, snapshot, "config/config.json").write_bytes(b'{"temperature":9}')
    report = verify_snapshot(store, ref.manifest_digest)
    assert report.integrity == "part_corrupt"
    assert report.problems == (("config/config.json", "part_corrupt"),)
    with pytest.raises(SnapshotStoreError) as caught:
        load_snapshot(store, ref.manifest_digest)
    assert caught.value.code is SnapshotErrorCode.CORRUPT
    assert caught.value.part == "config/config.json"


def test_same_size_tamper_is_detected(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    ref = publish_snapshot(store, snapshot)
    path = blob_path(store, snapshot, "config/config.json")
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert verify_snapshot(store, ref.manifest_digest).integrity == "part_corrupt"


def test_deleted_blob_is_part_missing(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    ref = publish_snapshot(store, snapshot)
    blob_path(store, snapshot, "dataset/examples.jsonl").unlink()
    report = verify_snapshot(store, ref.manifest_digest)
    assert report.integrity == "part_missing"
    assert ("dataset/examples.jsonl", "part_missing") in report.problems
    with pytest.raises(SnapshotStoreError) as caught:
        load_snapshot(store, ref.manifest_digest)
    assert caught.value.code is SnapshotErrorCode.NOT_FOUND


def test_replaced_manifest_is_manifest_mismatch(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    ref = publish_snapshot(store, snapshot)
    other = build_execution_snapshot(config={"different": True})
    root = Path(store.locator()[len("file://") :])
    hexdigest = ref.manifest_digest[len("sha256:") :]
    (root / "snapshots" / hexdigest / "manifest.json").write_bytes(other.manifest_bytes)
    report = verify_snapshot(store, ref.manifest_digest)
    assert report.integrity == "manifest_mismatch"
    with pytest.raises(SnapshotStoreError) as caught:
        load_snapshot(store, ref.manifest_digest)
    assert caught.value.code is SnapshotErrorCode.CORRUPT


def test_wrong_digest_asked_for_is_not_found(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    publish_snapshot(store, snapshot)
    with pytest.raises(SnapshotStoreError) as caught:
        verify_snapshot(store, "sha256:" + "0" * 64)
    assert caught.value.code is SnapshotErrorCode.NOT_FOUND
    with pytest.raises(ValueError):
        verify_snapshot(store, "not-a-digest")


def test_completeness_partial_lists_reasons(store: LocalDirStore) -> None:
    partial = build_execution_snapshot(config={"a": 1})
    ref = publish_snapshot(store, partial)
    report = verify_snapshot(store, ref.manifest_digest)
    assert report.integrity == "ok" and report.completeness == "partial"
    reasons = dict(report.unavailable)
    assert reasons["dataset/examples.jsonl"] == "NOT_PROVIDED"
    assert reasons["evaluator/source"] == "NOT_PROVIDED"


def test_completeness_complete_when_nothing_unavailable(store: LocalDirStore) -> None:
    snap = build_execution_snapshot(
        dataset=[{"input": 1, "expected": 1}],
        config={"a": 1},
        config_space={"b": [1]},
        evaluator=EvaluatorInputs(metric_functions={"a": helpers.metric_a}),
        agent=helpers.sample_agent,
        declared_assets={"prompts": {}, "tool_definitions": {}},
        capture_git_metadata=True,
    )
    unavailable = dict(snap.unavailable())
    # git state of this checkout varies; everything else is stored
    assert set(unavailable) <= {"agent/git.json", "agent/git_diff.patch"}


class FlakyStore(CountingStore):
    def __init__(self, inner: Any, code: SnapshotErrorCode) -> None:
        super().__init__(inner)
        self.code = code
        self.broken = False

    def read(self, key: str) -> bytes | None:
        if self.broken and "blobs/" in key:
            raise SnapshotStoreError(self.code, backend="fake")
        return super().read(key)


@pytest.mark.parametrize(
    "code", [SnapshotErrorCode.AUTH, SnapshotErrorCode.BACKEND_UNAVAILABLE]
)
def test_transport_or_auth_error_raises_never_part_missing(
    store: LocalDirStore, snapshot: ExecutionSnapshot, code: SnapshotErrorCode
) -> None:
    flaky = FlakyStore(store, code)
    ref = publish_snapshot(flaky, snapshot)
    flaky.broken = True
    with pytest.raises(SnapshotStoreError) as caught:
        verify_snapshot(flaky, ref.manifest_digest)
    assert caught.value.code is code
    with pytest.raises(SnapshotStoreError):
        load_snapshot(flaky, ref.manifest_digest)


def test_publish_refuses_inconsistent_snapshot(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    bad_parts = dict(snapshot.parts)
    bad_parts["config/config.json"] = b'{"temperature":9}'
    bad = ExecutionSnapshot(snapshot.manifest_bytes, bad_parts)
    with pytest.raises(SnapshotStoreError) as caught:
        publish_snapshot(store, bad)
    assert caught.value.code is SnapshotErrorCode.CORRUPT
    missing = {k: v for k, v in snapshot.parts.items() if k != "config/config.json"}
    with pytest.raises(SnapshotStoreError):
        publish_snapshot(store, ExecutionSnapshot(snapshot.manifest_bytes, missing))
    extra = {**snapshot.parts, "agent/extra.txt": b"x"}
    with pytest.raises(SnapshotStoreError):
        publish_snapshot(store, ExecutionSnapshot(snapshot.manifest_bytes, extra))


def test_publish_blob_conflict_is_reported(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    sha = snapshot.manifest["parts"]["config/config.json"]["sha256"][len("sha256:") :]
    store.create_if_absent(f"blobs/sha256/{sha}", b"poisoned")
    with pytest.raises(SnapshotStoreError) as caught:
        publish_snapshot(store, snapshot)
    assert caught.value.code is SnapshotErrorCode.CONFLICT


# ---------------------------------------------------------------- idempotency


def test_idempotency_same_key_same_digest_same_ref(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    artifacts = SnapshotArtifactStore(store)
    first = artifacts.put(snapshot, "run-1")
    second = artifacts.put(snapshot, "run-1")
    assert first == second
    record = store.read("idem/" + hashlib.sha256(b"run-1").hexdigest())
    assert record == snapshot.manifest_digest.encode()


def test_idempotency_same_key_different_digest_conflicts(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    artifacts = SnapshotArtifactStore(store)
    artifacts.put(snapshot, "run-1")
    other = build_execution_snapshot(config={"other": 1})
    with pytest.raises(SnapshotStoreError) as caught:
        artifacts.put(other, "run-1")
    assert caught.value.code is SnapshotErrorCode.CONFLICT
    # the conflicting snapshot was not published
    assert store.read(f"snapshots/{other.manifest_digest[7:]}/manifest.json") is None


def test_idempotency_key_is_hashed_in_the_store(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    SnapshotArtifactStore(store).put(snapshot, "SECRET-KEY-NAME")
    root = Path(store.locator()[len("file://") :])
    assert "SECRET-KEY-NAME" not in "".join(str(p) for p in root.rglob("*"))
    with pytest.raises(ValueError):
        SnapshotArtifactStore(store).put(snapshot, "")


def test_artifact_store_get_roundtrip(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    artifacts = SnapshotArtifactStore(store)
    ref = artifacts.put(snapshot, "k")
    loaded = artifacts.get(ref)
    assert loaded.manifest_digest == snapshot.manifest_digest
    assert dict(loaded.parts) == dict(snapshot.parts)
    assert store.locator() in repr(artifacts)


def test_artifact_store_get_validates_bytes(
    store: LocalDirStore, snapshot: ExecutionSnapshot
) -> None:
    artifacts = SnapshotArtifactStore(store)
    ref = artifacts.put(snapshot, "k")
    blob_path(store, snapshot, "config/config.json").write_bytes(b"{}")
    with pytest.raises(SnapshotStoreError) as caught:
        artifacts.get(ref)
    assert caught.value.code is SnapshotErrorCode.CORRUPT


def test_satisfies_artifact_store_port_shape() -> None:
    from traigent.connectors.ports import ArtifactStore

    assert {"put", "get"} <= set(
        dir(SnapshotArtifactStore)
    ) and ArtifactStore is not None
