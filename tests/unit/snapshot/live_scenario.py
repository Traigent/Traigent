"""One end-to-end scenario shared by the optional live emulator tests.

Not collected by pytest (no ``test_`` prefix). Each live test supplies a store
plus ``tamper(key, data)``, which overwrites an object through the provider's own
raw client, bypassing the store's no-clobber rule.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.unit.snapshot import helpers
from traigent.snapshot import (
    CreateResult,
    EvaluatorInputs,
    SnapshotErrorCode,
    SnapshotStoreError,
    build_execution_snapshot,
    publish_snapshot,
    verify_snapshot,
)
from traigent.snapshot.store import blob_key


class Counting:
    """Wraps a store and records the keys it actually created."""

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


def run_scenario(
    store: Any, tamper: Callable[[str, bytes], None], tmp_path: Path
) -> dict[str, Any]:
    """publish -> verify -> republish -> tamper -> verify -> conflict. Returns facts."""
    _, agent = helpers.make_git_agent(tmp_path)
    snapshot = build_execution_snapshot(
        dataset=[{"input": {"q": "a"}, "expected": "b"}],
        config={"temperature": 0.2},
        config_space={"model": ["x", "y"]},
        evaluator=EvaluatorInputs(metric_functions={"a": helpers.metric_a}),
        agent=agent,
        declared_assets={"prompts": {"p": "hi"}, "tool_definitions": {}},
    )
    counting = Counting(store)
    ref = publish_snapshot(counting, snapshot)
    first_created = len(counting.created)
    report = verify_snapshot(store, ref.manifest_digest)
    assert report.integrity == "ok" and report.completeness == "complete", report

    counting.created.clear()
    assert publish_snapshot(counting, snapshot).manifest_digest == ref.manifest_digest
    assert counting.created == []  # republish: zero new objects

    entry = snapshot.manifest["parts"]["config/config.json"]
    victim = blob_key(entry["sha256"])
    tamper(victim, b'{"temperature":9}')
    tampered = verify_snapshot(store, ref.manifest_digest)
    assert tampered.integrity == "part_corrupt", tampered
    assert ("config/config.json", "part_corrupt") in tampered.problems

    try:  # the store never overwrites the tampered blob either
        publish_snapshot(counting, snapshot)
    except SnapshotStoreError as error:
        assert error.code is SnapshotErrorCode.CONFLICT
    else:
        raise AssertionError("republish over a tampered blob must conflict")
    assert counting.created == []

    assert store.create_if_absent("blobs/scenario-x", b"one") is CreateResult.CREATED
    try:
        store.create_if_absent("blobs/scenario-x", b"two")
    except SnapshotStoreError as error:
        assert error.code is SnapshotErrorCode.CONFLICT
    else:
        raise AssertionError("a conflicting create must raise CONFLICT")
    assert store.read("blobs/scenario-x") == b"one"
    return {"objects_first_publish": first_created}
