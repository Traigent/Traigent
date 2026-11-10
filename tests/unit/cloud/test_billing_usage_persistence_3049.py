"""UsageTracker must not lose or silently discard usage history (#3049)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from traigent.cloud.billing import UsageTracker


def _record(tracker: UsageTracker, name: str) -> None:
    asyncio.run(tracker.record_optimization(name, 5, 10, 1.0))


def _on_disk(path: Path) -> list[dict]:
    return json.loads(path.read_text())["usage_records"]


def test_two_trackers_on_one_file_keep_both_records(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    first = UsageTracker(storage_path=str(path))
    second = UsageTracker(storage_path=str(path))
    _record(first, "a")
    _record(second, "b")
    assert sorted(r["function_name"] for r in _on_disk(path)) == ["a", "b"]
    assert len(second._usage_records) == 2


def test_unreadable_file_is_quarantined_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    first = UsageTracker(storage_path=str(path))
    for name in "abc":
        _record(first, name)
    original = path.read_text()
    path.write_text(original[:60])  # torn file
    torn = path.read_text()

    second = UsageTracker(storage_path=str(path))
    _record(second, "d")

    assert [r["function_name"] for r in _on_disk(path)] == ["d"]
    quarantined = list(tmp_path.glob("usage.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == torn


def test_write_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    tracker = UsageTracker(storage_path=str(path))
    _record(tracker, "a")
    assert {p.name for p in tmp_path.iterdir()} <= {"usage.json", "usage.json.lock"}
