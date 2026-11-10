"""UsageTracker must not lose or silently discard usage history (#3049)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from traigent.cloud import billing
from traigent.cloud.billing import BillingManager, UsageStorageError, UsageTracker


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


def test_failed_replace_leaves_original_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "usage.json"
    tracker = UsageTracker(storage_path=str(path))
    _record(tracker, "a")
    before = path.read_text()

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(billing.os, "replace", boom)
    _record(tracker, "b")  # write failure is logged, not raised
    monkeypatch.undo()

    assert path.read_text() == before
    assert {p.name for p in tmp_path.iterdir()} <= {"usage.json", "usage.json.lock"}


def test_unsaved_record_survives_next_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "usage.json"
    tracker = UsageTracker(storage_path=str(path))
    real_replace = billing.os.replace
    monkeypatch.setattr(
        billing.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
    )
    _record(tracker, "lost?")
    monkeypatch.setattr(billing.os, "replace", real_replace)
    _record(tracker, "next")
    assert sorted(r["function_name"] for r in _on_disk(path)) == ["lost?", "next"]


def test_unquarantinable_corrupt_file_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "usage.json"
    path.write_text("{not json")
    monkeypatch.setattr(
        billing.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("ro"))
    )
    with pytest.raises(UsageStorageError):
        UsageTracker(storage_path=str(path))
    assert path.read_text() == "{not json"


def test_stats_and_limits_count_other_trackers_writes(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    first = UsageTracker(storage_path=str(path))
    second = UsageTracker(storage_path=str(path))
    _record(second, "other")
    stats = asyncio.run(first.get_usage_stats())
    assert stats["total_optimizations"] == 1
    manager = BillingManager(first)
    result = asyncio.run(manager.check_usage_limits(1, 1))
    plan = first.get_billing_plan(manager.current_plan)
    assert result["remaining_credits"] == plan.monthly_credits - stats["total_credits"]
    assert stats["total_credits"] > 0


def test_lock_falls_back_to_msvcrt_when_fcntl_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(_fd: int, mode: int, _n: int) -> None:
            calls.append(mode)

    monkeypatch.setattr(billing, "fcntl", None)
    monkeypatch.setattr(billing, "msvcrt", FakeMsvcrt)
    with billing._storage_lock(tmp_path / "usage.json"):
        pass
    assert calls == [1, 2]


def test_no_lock_support_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(billing, "fcntl", None)
    monkeypatch.setattr(billing, "msvcrt", None)
    monkeypatch.setattr(billing, "_lock_warning_emitted", False)
    with caplog.at_level("WARNING"):
        for _ in range(2):
            with billing._storage_lock(tmp_path / "usage.json"):
                pass
    assert sum("file-locking" in r.getMessage() for r in caplog.records) == 1
