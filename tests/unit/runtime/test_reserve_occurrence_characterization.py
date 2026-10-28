"""Characterization of the occurrence ledger reservation, pinned before extraction.

File-backed: the ledger document is seeded and read back as plain JSON, expected values
are literals, and collaborator call order and counts are recorded.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from traigent.runtime import manifest
from traigent.runtime.model import RuntimeRequirementError

AGENT = "agent-under-test"
NOW = datetime(2030, 1, 15, 12, 0, 0, tzinfo=UTC)
NOW_ISO = "2030-01-15T12:00:00+00:00"
LEASE = timedelta(hours=24)
FRESH = "2030-01-15T11:00:00+00:00"  # one hour old: inside the lease
OLD = "2030-01-14T10:00:00+00:00"  # 26 hours old: expired
EDGE = "2030-01-14T12:00:00+00:00"  # exactly 24 hours old
JUST_INSIDE = "2030-01-14T12:00:00.000001+00:00"  # 24 hours minus one microsecond


@pytest.fixture
def state(tmp_path: Path) -> Path:
    directory = tmp_path / "state"
    manifest.check_state_dir(directory, create=True)
    return directory


def _path(state: Path) -> Path:
    return manifest._occ_path(state, AGENT)


def _seed(state: Path, doc: Any) -> None:
    path = _path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _load(state: Path) -> Any:
    return json.loads(_path(state).read_text(encoding="utf-8"))


def _reserve(
    state: Path, key: str = "b1|fix|0", info: Any = None, **kwargs: Any
) -> bool:
    kwargs.setdefault("now", NOW)
    return manifest.reserve_occurrence(
        state, AGENT, key, {} if info is None else info, **kwargs
    )


def _refused(
    state: Path, code: str, path: str, key: str = "b1|fix|0", **kwargs: Any
) -> None:
    before = _path(state).read_text(encoding="utf-8") if _path(state).exists() else None
    with pytest.raises(RuntimeRequirementError) as info:
        _reserve(state, key, **kwargs)
    assert (info.value.code, info.value.path) == (code, path)
    after = _path(state).read_text(encoding="utf-8") if _path(state).exists() else None
    assert after == before


def _started(heartbeat: str = FRESH, **extra: Any) -> dict[str, Any]:
    return {
        "status": "started",
        "started_at": heartbeat,
        "heartbeat_at": heartbeat,
        **extra,
    }


# ------------------------------------------------------------ the own-key record


def test_fresh_reservation_returns_true_and_writes_the_record(state: Path) -> None:
    assert _reserve(state) is True
    assert _load(state) == {
        "b1|fix|0": {
            "started_at": NOW_ISO,
            "heartbeat_at": NOW_ISO,
            "status": "started",
            "attempts": 1,
        }
    }


def test_record_keeps_run_info_but_fixed_fields_win(state: Path) -> None:
    info = {
        "run_id": "r9",
        "status": "x",
        "attempts": 99,
        "heartbeat_at": "h",
        "note": "n",
    }
    assert _reserve(state, info=info) is True
    assert _load(state)["b1|fix|0"] == {
        "run_id": "r9",
        "note": "n",
        "started_at": NOW_ISO,
        "heartbeat_at": NOW_ISO,
        "status": "started",
        "attempts": 1,
    }


def test_caller_start_time_is_honoured_for_both_stamps(state: Path) -> None:
    assert _reserve(state, info={"started_at": "2029-12-31T00:00:00+00:00"}) is True
    record = _load(state)["b1|fix|0"]
    assert record["started_at"] == "2029-12-31T00:00:00+00:00"
    assert record["heartbeat_at"] == "2029-12-31T00:00:00+00:00"


def test_non_string_start_time_is_stringified(state: Path) -> None:
    assert _reserve(state, info={"started_at": 5}) is True
    record = _load(state)["b1|fix|0"]
    assert (record["started_at"], record["heartbeat_at"]) == ("5", "5")


def test_run_info_mapping_is_converted_twice(state: Path) -> None:
    class Counting(Mapping[str, Any]):
        def __init__(self) -> None:
            self.iterations = 0

        def __getitem__(self, key: str) -> Any:
            return {"note": "n"}[key]

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            return iter(["note"])

        def __len__(self) -> int:
            return 1

    info = Counting()
    assert manifest.reserve_occurrence(state, AGENT, "b1|fix|0", info, now=NOW) is True
    assert info.iterations == 2
    assert _load(state)["b1|fix|0"]["note"] == "n"


@pytest.mark.parametrize(
    "record",
    [
        {"status": "started", "started_at": FRESH, "heartbeat_at": FRESH},
        {"status": "started", "run_id": "r1"},
        {"status": "started", "finished": True},
        {"status": "succeeded"},
        {"status": "cancelled", "attempts": 3},
        {},
    ],
)
def test_existing_non_failed_record_blocks_the_key(
    state: Path, record: dict[str, Any]
) -> None:
    _seed(state, {"b1|fix|0": record})
    assert _reserve(state, now=NOW) is False
    assert _load(state) == {"b1|fix|0": record}


def test_failed_record_is_released_with_attempt_count_plus_one(state: Path) -> None:
    _seed(state, {"b1|fix|0": {"status": "failed", "attempts": 2, "keep": "no"}})
    assert _reserve(state, info={"run_id": "r2"}, attempt=3) is True
    assert _load(state) == {
        "b1|fix|0": {
            "run_id": "r2",
            "started_at": NOW_ISO,
            "heartbeat_at": NOW_ISO,
            "status": "started",
            "attempts": 3,
        }
    }


def test_failed_record_without_attempts_counts_one_failure(state: Path) -> None:
    _seed(state, {"b1|fix|0": {"status": "failed"}})
    assert _reserve(state, attempt=2) is True
    assert _load(state)["b1|fix|0"]["attempts"] == 2


def test_null_record_is_a_first_attempt(state: Path) -> None:
    _seed(state, {"b1|fix|0": None})
    assert _reserve(state, attempt=1) is True
    assert _load(state)["b1|fix|0"]["attempts"] == 1


@pytest.mark.parametrize("attempt", [1, 2, 4, 0, -1])
def test_attempt_that_does_not_match_the_failures_is_a_hold(
    state: Path, attempt: int
) -> None:
    _seed(state, {"b1|fix|0": {"status": "failed", "attempts": 2}})
    _refused(state, "hold", "approval", attempt=attempt)


def test_attempt_none_skips_the_generation_check(state: Path) -> None:
    _seed(state, {"b1|fix|0": {"status": "failed", "attempts": 5}})
    assert _reserve(state, attempt=None) is True
    assert _load(state)["b1|fix|0"]["attempts"] == 6


def test_attempt_one_on_a_first_reservation_is_accepted_and_two_is_a_hold(
    state: Path,
) -> None:
    _refused(state, "hold", "approval", attempt=2)
    assert not _path(state).exists()
    assert _reserve(state, attempt=1) is True


def test_expired_lease_record_is_read_as_failed_and_released(state: Path) -> None:
    _seed(state, {"b1|fix|0": _started(OLD, attempts=1)})
    assert _reserve(state, attempt=2) is True
    assert _load(state)["b1|fix|0"]["attempts"] == 2


def test_lease_boundary_exactly_the_lease_is_expired(state: Path) -> None:
    _seed(state, {"b1|fix|0": _started(EDGE)})
    assert _reserve(state) is True


def test_lease_boundary_one_microsecond_inside_is_still_held(state: Path) -> None:
    record = _started(JUST_INSIDE)
    _seed(state, {"b1|fix|0": record})
    assert _reserve(state) is False
    assert _load(state) == {"b1|fix|0": record}


def test_custom_lease_changes_the_boundary(state: Path) -> None:
    _seed(state, {"b1|fix|0": _started(FRESH)})
    assert _reserve(state, lease=timedelta(hours=2)) is False
    assert _reserve(state, lease=timedelta(hours=1)) is True


def test_run_id_keeps_an_old_record_alive(state: Path) -> None:
    record = _started(OLD, run_id="r1")
    _seed(state, {"b1|fix|0": record})
    assert _reserve(state) is False
    assert _load(state) == {"b1|fix|0": record}


def test_unreadable_start_time_keeps_blocking(state: Path) -> None:
    record = {"status": "started", "started_at": "garbage"}
    _seed(state, {"b1|fix|0": record})
    assert _reserve(state) is False


@pytest.mark.parametrize("bad", ["text", ["x"], 5])
def test_non_mapping_record_for_the_key_raises_attribute_error(
    state: Path, bad: Any
) -> None:
    _seed(state, {"b1|fix|0": bad})
    with pytest.raises(AttributeError):
        _reserve(state)
    assert _load(state) == {"b1|fix|0": bad}


# ------------------------------------------------------------ whole-document shape


def test_non_object_document_is_a_state_error(state: Path) -> None:
    _seed(state, ["not", "a", "dict"])
    _refused(state, "local_state_invalid", "occurrences.json")


def test_unparseable_document_is_a_state_error(state: Path) -> None:
    path = _path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    _refused(state, "local_state_invalid", "occurrences.json")


def test_other_entries_are_preserved_on_write(state: Path) -> None:
    other = {"z|rule|1": {"status": "succeeded"}}
    _seed(state, other)
    assert _reserve(state) is True
    doc = _load(state)
    assert doc["z|rule|1"] == {"status": "succeeded"}
    assert set(doc) == {"z|rule|1", "b1|fix|0"}


# --------------------------------------------------------- episode-wide exclusion


def test_live_record_of_another_rule_in_the_episode_blocks(state: Path) -> None:
    _seed(state, {"b1|notify|0": _started(FRESH)})
    _refused(state, "episode_fix_running", "breach")


def test_live_record_with_a_run_id_still_blocks_however_old(state: Path) -> None:
    _seed(state, {"b1|notify|0": _started(OLD, run_id="r1")})
    _refused(state, "episode_fix_running", "breach")


@pytest.mark.parametrize("finished", [True, "yes", 1, "2030-01-15T00:00:00+00:00"])
def test_finished_record_in_the_episode_does_not_block(
    state: Path, finished: Any
) -> None:
    _seed(state, {"b1|notify|0": _started(FRESH, finished=finished)})
    assert _reserve(state) is True


@pytest.mark.parametrize("finished", [None, False, 0, ""])
def test_falsy_finished_marker_does_not_release_the_block(
    state: Path, finished: Any
) -> None:
    _seed(state, {"b1|notify|0": _started(FRESH, finished=finished)})
    _refused(state, "episode_fix_running", "breach")


@pytest.mark.parametrize("status", ["failed", "succeeded", "cancelled", "", None])
def test_non_started_record_in_the_episode_does_not_block(
    state: Path, status: Any
) -> None:
    _seed(state, {"b1|notify|0": {"status": status, "started_at": FRESH}})
    assert _reserve(state) is True


def test_expired_record_in_the_episode_reads_as_failed_and_does_not_block(
    state: Path,
) -> None:
    _seed(state, {"b1|notify|0": _started(OLD)})
    assert _reserve(state) is True


def test_episode_lease_boundary_exactly_the_lease_does_not_block(state: Path) -> None:
    _seed(state, {"b1|notify|0": _started(EDGE)})
    assert _reserve(state) is True


def test_episode_lease_boundary_one_microsecond_inside_blocks(state: Path) -> None:
    _seed(state, {"b1|notify|0": _started(JUST_INSIDE)})
    _refused(state, "episode_fix_running", "breach")


def test_episode_check_uses_the_given_lease(state: Path) -> None:
    _seed(state, {"b1|notify|0": _started(FRESH)})
    assert _reserve(state, lease=timedelta(minutes=30)) is True


def test_record_of_another_episode_does_not_block(state: Path) -> None:
    _seed(state, {"b2|fix|0": _started(FRESH)})
    assert _reserve(state) is True


@pytest.mark.parametrize("other", [None, "text", 5, ["x"], True])
def test_non_mapping_entries_of_the_episode_are_ignored(
    state: Path, other: Any
) -> None:
    _seed(state, {"b1|notify|0": other})
    assert _reserve(state) is True


def test_key_without_separators_shares_an_episode_with_its_prefix_keys(
    state: Path,
) -> None:
    _seed(state, {"k|rule|1": _started(FRESH)})
    _refused(state, "episode_fix_running", "breach", key="k")


def test_key_with_one_separator_is_its_own_episode(state: Path) -> None:
    _seed(state, {"a|1": _started(FRESH)})
    # `a|1` has a single separator: its episode is the whole key, so `a|2` is unrelated
    assert _reserve(state, key="a|2") is True
    assert set(_load(state)) == {"a|1", "a|2"}


def test_hold_is_raised_before_the_episode_exclusion(state: Path) -> None:
    _seed(
        state,
        {
            "b1|fix|0": {"status": "failed", "attempts": 1},
            "b1|notify|0": _started(FRESH),
        },
    )
    _refused(state, "hold", "approval", attempt=5)


def test_own_key_is_never_its_own_episode_blocker(state: Path) -> None:
    _seed(state, {"b1|fix|0": {"status": "failed", "attempts": 1}})
    assert _reserve(state, attempt=2) is True


def test_first_blocking_record_ends_the_scan(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(
        state,
        {
            "b1|a|0": _started(FRESH),
            "b1|b|0": _started(FRESH),
            "b1|c|0": _started(FRESH),
        },
    )
    seen: list[Any] = []
    real = manifest.effective_record

    def spy(record: Any, **kwargs: Any) -> Any:
        seen.append(record)
        return real(record, **kwargs)

    monkeypatch.setattr(manifest, "effective_record", spy)
    _refused(state, "episode_fix_running", "breach")
    assert seen == [None, _started(FRESH)]


# -------------------------------------------------------- clock and time handling


def test_clock_is_read_once_after_the_lock_and_replaces_now(state: Path) -> None:
    calls: list[str] = []

    def clock() -> datetime:
        calls.append("clock")
        return NOW + timedelta(hours=1)

    assert _reserve(state, clock=clock, now=NOW) is True
    assert calls == ["clock"]
    assert _load(state)["b1|fix|0"]["started_at"] == "2030-01-15T13:00:00+00:00"


def test_clock_value_decides_lease_expiry(state: Path) -> None:
    _seed(state, {"b1|fix|0": _started(FRESH)})
    assert _reserve(state, now=NOW, clock=lambda: NOW) is False
    assert _reserve(state, now=NOW, clock=lambda: NOW + timedelta(days=2)) is True


def test_failing_clock_propagates_before_the_ledger_is_read(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[str] = []
    real = manifest._read_json

    def spy(path: Path, default: Any) -> Any:
        reads.append("read")
        return real(path, default)

    monkeypatch.setattr(manifest, "_read_json", spy)

    def clock() -> datetime:
        raise RuntimeError("clock failed")

    with pytest.raises(RuntimeError, match="clock failed"):
        _reserve(state, clock=clock)
    assert reads == []


def test_utcnow_is_consulted_only_where_no_time_was_given(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake() -> datetime:
        calls.append(1)
        return NOW

    monkeypatch.setattr(manifest, "utcnow", fake)
    assert manifest.reserve_occurrence(state, AGENT, "b1|fix|0", {}) is True
    assert len(calls) == 1
    assert _load(state)["b1|fix|0"]["started_at"] == NOW_ISO


def test_utcnow_call_count_with_an_existing_record_and_an_episode_entry(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(
        state,
        {
            "b1|fix|0": {"status": "failed", "attempts": 1},
            "b1|notify|0": {"status": "succeeded"},
            "b2|x|0": {"status": "succeeded"},
        },
    )
    calls: list[int] = []

    def fake() -> datetime:
        calls.append(1)
        return NOW

    monkeypatch.setattr(manifest, "utcnow", fake)
    assert manifest.reserve_occurrence(state, AGENT, "b1|fix|0", {}) is True
    # own record (1) + the one other record of the episode (1) + the start stamp (1)
    assert len(calls) == 3


# --------------------------------------------------------- collaborators and locks


def test_collaborator_call_sequences(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(
        state,
        {
            "b1|fix|0": {"status": "failed", "attempts": 1},
            "b2|fix|0": {"status": "succeeded"},
            "b1|notify|0": {"status": "succeeded"},
            "b1|fix|1": "text",
        },
    )
    events: list[tuple[Any, ...]] = []
    real_effective = manifest.effective_record
    real_breach = manifest._breach_of_key
    real_read = manifest._read_json
    real_write = manifest._write_json

    def effective(record: Any, **kwargs: Any) -> Any:
        events.append(
            ("effective", record, tuple(sorted(kwargs)), kwargs["now"], kwargs["lease"])
        )
        return real_effective(record, **kwargs)

    def breach(key: str) -> str:
        events.append(("breach", key))
        return real_breach(key)

    def read(path: Path, default: Any) -> Any:
        events.append(("read", path.name))
        return real_read(path, default)

    def write(path: Path, doc: Any) -> None:
        events.append(("write", path.name))
        real_write(path, doc)

    monkeypatch.setattr(manifest, "effective_record", effective)
    monkeypatch.setattr(manifest, "_breach_of_key", breach)
    monkeypatch.setattr(manifest, "_read_json", read)
    monkeypatch.setattr(manifest, "_write_json", write)
    assert _reserve(state, attempt=2, lease=LEASE) is True
    assert events == [
        ("read", "occurrences.json"),
        (
            "effective",
            {"status": "failed", "attempts": 1},
            ("lease", "now"),
            NOW,
            LEASE,
        ),
        ("breach", "b1|fix|0"),
        ("breach", "b2|fix|0"),
        ("breach", "b1|notify|0"),
        ("effective", {"status": "succeeded"}, ("lease", "now"), NOW, LEASE),
        ("breach", "b1|fix|1"),
        ("effective", None, ("lease", "now"), NOW, LEASE),
        ("write", "occurrences.json"),
    ]


def test_refusals_never_write(state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(state, {"b1|notify|0": _started(FRESH)})
    writes: list[int] = []
    monkeypatch.setattr(manifest, "_write_json", lambda path, doc: writes.append(1))
    with pytest.raises(RuntimeRequirementError):
        _reserve(state)
    assert writes == []


def test_lock_is_released_after_a_refusal(state: Path) -> None:
    _seed(state, {"b1|notify|0": _started(FRESH)})
    with pytest.raises(RuntimeRequirementError):
        _reserve(state)
    assert _reserve(state, key="b2|fix|0") is True
    assert manifest._HELD.__dict__.get("paths", set()) == set()


def test_lock_is_released_after_a_write_failure(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(path: Path, doc: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manifest, "_write_json", boom)
    with pytest.raises(OSError, match="disk full"):
        _reserve(state)
    monkeypatch.undo()
    assert _reserve(state) is True


def test_effective_record_failure_in_the_scan_propagates_without_a_write(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(state, {"b1|notify|0": {"status": "succeeded"}})
    real = manifest.effective_record
    calls: list[int] = []

    def flaky(record: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("scan failed")
        return real(record, **kwargs)

    monkeypatch.setattr(manifest, "effective_record", flaky)
    with pytest.raises(RuntimeError, match="scan failed"):
        _reserve(state)
    assert _load(state) == {"b1|notify|0": {"status": "succeeded"}}
