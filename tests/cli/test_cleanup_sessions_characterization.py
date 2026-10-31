"""Characterization tests pinning the ``cleanup`` CLI command (cleanup_sessions)."""

from __future__ import annotations

import datetime as real_datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.local_commands import cleanup_sessions

MOD = "traigent.cli.local_commands"
NOW = real_datetime.datetime(2025, 6, 15, 12, 0, 0, tzinfo=real_datetime.UTC)
STORAGE_PATH = "/nonexistent/traigent-storage"


class _Clock:
    """Shared call log plus a datetime subclass with a pinned ``now``."""

    def __init__(self):
        self.events = []
        self.now_args = []
        self.iso_inputs = []
        clock = self

        class FakeDatetime(real_datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                clock.events.append("now")
                clock.now_args.append(tz)
                return NOW

            @classmethod
            def fromisoformat(cls, value):
                clock.iso_inputs.append(value)
                return super().fromisoformat(value)

        self.datetime_cls = FakeDatetime


class _Session:
    """Session stand-in that counts attribute reads."""

    def __init__(self, session_id, created_at, function_name="fn"):
        self._session_id = session_id
        self._created_at = created_at
        self._function_name = function_name
        self.status = "completed"  # only finished sessions are deletable
        self.reads = []

    @property
    def created_at(self):
        self.reads.append("created_at")
        return self._created_at

    @property
    def session_id(self):
        self.reads.append("session_id")
        return self._session_id

    @property
    def function_name(self):
        self.reads.append("function_name")
        return self._function_name


def _iso(**delta):
    return (NOW - real_datetime.timedelta(**delta)).replace(tzinfo=None).isoformat()


def _run(args, sessions=None, list_error=None, clock=None, cleanup_result=2):
    clock = clock or _Clock()
    storage = MagicMock()
    if list_error is not None:
        storage.list_sessions.side_effect = list_error
    else:
        storage.list_sessions.side_effect = lambda: (
            clock.events.append("list") or sessions
        )
    storage.cleanup_old_sessions.return_value = cleanup_result
    config = MagicMock()
    config.get_local_storage_path.return_value = STORAGE_PATH
    with (
        patch(f"{MOD}.TraigentConfig") as config_cls,
        patch(f"{MOD}.LocalStorageManager", return_value=storage) as storage_cls,
        patch("datetime.datetime", clock.datetime_cls),
    ):
        config_cls.from_environment.return_value = config
        result = CliRunner().invoke(cleanup_sessions, args)
    return result, storage, storage_cls, clock


def _lines(result):
    return result.output.splitlines()


def test_aware_older_listed_newer_not_listed():
    sessions = [
        _Session("old", "2025-06-01T00:00:00+00:00", "f_old"),
        _Session("new", "2025-06-14T00:00:00+00:00", "f_new"),
    ]
    result, *_ = _run(["--days", "10", "--dry-run"], sessions)
    assert result.exit_code == 0
    assert _lines(result) == [
        "Would delete 1 sessions older than 10 days:",
        "  - old (f_old)",
    ]


def test_naive_created_at_treated_as_utc():
    sessions = [
        _Session("old", "2025-06-01T00:00:00", "f_old"),
        _Session("new", "2025-06-14T00:00:00", "f_new"),
    ]
    result, *_ = _run(["--days", "10", "--dry-run"], sessions)
    assert result.exit_code == 0
    assert _lines(result) == [
        "Would delete 1 sessions older than 10 days:",
        "  - old (f_old)",
    ]


@pytest.mark.parametrize(
    ("created_at", "listed"),
    [
        ("2025-06-05T12:00:00+00:00", False),
        ("2025-06-05T12:00:00", False),
        ("2025-06-05T14:00:00+02:00", False),
        ("2025-06-05T11:59:59.999999+00:00", True),
        ("2025-06-05T11:59:59.999999", True),
        ("2025-06-05T14:59:59+03:00", True),
        ("2025-06-05T12:00:00.000001+00:00", False),
    ],
)
def test_cutoff_is_strictly_less_than(created_at, listed):
    result, *_ = _run(["--days", "10", "--dry-run"], [_Session("s", created_at)])
    assert result.exit_code == 0
    if listed:
        assert _lines(result) == [
            "Would delete 1 sessions older than 10 days:",
            "  - s (fn)",
        ]
    else:
        assert _lines(result) == ["No sessions older than 10 days found."]


def test_empty_listing_reports_none_found():
    result, *_ = _run(["--days", "7", "--dry-run"], [])
    assert result.exit_code == 0
    assert result.output == "No sessions older than 7 days found.\n"
    assert "Would delete" not in result.output


def test_no_session_old_enough_reports_none_found():
    sessions = [_Session("a", _iso(days=1)), _Session("b", _iso(hours=3))]
    result, *_ = _run(["--days", "5", "--dry-run"], sessions)
    assert result.exit_code == 0
    assert result.output == "No sessions older than 5 days found.\n"


def test_output_header_count_and_listing_order():
    sessions = [
        _Session("s3", _iso(days=20), "f3"),
        _Session("keep", _iso(days=1), "fk"),
        _Session("s1", _iso(days=40), "f1"),
        _Session("s2", _iso(days=30), "f2"),
    ]
    result, *_ = _run(["--days", "10", "--dry-run"], sessions)
    assert result.exit_code == 0
    assert result.output == (
        "Would delete 3 sessions older than 10 days:\n"
        "  - s3 (f3)\n"
        "  - s1 (f1)\n"
        "  - s2 (f2)\n"
    )


def test_days_default_is_thirty():
    sessions = [
        _Session("older", _iso(days=31)),
        _Session("younger", _iso(days=29)),
    ]
    result, *_ = _run(["--dry-run"], sessions)
    assert result.exit_code == 0
    assert _lines(result) == [
        "Would delete 1 sessions older than 30 days:",
        "  - older (fn)",
    ]


def test_clock_read_once_before_listing_and_listing_once():
    sessions = [_Session("a", _iso(days=20)), _Session("b", _iso(days=1))]
    result, storage, storage_cls, clock = _run(["--days", "10", "--dry-run"], sessions)
    assert result.exit_code == 0
    assert clock.events == ["now", "list"]
    assert clock.now_args == [real_datetime.UTC]
    storage.list_sessions.assert_called_once_with()
    storage_cls.assert_called_once_with(STORAGE_PATH)


def test_dry_run_never_calls_cleanup_old_sessions():
    result, storage, *_ = _run(["--days", "10", "--dry-run"], [])
    assert result.exit_code == 0
    storage.cleanup_old_sessions.assert_not_called()


def test_attribute_reads_per_session():
    old = _Session("old", _iso(days=20))
    new = _Session("new", _iso(days=1))
    result, *_ = _run(["--days", "10", "--dry-run"], [old, new])
    assert result.exit_code == 0
    assert old.reads == ["created_at", "session_id", "function_name"]
    assert new.reads == ["created_at"]


def test_iso_parse_called_once_per_session_in_order():
    sessions = [
        _Session("a", "2025-01-01T00:00:00"),
        _Session("b", "2025-06-14T00:00:00+00:00"),
    ]
    _, _, _, clock = _run(["--days", "10", "--dry-run"], sessions)
    assert clock.iso_inputs == ["2025-01-01T00:00:00", "2025-06-14T00:00:00+00:00"]


def test_malformed_created_at_halts_without_inspecting_later_rows():
    first = _Session("a", "2025-01-01T00:00:00")
    bad = _Session("b", "not-a-date")
    later = _Session("c", "2025-01-01T00:00:00")
    result, storage, _, clock = _run(["--days", "10", "--dry-run"], [first, bad, later])
    assert result.exit_code == 1
    assert result.output == (
        "Error cleaning up sessions: Invalid isoformat string: 'not-a-date'\n"
    )
    assert clock.iso_inputs == ["2025-01-01T00:00:00", "not-a-date"]
    assert later.reads == []
    assert first.reads == ["created_at"]
    assert bad.reads == ["created_at"]
    storage.cleanup_old_sessions.assert_not_called()


def test_non_string_created_at_reports_type_error():
    result, *_ = _run(["--days", "10", "--dry-run"], [_Session("a", None)])
    assert result.exit_code == 1
    assert result.output == (
        "Error cleaning up sessions: fromisoformat: argument must be str\n"
    )


def test_list_sessions_error_reported_and_exits():
    result, storage, _, clock = _run(
        ["--days", "10", "--dry-run"], list_error=RuntimeError("boom")
    )
    assert result.exit_code == 1
    assert result.output == "Error cleaning up sessions: boom\n"
    assert clock.events == ["now"]
    storage.cleanup_old_sessions.assert_not_called()


def test_actual_cleanup_branch_unchanged():
    result, storage, storage_cls, clock = _run(
        ["--days", "30", "--yes"], cleanup_result=2
    )
    assert result.exit_code == 0
    assert result.output == "✅ Cleaned up 2 sessions older than 30 days\n"
    storage.cleanup_old_sessions.assert_called_once_with(30)
    storage.list_sessions.assert_not_called()
    assert clock.events == []
    storage_cls.assert_called_once_with(STORAGE_PATH)


def test_actual_cleanup_default_days_and_error():
    result, storage, *_ = _run(["--yes"], cleanup_result=0)
    assert result.output == "✅ Cleaned up 0 sessions older than 30 days\n"
    storage.cleanup_old_sessions.assert_called_once_with(30)

    storage.cleanup_old_sessions.side_effect = Exception("Cleanup error")
    with (
        patch(f"{MOD}.TraigentConfig"),
        patch(f"{MOD}.LocalStorageManager", return_value=storage),
    ):
        failed = CliRunner().invoke(cleanup_sessions, ["--days", "5", "--yes"])
    assert failed.exit_code == 1
    assert failed.output == "Error cleaning up sessions: Cleanup error\n"


def test_config_error_reported_and_exits():
    with patch(f"{MOD}.TraigentConfig") as config_cls:
        config_cls.from_environment.side_effect = ValueError("bad config")
        result = CliRunner().invoke(cleanup_sessions, ["--dry-run"])
    assert result.exit_code == 1
    assert result.output == "Error cleaning up sessions: bad config\n"


def test_plain_namespace_rows_are_supported():
    ns = SimpleNamespace(
        session_id="x", function_name="g", created_at=_iso(days=99), status="completed"
    )
    result, *_ = _run(["--days", "1", "--dry-run"], [ns])
    assert _lines(result)[-1] == "  - x (g)"
