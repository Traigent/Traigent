"""#3099: one naive (offset-less) session created_at must not empty usage stats.

collect_usage_stats compared fromisoformat(created_at) with an aware ``now`` and
_get_days_since_first_use subtracted them; a naive value raised TypeError, both
caught it and returned {} / 0. Naive is read as UTC, like cleanup_old_sessions.
Offline: LocalAnalytics with list_sessions stubbed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.utils.local_analytics import LocalAnalytics


def _analytics(tmp_path) -> LocalAnalytics:
    return LocalAnalytics(
        TraigentConfig(
            execution_mode=ExecutionMode.LOCAL.value,
            enable_usage_analytics=True,
            local_storage_path=str(tmp_path),
            anonymous_user_id="test-user",
        )
    )


def _session(sid: str, created_at: str) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=sid,
        created_at=created_at,
        status="completed",
        completed_trials=3,
        function_name="f",
        optimization_config={},
    )


def test_naive_session_does_not_empty_usage_stats(tmp_path):
    analytics = _analytics(tmp_path)
    now = datetime.now(UTC)
    aware = _session("a", (now - timedelta(days=1)).isoformat())
    naive = _session("n", (now - timedelta(days=40)).replace(tzinfo=None).isoformat())
    with (
        patch.object(analytics.storage, "list_sessions", return_value=[aware, naive]),
        patch.object(analytics.storage, "get_session_summary", return_value=None),
    ):
        stats = analytics.collect_usage_stats()
        days = analytics._get_days_since_first_use()
    assert stats != {}
    assert stats["total_sessions"] == 2
    assert stats["sessions_last_30_days"] == 1
    assert days == 40
    assert stats["days_since_first_use"] == 40
