"""Tests for EventManager invariants and storage hygiene."""

import pytest

from traigent.cloud.event_manager import EventManager, LifecycleEventType
from traigent.utils.exceptions import ValidationError as ValidationException


class TestEventManagerValidation:
    """Validation scenarios for constructor parameters."""

    def test_invalid_max_events_rejected(self):
        """Non-positive capacities should raise validation errors."""
        with pytest.raises(ValidationException):
            EventManager(max_events_total=0)

        with pytest.raises(ValidationException):
            EventManager(max_events_total=10, max_events_per_session=0)

    def test_per_session_cannot_exceed_total(self):
        """Per-session cap must never exceed the global capacity."""
        with pytest.raises(ValidationException):
            EventManager(max_events_total=10, max_events_per_session=20)


class TestEventManagerClearing:
    """Ensure session clearing purges all storage layers."""

    def test_clear_session_events_purges_indexes(self):
        manager = EventManager(max_events_total=10, max_events_per_session=5)

        # Record events across two sessions to ensure isolation
        manager.record_event(
            LifecycleEventType.SESSION_STARTED,
            session_id="session-a",
            trial_id="trial-1",
        )
        manager.record_event(
            LifecycleEventType.TRIAL_COMPLETED,
            session_id="session-a",
            trial_id="trial-1",
            severity="warning",
        )
        manager.record_event(
            LifecycleEventType.SESSION_STARTED,
            session_id="session-b",
            trial_id="trial-2",
        )

        removed = manager.clear_session_events("session-a")
        assert removed == 2

        # Session mappings should be removed entirely
        assert "session-a" not in manager._events_by_session  # noqa: SLF001
        assert all(
            event.session_id != "session-a" for event in manager._events
        )  # noqa: SLF001

        # Type and trial indexes should no longer reference the session
        for events in manager._events_by_type.values():  # noqa: SLF001
            assert all(event.session_id != "session-a" for event in events)

        assert "trial-1" not in manager._events_by_trial  # noqa: SLF001

        # Other sessions remain intact
        assert any(
            event.session_id == "session-b" for event in manager._events
        )  # noqa: SLF001

    def test_clear_session_events_noop_for_unknown_session(self):
        manager = EventManager()
        assert manager.clear_session_events("missing") == 0
