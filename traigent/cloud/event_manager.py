"""Event Management for Traigent SDK.

Focused component responsible for lifecycle event recording and management,
extracted from the SessionLifecycleManager to follow Single Responsibility Principle.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.validation import CoreValidators, validate_or_raise

logger = get_logger(__name__)


class LifecycleEventType(Enum):
    """Types of lifecycle events."""

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    SESSION_CANCELLED = "session_cancelled"

    TRIAL_SUGGESTED = "trial_suggested"
    TRIAL_STARTED = "trial_started"
    TRIAL_COMPLETED = "trial_completed"
    TRIAL_FAILED = "trial_failed"
    TRIAL_SKIPPED = "trial_skipped"

    SYNC_STARTED = "sync_started"
    SYNC_COMPLETED = "sync_completed"
    SYNC_FAILED = "sync_failed"

    # Additional event types for comprehensive tracking
    CONFIG_UPDATED = "config_updated"
    METRICS_RECORDED = "metrics_recorded"
    ERROR_OCCURRED = "error_occurred"
    WARNING_ISSUED = "warning_issued"


@dataclass
class LifecycleEvent:
    """Represents a lifecycle event."""

    event_type: LifecycleEventType
    session_id: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    trial_id: str | None = None
    error_message: str | None = None
    severity: str = "info"  # info, warning, error, critical

    def get_summary(self) -> dict[str, Any]:
        """Get event summary for logging/display."""
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "has_data": bool(self.data),
            "has_error": bool(self.error_message),
        }

    def __str__(self) -> str:
        """String representation of event."""
        parts = [f"{self.event_type.value}"]

        if self.trial_id:
            parts.append(f"trial:{self.trial_id}")

        parts.append(f"session:{self.session_id}")

        if self.error_message:
            parts.append(f"error:{self.error_message[:50]}...")

        return f"LifecycleEvent({', '.join(parts)})"


class EventFilter:
    """Filters events based on criteria."""

    def __init__(
        self,
        event_types: list[LifecycleEventType] | None = None,
        session_ids: list[str] | None = None,
        trial_ids: list[str] | None = None,
        severity_levels: list[str] | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> None:
        """Initialize event filter.

        Args:
            event_types: Filter by event types
            session_ids: Filter by session IDs
            trial_ids: Filter by trial IDs
            severity_levels: Filter by severity levels
            time_range: Filter by time range (start, end)
        """
        self.event_types = set(event_types) if event_types else None
        self.session_ids = set(session_ids) if session_ids else None
        self.trial_ids = set(trial_ids) if trial_ids else None
        self.severity_levels = set(severity_levels) if severity_levels else None
        self.time_range = time_range

    def matches(self, event: LifecycleEvent) -> bool:
        """Check if event matches filter criteria.

        Args:
            event: Event to check

        Returns:
            True if event matches filter
        """
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check session ID
        if self.session_ids and event.session_id not in self.session_ids:
            return False

        # Check trial ID
        if self.trial_ids and event.trial_id not in self.trial_ids:
            return False

        # Check severity
        if self.severity_levels and event.severity not in self.severity_levels:
            return False

        # Check time range
        if self.time_range:
            start_time, end_time = self.time_range
            if not (start_time <= event.timestamp <= end_time):
                return False

        return True


class EventManager:
    """Manages lifecycle events with efficient storage and retrieval."""

    def __init__(
        self,
        max_events_total: int = 100000,
        max_events_per_session: int = 1000,
        enable_persistence: bool = False,
    ) -> None:
        """Initialize event manager.

        Args:
            max_events_total: Maximum total events to store
            max_events_per_session: Maximum events per session
            enable_persistence: Enable event persistence to disk
        """
        validate_or_raise(
            CoreValidators.validate_positive_int(max_events_total, "max_events_total")
        )
        validate_or_raise(
            CoreValidators.validate_positive_int(
                max_events_per_session, "max_events_per_session"
            )
        )
        if max_events_per_session > max_events_total:
            raise ValidationException(
                "max_events_per_session cannot exceed max_events_total"
            )

        self.max_events_total = max_events_total
        self.max_events_per_session = max_events_per_session
        self.enable_persistence = enable_persistence

        # Event storage
        self._events: deque[LifecycleEvent] = deque(maxlen=max_events_total)
        self._events_by_session: dict[str, deque[LifecycleEvent]] = {}
        self._event_handlers: dict[LifecycleEventType, list[Callable[..., Any]]] = {}

        # Event indexing for fast retrieval
        self._events_by_type: dict[LifecycleEventType, list[LifecycleEvent]] = {}
        self._events_by_trial: dict[str, list[LifecycleEvent]] = {}

        # Statistics
        self._stats: dict[str, Any] = {
            "total_events_recorded": 0,
            "events_by_type": {},
            "events_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0},
            "handler_invocations": 0,
            "handler_failures": 0,
        }

    def record_event(
        self,
        event_type: LifecycleEventType,
        session_id: str,
        data: dict[str, Any] | None = None,
        trial_id: str | None = None,
        error_message: str | None = None,
        severity: str = "info",
    ) -> LifecycleEvent:
        """Record a new lifecycle event.

        Args:
            event_type: Type of event
            session_id: Session ID
            data: Optional event data
            trial_id: Optional trial ID
            error_message: Optional error message
            severity: Event severity level

        Returns:
            Created lifecycle event
        """
        # Validate inputs
        validate_or_raise(
            CoreValidators.validate_string_non_empty(session_id, "session_id")
        )

        if trial_id:
            validate_or_raise(
                CoreValidators.validate_string_non_empty(trial_id, "trial_id")
            )

        severity_levels = ["info", "warning", "error", "critical"]
        validate_or_raise(
            CoreValidators.validate_choices(severity, "severity", severity_levels)
        )

        # Create event
        event = LifecycleEvent(
            event_type=event_type,
            session_id=session_id,
            timestamp=time.time(),
            data=data.copy() if data else {},
            trial_id=trial_id,
            error_message=error_message,
            severity=severity,
        )

        # Store event
        self._store_event(event)

        # Update statistics
        self._update_stats(event)

        # Trigger event handlers
        self._trigger_event_handlers(event)

        # Log event
        self._log_event(event)

        return event

    def _store_event(self, event: LifecycleEvent) -> None:
        """Store event in internal data structures."""
        # Add to main event queue
        self._events.append(event)

        # Add to session-specific queue
        if event.session_id not in self._events_by_session:
            self._events_by_session[event.session_id] = deque(
                maxlen=self.max_events_per_session
            )

        self._events_by_session[event.session_id].append(event)

        # Update indexes
        if event.event_type not in self._events_by_type:
            self._events_by_type[event.event_type] = []

        self._events_by_type[event.event_type].append(event)

        if event.trial_id:
            if event.trial_id not in self._events_by_trial:
                self._events_by_trial[event.trial_id] = []

            self._events_by_trial[event.trial_id].append(event)

        # Cleanup old indexes if needed
        self._cleanup_indexes()

    def _update_stats(self, event: LifecycleEvent) -> None:
        """Update event statistics."""
        self._stats["total_events_recorded"] += 1

        # Update by type
        event_type_str = event.event_type.value
        if event_type_str not in self._stats["events_by_type"]:
            self._stats["events_by_type"][event_type_str] = 0
        self._stats["events_by_type"][event_type_str] += 1

        # Update by severity
        self._stats["events_by_severity"][event.severity] += 1

    def _trigger_event_handlers(self, event: LifecycleEvent) -> None:
        """Trigger registered event handlers."""
        handlers = self._event_handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                handler(event)
                self._stats["handler_invocations"] += 1
            except Exception as e:
                self._stats["handler_failures"] += 1
                logger.error(f"Event handler failed for {event.event_type}: {e}")

    def _log_event(self, event: LifecycleEvent) -> None:
        """Log event based on severity."""
        message = f"Event {event.event_type.value} for session {event.session_id}"

        if event.trial_id:
            message += f" trial {event.trial_id}"

        if event.error_message:
            message += f": {event.error_message}"

        if event.severity == "critical":
            logger.critical(message)
        elif event.severity == "error":
            logger.error(message)
        elif event.severity == "warning":
            logger.warning(message)
        else:
            logger.debug(message)

    def get_events(
        self,
        filter_criteria: EventFilter | None = None,
        limit: int | None = None,
        reverse: bool = True,
    ) -> list[LifecycleEvent]:
        """Get events with optional filtering.

        Args:
            filter_criteria: Optional event filter
            limit: Maximum number of events to return
            reverse: Return newest events first

        Returns:
            List of events matching criteria
        """
        # Get base event list
        events = list(self._events)

        # Apply filter
        if filter_criteria:
            events = [event for event in events if filter_criteria.matches(event)]

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp, reverse=reverse)

        # Apply limit
        if limit:
            events = events[:limit]

        return events

    def get_session_events(
        self,
        session_id: str,
        event_types: list[LifecycleEventType] | None = None,
        limit: int | None = None,
    ) -> list[LifecycleEvent]:
        """Get events for a specific session.

        Args:
            session_id: Session ID
            event_types: Optional event types to filter
            limit: Maximum number of events to return

        Returns:
            List of events for the session
        """
        session_events = list(self._events_by_session.get(session_id, []))

        # Filter by event types
        if event_types:
            event_type_set = set(event_types)
            session_events = [
                event for event in session_events if event.event_type in event_type_set
            ]

        # Sort by timestamp (newest first)
        session_events.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        if limit:
            session_events = session_events[:limit]

        return session_events

    def get_trial_events(self, trial_id: str) -> list[LifecycleEvent]:
        """Get events for a specific trial.

        Args:
            trial_id: Trial ID

        Returns:
            List of events for the trial
        """
        trial_events = self._events_by_trial.get(trial_id, [])
        return sorted(trial_events, key=lambda e: e.timestamp)

    def get_recent_errors(
        self, session_id: str | None = None, hours: int = 24, limit: int = 100
    ) -> list[LifecycleEvent]:
        """Get recent error events.

        Args:
            session_id: Optional session ID to filter
            hours: Hours back to look for errors
            limit: Maximum number of errors to return

        Returns:
            List of recent error events
        """
        cutoff_time = time.time() - (hours * 3600)

        filter_criteria = EventFilter(
            severity_levels=["error", "critical"],
            time_range=(cutoff_time, time.time()),
            session_ids=[session_id] if session_id else None,
        )

        return self.get_events(filter_criteria, limit=limit)

    def add_event_handler(
        self, event_type: LifecycleEventType, handler: Callable[[LifecycleEvent], None]
    ) -> None:
        """Add event handler for specific event type.

        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []

        self._event_handlers[event_type].append(handler)
        logger.debug(f"Added event handler for {event_type}")

    def remove_event_handler(
        self, event_type: LifecycleEventType, handler: Callable[[LifecycleEvent], None]
    ) -> bool:
        """Remove event handler.

        Args:
            event_type: Event type
            handler: Handler function to remove

        Returns:
            True if handler was removed
        """
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
                logger.debug(f"Removed event handler for {event_type}")
                return True
            except ValueError:
                pass

        return False

    def clear_session_events(self, session_id: str) -> int:
        """Clear all events for a session.

        Args:
            session_id: Session ID

        Returns:
            Number of events cleared
        """
        session_events = self._events_by_session.get(session_id)
        if session_events is None or len(session_events) == 0:
            return 0

        event_count = len(session_events)
        del self._events_by_session[session_id]

        # Rebuild global event storage without the session's events
        self._events = deque(
            (event for event in self._events if event.session_id != session_id),
            maxlen=self.max_events_total,
        )

        # Remove from type index
        for event_type in list(self._events_by_type.keys()):
            filtered_events = [
                event
                for event in self._events_by_type[event_type]
                if event.session_id != session_id
            ]
            if filtered_events:
                self._events_by_type[event_type] = filtered_events
            else:
                del self._events_by_type[event_type]

        # Remove from trial index
        for trial_id in list(self._events_by_trial.keys()):
            filtered_events = [
                event
                for event in self._events_by_trial[trial_id]
                if event.session_id != session_id
            ]
            if filtered_events:
                self._events_by_trial[trial_id] = filtered_events
            else:
                del self._events_by_trial[trial_id]

        if event_count > 0:
            logger.info(f"Cleared {event_count} events for session {session_id}")

        return event_count

    def _cleanup_indexes(self) -> None:
        """Clean up event indexes when they get too large."""
        # Cleanup type index (keep last 10000 events per type)
        for event_type in self._events_by_type:
            events = self._events_by_type[event_type]
            if len(events) > 10000:
                self._events_by_type[event_type] = events[-10000:]

        # Cleanup trial index (keep last 1000 events per trial)
        for trial_id in list(self._events_by_trial.keys()):
            events = self._events_by_trial[trial_id]
            if len(events) > 1000:
                self._events_by_trial[trial_id] = events[-1000:]

    def get_statistics(self) -> dict[str, Any]:
        """Get event manager statistics."""
        return {
            **self._stats,
            "total_events_stored": len(self._events),
            "sessions_with_events": len(self._events_by_session),
            "event_types_tracked": len(self._events_by_type),
            "trials_with_events": len(self._events_by_trial),
            "active_handlers": sum(
                len(handlers) for handlers in self._event_handlers.values()
            ),
        }
