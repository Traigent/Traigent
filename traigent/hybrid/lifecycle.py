"""Agent lifecycle management for Hybrid API mode.

Provides keep-alive heartbeat management for stateful external agents,
ensuring sessions remain active during optimization runs.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION LIFECYCLE-MANAGEMENT

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.hybrid.transport import HybridTransport

logger = get_logger(__name__)


@dataclass
class SessionInfo:
    """Information about an active session.

    Attributes:
        session_id: Unique session identifier
        created_at: Timestamp when session was created
        last_heartbeat: Timestamp of last successful heartbeat
        heartbeat_count: Number of successful heartbeats
        is_alive: Whether session is currently alive
        keep_alive_status: Last keep-alive status: alive, dead, or unsupported
    """

    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_heartbeat: float = 0.0
    heartbeat_count: int = 0
    is_alive: bool = True
    keep_alive_status: str = "alive"


class AgentLifecycleManager:
    """Manages external agent instance lifecycle with heartbeats.

    Provides keep-alive functionality for stateful agents that require
    periodic heartbeat signals to maintain session state.

    The manager runs a background task that periodically sends keep-alive
    signals to registered sessions. Sessions that fail heartbeats are
    marked as expired.

    Attributes:
        transport: HybridTransport for sending keep-alive signals
        heartbeat_interval: Seconds between heartbeats (default 30)
        max_missed_heartbeats: Max consecutive failures before marking dead (default 3)
    """

    def __init__(
        self,
        transport: HybridTransport,
        heartbeat_interval: float = 30.0,
        max_missed_heartbeats: int = 3,
    ) -> None:
        """Initialize lifecycle manager.

        Args:
            transport: HybridTransport for sending keep-alive signals.
            heartbeat_interval: Seconds between heartbeat attempts.
            max_missed_heartbeats: Maximum consecutive failures before
                marking a session as dead.
        """
        self._transport = transport
        self._heartbeat_interval = heartbeat_interval
        self._max_missed_heartbeats = max_missed_heartbeats

        self._sessions: dict[str, SessionInfo] = {}
        self._missed_heartbeats: dict[str, int] = {}
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._running = False

    @property
    def active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return [
            session_id for session_id, info in self._sessions.items() if info.is_alive
        ]

    @property
    def session_count(self) -> int:
        """Get count of tracked sessions (alive or dead)."""
        return len(self._sessions)

    def create_session(self) -> str:
        """Create a new session ID.

        Returns:
            New unique session ID.

        Note:
            This does not register with the external service.
            The session is registered when the first execute request
            returns a session_id.
        """
        session_id = str(uuid.uuid4())
        logger.debug(f"Created new session ID: {session_id}")
        return session_id

    async def register(self, session_id: str) -> None:  # NOSONAR(S7503)
        """Register a session for keep-alive management.

        Starts the background heartbeat task if not already running.

        Args:
            session_id: Session identifier to register.
        """
        if session_id in self._sessions:
            logger.debug(f"Session {session_id} already registered")
            return

        self._sessions[session_id] = SessionInfo(session_id=session_id)
        self._missed_heartbeats[session_id] = 0
        logger.info(f"Registered session for keep-alive: {session_id}")

        # Start heartbeat task if needed
        self._ensure_heartbeat_running()

    async def unregister(self, session_id: str) -> None:
        """Unregister a session from keep-alive management.

        Args:
            session_id: Session identifier to unregister.
        """
        if session_id not in self._sessions:
            return

        del self._sessions[session_id]
        self._missed_heartbeats.pop(session_id, None)
        logger.debug(f"Unregistered session: {session_id}")

        # Stop heartbeat task if no more sessions
        if not self._sessions and self._heartbeat_task:
            await self._stop_heartbeat_task()

    def _ensure_heartbeat_running(self) -> None:
        """Ensure heartbeat background task is running."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._shutdown_event.clear()
            self._running = True
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(),
                name="hybrid-lifecycle-heartbeat",
            )
            logger.debug("Started heartbeat background task")

    async def _stop_heartbeat_task(self) -> None:
        """Stop the heartbeat background task."""
        if self._heartbeat_task is not None:
            self._running = False
            self._shutdown_event.set()

            # Give task time to clean up
            try:
                await asyncio.wait_for(self._heartbeat_task, timeout=2.0)
            except TimeoutError:
                self._heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._heartbeat_task

            self._heartbeat_task = None
            logger.debug("Stopped heartbeat background task")

    async def _heartbeat_loop(self) -> None:
        """Background task that sends periodic heartbeats."""
        logger.debug(f"Heartbeat loop started (interval={self._heartbeat_interval}s)")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Wait for next heartbeat interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._heartbeat_interval,
                    )
                    # Shutdown event was set
                    break
                except TimeoutError:
                    # Normal timeout, proceed with heartbeat
                    pass

                # Send heartbeats to all active sessions
                await self._send_heartbeats()

            except asyncio.CancelledError:
                logger.debug("Heartbeat loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                # Continue running despite errors
                await asyncio.sleep(1.0)

        logger.debug("Heartbeat loop ended")

    async def _send_heartbeats(self) -> None:
        """Send heartbeats to all registered sessions."""
        # Get list of sessions to heartbeat (copy to avoid mutation during iteration)
        sessions_to_check = [
            (session_id, info)
            for session_id, info in self._sessions.items()
            if info.is_alive
        ]

        for session_id, info in sessions_to_check:
            # Check for shutdown between sessions for faster termination
            if self._shutdown_event.is_set():
                logger.debug("Shutdown requested, stopping heartbeats early")
                return

            try:
                alive = await self._transport.keep_alive(session_id)

                if alive:
                    info.last_heartbeat = asyncio.get_event_loop().time()
                    info.heartbeat_count += 1
                    info.keep_alive_status = "alive"
                    self._missed_heartbeats[session_id] = 0
                    logger.debug(
                        f"Heartbeat success for session {session_id} "
                        f"(count={info.heartbeat_count})"
                    )
                else:
                    self._handle_heartbeat_failure(session_id, info, "session expired")

            except NotImplementedError:
                # Keep-alive not supported by external service
                logger.debug(
                    f"Keep-alive not supported, skipping heartbeat for {session_id}"
                )
                info.is_alive = False
                info.keep_alive_status = "unsupported"
                self._missed_heartbeats[session_id] = 0
                return

            except Exception as e:
                self._handle_heartbeat_failure(session_id, info, str(e))

    def _handle_heartbeat_failure(
        self,
        session_id: str,
        info: SessionInfo,
        reason: str,
    ) -> None:
        """Handle a heartbeat failure.

        Args:
            session_id: Session that failed heartbeat.
            info: Session info to update.
            reason: Reason for failure.
        """
        self._missed_heartbeats[session_id] = (
            self._missed_heartbeats.get(session_id, 0) + 1
        )
        missed = self._missed_heartbeats[session_id]

        logger.warning(
            f"Heartbeat failed for session {session_id}: {reason} "
            f"(missed={missed}/{self._max_missed_heartbeats})"
        )

        if missed >= self._max_missed_heartbeats:
            info.is_alive = False
            info.keep_alive_status = "dead"
            logger.error(
                f"Session {session_id} marked as dead after {missed} missed heartbeats"
            )

    def get_session_info(self, session_id: str) -> SessionInfo | None:
        """Get info for a specific session.

        Args:
            session_id: Session to look up.

        Returns:
            SessionInfo if found, None otherwise.
        """
        return self._sessions.get(session_id)

    def is_session_alive(self, session_id: str) -> bool:
        """Check if a session is still alive.

        Args:
            session_id: Session to check.

        Returns:
            True if session exists and is alive, False otherwise.
        """
        info = self._sessions.get(session_id)
        return info.is_alive if info else False

    async def release(self) -> None:
        """Release all sessions and stop heartbeat task.

        Should be called when optimization is complete to clean up
        resources. Safe to call multiple times.
        """
        logger.info(f"Releasing lifecycle manager ({len(self._sessions)} sessions)")

        # Stop heartbeat task first
        await self._stop_heartbeat_task()

        # Clear all sessions
        self._sessions.clear()
        self._missed_heartbeats.clear()

        logger.debug("Lifecycle manager released")

    async def __aenter__(self) -> AgentLifecycleManager:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.release()
