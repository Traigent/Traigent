"""Unit tests for hybrid mode lifecycle management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.hybrid.lifecycle import AgentLifecycleManager, SessionInfo


class TestSessionInfo:
    """Tests for SessionInfo class."""

    def test_session_info_defaults(self) -> None:
        """Test SessionInfo default values."""
        info = SessionInfo(session_id="test")
        assert info.session_id == "test"
        assert info.is_alive is True
        assert info.heartbeat_count == 0


class TestAgentLifecycleManager:
    """Tests for AgentLifecycleManager."""

    @pytest.fixture
    def mock_transport(self) -> MagicMock:
        """Create mock transport for testing."""
        transport = MagicMock()
        transport.keep_alive = AsyncMock(return_value=True)
        transport.capabilities = AsyncMock(
            return_value=MagicMock(supports_keep_alive=True)
        )
        return transport

    @pytest.fixture
    def manager(self, mock_transport: MagicMock) -> AgentLifecycleManager:
        """Create lifecycle manager for testing."""
        return AgentLifecycleManager(
            transport=mock_transport,
            heartbeat_interval=0.1,  # Fast for tests
            max_missed_heartbeats=2,
        )

    def test_create_session(self, manager: AgentLifecycleManager) -> None:
        """Test creating a new session ID."""
        session_id = manager.create_session()
        assert session_id is not None
        assert len(session_id) > 0

    @pytest.mark.asyncio
    async def test_register_session(self, manager: AgentLifecycleManager) -> None:
        """Test registering a session."""
        session_id = "test_session"
        await manager.register(session_id)

        assert session_id in manager.active_sessions
        assert manager.session_count == 1

    @pytest.mark.asyncio
    async def test_unregister_session(self, manager: AgentLifecycleManager) -> None:
        """Test unregistering a session."""
        session_id = "test_session"
        await manager.register(session_id)
        await manager.unregister(session_id)

        assert session_id not in manager.active_sessions
        assert manager.session_count == 0

    @pytest.mark.asyncio
    async def test_duplicate_register(self, manager: AgentLifecycleManager) -> None:
        """Test registering same session twice."""
        session_id = "test_session"
        await manager.register(session_id)
        await manager.register(session_id)  # Should not raise

        assert manager.session_count == 1

    @pytest.mark.asyncio
    async def test_unregister_unknown_session(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Test unregistering unknown session."""
        await manager.unregister("unknown")  # Should not raise
        assert manager.session_count == 0

    def test_get_session_info(self, manager: AgentLifecycleManager) -> None:
        """Test getting session info."""
        info = manager.get_session_info("unknown")
        assert info is None

    @pytest.mark.asyncio
    async def test_get_session_info_registered(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Test getting info for registered session."""
        await manager.register("test")
        info = manager.get_session_info("test")

        assert info is not None
        assert info.session_id == "test"
        assert info.is_alive is True

    def test_is_session_alive_unknown(self, manager: AgentLifecycleManager) -> None:
        """Test checking unknown session."""
        assert manager.is_session_alive("unknown") is False

    @pytest.mark.asyncio
    async def test_is_session_alive_registered(
        self, manager: AgentLifecycleManager
    ) -> None:
        """Test checking registered session."""
        await manager.register("test")
        assert manager.is_session_alive("test") is True

    @pytest.mark.asyncio
    async def test_release(self, manager: AgentLifecycleManager) -> None:
        """Test releasing all sessions."""
        await manager.register("session1")
        await manager.register("session2")

        await manager.release()

        assert manager.session_count == 0
        assert len(manager.active_sessions) == 0

    @pytest.mark.asyncio
    async def test_release_idempotent(self, manager: AgentLifecycleManager) -> None:
        """Test releasing multiple times is safe."""
        await manager.register("test")
        await manager.release()
        await manager.release()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_transport: MagicMock) -> None:
        """Test async context manager."""
        async with AgentLifecycleManager(
            transport=mock_transport, heartbeat_interval=0.1
        ) as manager:
            await manager.register("test")
            assert manager.session_count == 1

        # After exit, should be released
        assert manager.session_count == 0

    @pytest.mark.asyncio
    async def test_heartbeat_updates_session(
        self,
        manager: AgentLifecycleManager,
        mock_transport: MagicMock,
    ) -> None:
        """Test heartbeat updates session info."""
        await manager.register("test")

        # Wait for at least one heartbeat cycle
        await asyncio.sleep(0.2)

        info = manager.get_session_info("test")
        assert info is not None
        # Heartbeat should have been called
        mock_transport.keep_alive.assert_called()

        await manager.release()

    @pytest.mark.asyncio
    async def test_failed_heartbeat_marks_dead(
        self,
        mock_transport: MagicMock,
    ) -> None:
        """Test that failed heartbeats mark session as dead."""
        mock_transport.keep_alive = AsyncMock(return_value=False)

        manager = AgentLifecycleManager(
            transport=mock_transport,
            heartbeat_interval=0.05,
            max_missed_heartbeats=2,
        )

        await manager.register("test")

        # Wait for multiple heartbeat failures
        await asyncio.sleep(0.3)

        info = manager.get_session_info("test")
        assert info is not None
        assert info.is_alive is False

        await manager.release()

    @pytest.mark.asyncio
    async def test_keep_alive_not_supported(
        self,
        mock_transport: MagicMock,
    ) -> None:
        """Test behavior when keep-alive not supported."""
        mock_transport.keep_alive = AsyncMock(side_effect=NotImplementedError)

        manager = AgentLifecycleManager(
            transport=mock_transport,
            heartbeat_interval=0.05,
        )

        await manager.register("test")
        await asyncio.sleep(0.1)

        # Session should still be considered alive
        info = manager.get_session_info("test")
        assert info is not None
        assert info.is_alive is True

        await manager.release()
