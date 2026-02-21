"""Unit tests for session management."""

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from traigent.cloud.models import (
    OptimizationSession,
    OptimizationSessionStatus,
    TrialResultSubmission,
    TrialStatus,
)
from traigent.cloud.sessions import InMemorySessionStorage, SessionManager
from traigent.utils.exceptions import SessionError


@pytest.fixture
def session_manager():
    """Create a session manager for testing."""
    return SessionManager(
        storage=InMemorySessionStorage(),
        session_timeout_hours=24,
        max_sessions_per_user=5,
    )


@pytest_asyncio.fixture
async def sample_session(session_manager):
    """Create a sample session."""
    session = await session_manager.create_session(
        function_name="test_function",
        configuration_space={"temperature": (0.0, 1.0), "model": ["gpt-3.5", "GPT-4o"]},
        objectives=["accuracy"],
        max_trials=50,
        metadata={"user_id": "test_user"},
    )
    return session


class TestInMemorySessionStorage:
    """Test in-memory session storage."""

    @pytest.mark.asyncio
    async def test_create_and_get_session(self):
        """Test creating and retrieving a session."""
        storage = InMemorySessionStorage()

        session = OptimizationSession(
            session_id="test-123",
            function_name="my_func",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=10,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        await storage.create(session)
        retrieved = await storage.get("test-123")

        assert retrieved is not None
        assert retrieved.session_id == "test-123"
        assert retrieved.function_name == "my_func"

    @pytest.mark.asyncio
    async def test_create_duplicate_session(self):
        """Test creating duplicate session raises error."""
        storage = InMemorySessionStorage()

        session = OptimizationSession(
            session_id="test-123",
            function_name="my_func",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=10,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        await storage.create(session)

        with pytest.raises(SessionError, match="already exists"):
            await storage.create(session)

    @pytest.mark.asyncio
    async def test_update_session(self):
        """Test updating a session."""
        storage = InMemorySessionStorage()

        session = OptimizationSession(
            session_id="test-123",
            function_name="my_func",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=10,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        await storage.create(session)

        # Update session
        session.completed_trials = 5
        session.status = OptimizationSessionStatus.COMPLETED
        await storage.update(session)

        retrieved = await storage.get("test-123")
        assert retrieved.completed_trials == 5
        assert retrieved.status == OptimizationSessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_delete_session(self):
        """Test deleting a session."""
        storage = InMemorySessionStorage()

        session = OptimizationSession(
            session_id="test-123",
            function_name="my_func",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=10,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        await storage.create(session)
        await storage.delete("test-123")

        retrieved = await storage.get("test-123")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_active_sessions(self):
        """Test listing active sessions."""
        storage = InMemorySessionStorage()

        # Create multiple sessions
        for i in range(3):
            session = OptimizationSession(
                session_id=f"test-{i}",
                function_name="my_func",
                configuration_space={},
                objectives=["accuracy"],
                max_trials=10,
                status=(
                    OptimizationSessionStatus.ACTIVE
                    if i < 2
                    else OptimizationSessionStatus.COMPLETED
                ),
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                metadata={"user_id": "user1" if i == 0 else "user2"},
            )
            await storage.create(session)

        # List all active sessions
        active = await storage.list_active()
        assert len(active) == 2

        # List active sessions for user1
        user1_active = await storage.list_active("user1")
        assert len(user1_active) == 1
        assert user1_active[0].session_id == "test-0"


class TestSessionManager:
    """Test session manager functionality."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test creating a new session."""
        session = await session_manager.create_session(
            function_name="optimize_llm",
            configuration_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy", "speed"],
            max_trials=100,
            metadata={"user_id": "test_user"},
        )

        assert session.session_id is not None
        assert session.function_name == "optimize_llm"
        assert session.max_trials == 100
        assert session.status == OptimizationSessionStatus.ACTIVE
        assert session.optimization_strategy is not None

    @pytest.mark.asyncio
    async def test_user_session_limit(self, session_manager):
        """Test user session limit enforcement."""
        # Create max sessions for a user
        for i in range(5):
            await session_manager.create_session(
                function_name=f"func_{i}",
                configuration_space={},
                objectives=["accuracy"],
                max_trials=10,
                metadata={"user_id": "limited_user"},
            )

        # Try to create one more
        with pytest.raises(SessionError, match="reached maximum"):
            await session_manager.create_session(
                function_name="func_extra",
                configuration_space={},
                objectives=["accuracy"],
                max_trials=10,
                metadata={"user_id": "limited_user"},
            )

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager, sample_session):
        """Test retrieving a session."""
        retrieved = await session_manager.get_session(sample_session.session_id)

        assert retrieved.session_id == sample_session.session_id
        assert retrieved.function_name == sample_session.function_name

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, session_manager):
        """Test retrieving non-existent session raises error."""
        with pytest.raises(SessionError, match="not found"):
            await session_manager.get_session("nonexistent-123")

    @pytest.mark.asyncio
    async def test_session_expiration(self, session_manager):
        """Test session expiration."""
        # Create a session
        session = await session_manager.create_session(
            function_name="test_func",
            configuration_space={},
            objectives=["accuracy"],
            max_trials=10,
        )

        # Directly modify the session in storage to bypass the update method
        # which always sets updated_at to now
        stored_session = await session_manager.storage.get(session.session_id)
        stored_session.updated_at = datetime.now(UTC) - timedelta(hours=25)

        # Directly update the storage without going through update() method
        async with session_manager.storage._lock:
            session_manager.storage._sessions[session.session_id] = stored_session

        # Verify the session is actually expired
        assert session_manager._is_expired(stored_session)

        # Try to get expired session
        with pytest.raises(SessionError, match="expired"):
            await session_manager.get_session(session.session_id)

    @pytest.mark.asyncio
    async def test_suggest_next_trial(self, session_manager, sample_session):
        """Test suggesting next trial."""
        suggestion = await session_manager.suggest_next_trial(
            sample_session.session_id, dataset_size=1000
        )

        assert suggestion is not None
        assert suggestion.session_id == sample_session.session_id
        assert suggestion.trial_number == 1
        assert "temperature" in suggestion.config
        assert "model" in suggestion.config
        assert len(suggestion.dataset_subset.indices) > 0
        assert suggestion.exploration_type in [
            "exploration",
            "exploitation",
            "verification",
        ]

    @pytest.mark.asyncio
    async def test_suggest_trial_after_max_trials(self, session_manager):
        """Test suggesting trial after max trials returns None."""
        session = await session_manager.create_session(
            function_name="test_func",
            configuration_space={"param": [1, 2, 3]},
            objectives=["accuracy"],
            max_trials=2,
        )

        # Complete max trials
        session.completed_trials = 2
        await session_manager.storage.update(session)

        suggestion = await session_manager.suggest_next_trial(
            session.session_id, dataset_size=100
        )

        assert suggestion is None

    @pytest.mark.asyncio
    async def test_submit_trial_result(self, session_manager, sample_session):
        """Test submitting trial results."""
        # Get a trial suggestion first
        suggestion = await session_manager.suggest_next_trial(
            sample_session.session_id, dataset_size=100
        )

        # Submit result
        result = TrialResultSubmission(
            session_id=sample_session.session_id,
            trial_id=suggestion.trial_id,
            metrics={"accuracy": 0.85, "speed": 0.92},
            duration=45.2,
            status=TrialStatus.COMPLETED,
        )

        await session_manager.submit_trial_result(result)

        # Check session was updated
        updated_session = await session_manager.get_session(sample_session.session_id)
        assert updated_session.completed_trials == 1
        assert updated_session.best_metrics == {"accuracy": 0.85, "speed": 0.92}

    @pytest.mark.asyncio
    async def test_submit_better_result(self, session_manager, sample_session):
        """Test submitting better results updates best metrics."""
        # Submit first result
        suggestion1 = await session_manager.suggest_next_trial(
            sample_session.session_id, dataset_size=100
        )

        result1 = TrialResultSubmission(
            session_id=sample_session.session_id,
            trial_id=suggestion1.trial_id,
            metrics={"accuracy": 0.80},
            duration=40.0,
            status=TrialStatus.COMPLETED,
        )

        await session_manager.submit_trial_result(result1)

        # Submit better result
        suggestion2 = await session_manager.suggest_next_trial(
            sample_session.session_id, dataset_size=100
        )

        result2 = TrialResultSubmission(
            session_id=sample_session.session_id,
            trial_id=suggestion2.trial_id,
            metrics={"accuracy": 0.90},
            duration=42.0,
            status=TrialStatus.COMPLETED,
        )

        await session_manager.submit_trial_result(result2)

        # Check best metrics updated
        session = await session_manager.get_session(sample_session.session_id)
        assert session.best_metrics["accuracy"] == 0.90

    @pytest.mark.asyncio
    async def test_submit_lower_value_wins_for_minimize_objective(self, session_manager):
        """For minimize objectives (e.g., latency), lower values should win."""
        session = await session_manager.create_session(
            function_name="latency_test",
            configuration_space={"temperature": (0.0, 1.0)},
            objectives=["latency"],
            max_trials=10,
        )

        suggestion1 = await session_manager.suggest_next_trial(session.session_id, 100)
        await session_manager.submit_trial_result(
            TrialResultSubmission(
                session_id=session.session_id,
                trial_id=suggestion1.trial_id,
                metrics={"latency": 120.0},
                duration=10.0,
                status=TrialStatus.COMPLETED,
            )
        )

        suggestion2 = await session_manager.suggest_next_trial(session.session_id, 100)
        await session_manager.submit_trial_result(
            TrialResultSubmission(
                session_id=session.session_id,
                trial_id=suggestion2.trial_id,
                metrics={"latency": 80.0},
                duration=10.0,
                status=TrialStatus.COMPLETED,
            )
        )

        updated = await session_manager.get_session(session.session_id)
        assert updated.best_metrics["latency"] == 80.0

        # Regressions guard: a worse (higher) latency must not replace best metrics.
        suggestion3 = await session_manager.suggest_next_trial(session.session_id, 100)
        await session_manager.submit_trial_result(
            TrialResultSubmission(
                session_id=session.session_id,
                trial_id=suggestion3.trial_id,
                metrics={"latency": 200.0},
                duration=10.0,
                status=TrialStatus.COMPLETED,
            )
        )

        updated_again = await session_manager.get_session(session.session_id)
        assert updated_again.best_metrics["latency"] == 80.0

    @pytest.mark.asyncio
    async def test_finalize_session(self, session_manager, sample_session):
        """Test finalizing a session."""
        # Run some trials
        for i in range(3):
            suggestion = await session_manager.suggest_next_trial(
                sample_session.session_id, dataset_size=100
            )

            result = TrialResultSubmission(
                session_id=sample_session.session_id,
                trial_id=suggestion.trial_id,
                metrics={"accuracy": 0.8 + i * 0.05},
                duration=40.0 + i,
                status=TrialStatus.COMPLETED,
            )

            await session_manager.submit_trial_result(result)

        # Finalize
        final_session = await session_manager.finalize_session(
            sample_session.session_id
        )

        assert final_session.status == OptimizationSessionStatus.COMPLETED
        assert final_session.completed_trials == 3
        assert final_session.best_metrics["accuracy"] == 0.90

    @pytest.mark.asyncio
    async def test_cancel_session(self, session_manager, sample_session):
        """Test cancelling a session."""
        await session_manager.cancel_session(sample_session.session_id)

        session = await session_manager.get_session(sample_session.session_id)
        assert session.status == OptimizationSessionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_trial_history(self, session_manager, sample_session):
        """Test getting trial history."""
        # Submit some results
        results = []
        for i in range(3):
            suggestion = await session_manager.suggest_next_trial(
                sample_session.session_id, dataset_size=100
            )

            result = TrialResultSubmission(
                session_id=sample_session.session_id,
                trial_id=suggestion.trial_id,
                metrics={"accuracy": 0.8 + i * 0.05},
                duration=40.0,
                status=TrialStatus.COMPLETED,
            )

            await session_manager.submit_trial_result(result)
            results.append(result)

        # Get history
        history = session_manager.get_trial_history(sample_session.session_id)

        assert len(history) == 3
        assert all(h.session_id == sample_session.session_id for h in history)

    @pytest.mark.asyncio
    async def test_dataset_subset_progression(self, session_manager, sample_session):
        """Test dataset subset size increases with trials."""
        dataset_size = 1000

        # Early trial - small subset
        suggestion1 = await session_manager.suggest_next_trial(
            sample_session.session_id, dataset_size=dataset_size
        )
        early_size = len(suggestion1.dataset_subset.indices)

        # Complete some trials
        for _i in range(5):
            suggestion = await session_manager.suggest_next_trial(
                sample_session.session_id, dataset_size=dataset_size
            )

            result = TrialResultSubmission(
                session_id=sample_session.session_id,
                trial_id=suggestion.trial_id,
                metrics={"accuracy": 0.8},
                duration=40.0,
                status=TrialStatus.COMPLETED,
            )

            await session_manager.submit_trial_result(result)

        # Later trial - larger subset
        suggestion_later = await session_manager.suggest_next_trial(
            sample_session.session_id, dataset_size=dataset_size
        )
        later_size = len(suggestion_later.dataset_subset.indices)

        assert later_size > early_size
        assert suggestion1.dataset_subset.selection_strategy == "diverse_sampling"
        assert (
            suggestion_later.dataset_subset.confidence_level
            > suggestion1.dataset_subset.confidence_level
        )
