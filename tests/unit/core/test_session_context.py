"""Comprehensive tests for traigent.core.session_context module.

Tests cover SessionContext dataclass for backend session management.
"""

from __future__ import annotations

import time

from traigent.core.session_context import SessionContext


class TestSessionContext:
    """Test SessionContext dataclass."""

    def test_basic_creation(self):
        """Test basic SessionContext creation."""
        start = time.time()
        context = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_module.my_function",
            optimization_id="opt_456",
            start_time=start,
        )

        assert context.session_id == "session_123"
        assert context.dataset_name == "test_dataset.jsonl"
        assert context.function_name == "my_module.my_function"
        assert context.optimization_id == "opt_456"
        assert context.start_time == start

    def test_no_session_id(self):
        """Test SessionContext without session ID (backend disabled)."""
        context = SessionContext(
            session_id=None,
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_789",
            start_time=time.time(),
        )

        assert context.session_id is None

    def test_no_function_name(self):
        """Test SessionContext without function name."""
        context = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name=None,
            optimization_id="opt_999",
            start_time=time.time(),
        )

        assert context.function_name is None

    def test_attribute_access(self):
        """Test all attributes are accessible."""
        start = time.time()
        context = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_123",
            start_time=start,
        )

        assert hasattr(context, "session_id")
        assert hasattr(context, "dataset_name")
        assert hasattr(context, "function_name")
        assert hasattr(context, "optimization_id")
        assert hasattr(context, "start_time")

    def test_different_dataset_names(self):
        """Test contexts with different dataset names."""
        start = time.time()
        context1 = SessionContext(
            session_id="session_1",
            dataset_name="dataset1.jsonl",
            function_name="func1",
            optimization_id="opt_1",
            start_time=start,
        )

        context2 = SessionContext(
            session_id="session_1",
            dataset_name="dataset2.jsonl",
            function_name="func1",
            optimization_id="opt_2",
            start_time=start,
        )

        assert context1.dataset_name == "dataset1.jsonl"
        assert context2.dataset_name == "dataset2.jsonl"

    def test_different_optimization_ids(self):
        """Test contexts with different optimization IDs."""
        start = time.time()
        context1 = SessionContext(
            session_id="session_1",
            dataset_name="dataset.jsonl",
            function_name="func",
            optimization_id="opt_123",
            start_time=start,
        )

        context2 = SessionContext(
            session_id="session_1",
            dataset_name="dataset.jsonl",
            function_name="func",
            optimization_id="opt_456",
            start_time=start,
        )

        assert context1.optimization_id == "opt_123"
        assert context2.optimization_id == "opt_456"

    def test_dataclass_equality(self):
        """Test dataclass equality comparison."""
        start = time.time()
        context1 = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_456",
            start_time=start,
        )

        context2 = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_456",
            start_time=start,
        )

        # Dataclasses with same values should be equal
        assert context1 == context2

    def test_dataclass_inequality(self):
        """Test dataclass inequality with different values."""
        start = time.time()
        context1 = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_456",
            start_time=start,
        )

        context2 = SessionContext(
            session_id="session_456",  # Different session ID
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_456",
            start_time=start,
        )

        assert context1 != context2

    def test_start_time_timestamp(self):
        """Test start_time is a valid timestamp."""
        start = time.time()
        context = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_456",
            start_time=start,
        )

        assert isinstance(context.start_time, float)
        assert context.start_time > 0
        assert context.start_time <= time.time()

    def test_different_start_times(self):
        """Test contexts with different start times."""
        start1 = time.time()
        time.sleep(0.01)  # Small delay
        start2 = time.time()

        context1 = SessionContext(
            session_id="session_1",
            dataset_name="dataset.jsonl",
            function_name="func",
            optimization_id="opt_1",
            start_time=start1,
        )

        context2 = SessionContext(
            session_id="session_1",
            dataset_name="dataset.jsonl",
            function_name="func",
            optimization_id="opt_1",
            start_time=start2,
        )

        assert context1.start_time < context2.start_time

    def test_fully_qualified_function_name(self):
        """Test context with fully qualified function name."""
        context = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="package.module.submodule.function",
            optimization_id="opt_456",
            start_time=time.time(),
        )

        assert "." in context.function_name
        assert context.function_name.count(".") == 3

    def test_dataset_name_variations(self):
        """Test various dataset name formats."""
        start = time.time()

        # JSON format
        context1 = SessionContext(
            session_id="s1",
            dataset_name="data.json",
            function_name="func",
            optimization_id="opt_1",
            start_time=start,
        )

        # JSONL format
        context2 = SessionContext(
            session_id="s2",
            dataset_name="data.jsonl",
            function_name="func",
            optimization_id="opt_2",
            start_time=start,
        )

        # CSV format
        context3 = SessionContext(
            session_id="s3",
            dataset_name="data.csv",
            function_name="func",
            optimization_id="opt_3",
            start_time=start,
        )

        assert context1.dataset_name.endswith(".json")
        assert context2.dataset_name.endswith(".jsonl")
        assert context3.dataset_name.endswith(".csv")

    def test_empty_string_values(self):
        """Test context with empty string values."""
        context = SessionContext(
            session_id="",
            dataset_name="",
            function_name="",
            optimization_id="",
            start_time=time.time(),
        )

        assert context.session_id == ""
        assert context.dataset_name == ""
        assert context.function_name == ""
        assert context.optimization_id == ""

    def test_repr_string(self):
        """Test string representation of context."""
        start = time.time()
        context = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_456",
            start_time=start,
        )

        repr_str = repr(context)
        assert "SessionContext" in repr_str
        assert "session_id=" in repr_str
        assert "optimization_id=" in repr_str

    def test_multiple_contexts_different_sessions(self):
        """Test multiple contexts for different sessions."""
        start = time.time()
        contexts = [
            SessionContext(
                session_id=f"session_{i}",
                dataset_name=f"dataset_{i}.jsonl",
                function_name=f"function_{i}",
                optimization_id=f"opt_{i}",
                start_time=start,
            )
            for i in range(5)
        ]

        assert len(contexts) == 5
        assert all(ctx.session_id.startswith("session_") for ctx in contexts)
        assert all(ctx.optimization_id.startswith("opt_") for ctx in contexts)

    def test_zero_start_time(self):
        """Test context with zero start time."""
        context = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_456",
            start_time=0.0,
        )

        assert context.start_time == 0.0

    def test_context_immutability_concept(self):
        """Test that context fields can be reassigned (dataclass not frozen)."""
        start = time.time()
        context = SessionContext(
            session_id="session_123",
            dataset_name="test_dataset.jsonl",
            function_name="my_function",
            optimization_id="opt_456",
            start_time=start,
        )

        # Dataclass is not frozen, so fields can be reassigned
        context.session_id = "new_session_456"
        assert context.session_id == "new_session_456"
