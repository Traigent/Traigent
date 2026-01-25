"""Unit tests for Langfuse tracker.

Tests the LangfuseTracker class for Langfuse integration.
Run with: TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/langfuse/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.integrations.langfuse.client import LangfuseClient
from traigent.integrations.langfuse.tracker import (
    LangfuseTracker,
    create_langfuse_tracker,
)


class TestLangfuseTrackerInit:
    """Test LangfuseTracker initialization."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        return MagicMock(spec=LangfuseClient)

    @pytest.fixture
    def mock_resolver(self):
        """Create mock trace resolver."""
        return MagicMock(return_value="trace-123")

    def test_basic_init(self, mock_client, mock_resolver):
        """Test basic tracker initialization."""
        tracker = LangfuseTracker(
            client=mock_client,
            trace_id_resolver=mock_resolver,
        )
        assert tracker.client is mock_client

    def test_init_with_options(self, mock_client, mock_resolver):
        """Test tracker initialization with options."""
        tracker = LangfuseTracker(
            client=mock_client,
            trace_id_resolver=mock_resolver,
            metric_prefix="custom_",
            include_per_agent=False,
        )
        assert tracker._metric_prefix == "custom_"
        assert tracker._include_per_agent is False


class TestLangfuseTrackerClient:
    """Test client access."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        return MagicMock(spec=LangfuseClient)

    @pytest.fixture
    def mock_resolver(self):
        """Create mock trace resolver."""
        return MagicMock(return_value="trace-123")

    def test_client_property(self, mock_client, mock_resolver):
        """Test accessing client via property."""
        tracker = LangfuseTracker(
            client=mock_client,
            trace_id_resolver=mock_resolver,
        )
        assert tracker.client is mock_client


class TestLangfuseTrackerCallback:
    """Test callback creation."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        return MagicMock(spec=LangfuseClient)

    @pytest.fixture
    def mock_resolver(self):
        """Create mock trace resolver."""
        return MagicMock(return_value="trace-123")

    def test_get_callback(self, mock_client, mock_resolver):
        """Test getting callback."""
        tracker = LangfuseTracker(
            client=mock_client,
            trace_id_resolver=mock_resolver,
        )
        callback = tracker.get_callback()
        assert callback is not None

    def test_get_callback_returns_same_instance(self, mock_client, mock_resolver):
        """Test callback is cached."""
        tracker = LangfuseTracker(
            client=mock_client,
            trace_id_resolver=mock_resolver,
        )
        callback1 = tracker.get_callback()
        callback2 = tracker.get_callback()
        assert callback1 is callback2


class TestCreateLangfuseTracker:
    """Test the factory function."""

    def test_creates_tracker_with_env_vars(self, monkeypatch):
        """Test factory creates tracker with env vars."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        def resolver(trial):
            return f"trace-{trial.trial_id}"

        tracker = create_langfuse_tracker(trace_id_resolver=resolver)
        assert isinstance(tracker, LangfuseTracker)

    def test_creates_tracker_with_explicit_keys(self, monkeypatch):
        """Test factory creates tracker with explicit keys."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        def resolver(trial):
            return f"trace-{trial.trial_id}"

        tracker = create_langfuse_tracker(
            trace_id_resolver=resolver,
            public_key="pk-explicit",
            secret_key="sk-explicit",
        )
        assert isinstance(tracker, LangfuseTracker)
        assert tracker.client.public_key == "pk-explicit"

    def test_creates_tracker_with_custom_options(self, monkeypatch):
        """Test factory respects custom options."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        def resolver(trial):
            return None

        tracker = create_langfuse_tracker(
            trace_id_resolver=resolver,
            metric_prefix="lf_",
            include_per_agent=False,
        )
        assert tracker._metric_prefix == "lf_"
        assert tracker._include_per_agent is False
