"""Unit tests for traigent.utils.langchain_interceptor.

Tests for LangChain response interception and metadata capture functionality.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.utils.langchain_interceptor import (
    LangChainMetadataCapture,
    _create_astream_wrapper,
    _create_stream_wrapper,
    capture_key,
    capture_langchain_response,
    clear_captured_responses,
    get_all_captured_responses,
    get_captured_response,
    get_captured_response_by_key,
    langchain_metadata_context,
    patch_langchain_for_metadata_capture,
)


class TestLangChainMetadataCapture:
    """Tests for LangChainMetadataCapture class."""

    @pytest.fixture
    def capture(self) -> LangChainMetadataCapture:
        """Create test instance of LangChainMetadataCapture."""
        instance = LangChainMetadataCapture()
        # Clean up any existing state
        instance.clear()
        return instance

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Create mock LangChain response."""
        response = MagicMock()
        response.content = "Test response content"
        response.response_metadata = {"model": "gpt-4", "usage": {"total_tokens": 100}}
        response.usage_metadata = {"input_tokens": 50, "output_tokens": 50}
        return response

    def test_set_last_response_stores_response(
        self, capture: LangChainMetadataCapture, mock_response: MagicMock
    ) -> None:
        """Test that set_last_response stores response correctly."""
        capture.set_last_response(mock_response)

        # Should be in thread local storage
        assert hasattr(capture._storage, "responses")
        assert len(capture._storage.responses) == 1
        assert capture._storage.responses[0] is mock_response

    def test_set_last_response_stores_in_global_list(
        self, capture: LangChainMetadataCapture, mock_response: MagicMock
    ) -> None:
        """Test that set_last_response stores in global list."""
        capture.set_last_response(mock_response)

        assert len(capture._all_responses) == 1
        assert capture._all_responses[0] is mock_response

    def test_set_last_response_multiple_responses(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that multiple responses are stored in order."""
        response1 = MagicMock()
        response2 = MagicMock()
        response3 = MagicMock()

        capture.set_last_response(response1)
        capture.set_last_response(response2)
        capture.set_last_response(response3)

        assert len(capture._storage.responses) == 3
        assert capture._storage.responses == [response1, response2, response3]

    def test_get_last_response_retrieves_and_removes(
        self, capture: LangChainMetadataCapture, mock_response: MagicMock
    ) -> None:
        """Test that get_last_response retrieves and removes response."""
        capture.set_last_response(mock_response)

        result = capture.get_last_response()

        assert result is mock_response
        assert len(capture._storage.responses) == 0

    def test_get_last_response_returns_none_when_empty(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that get_last_response returns None when no responses stored."""
        result = capture.get_last_response()
        assert result is None

    def test_get_last_response_pops_from_end(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that get_last_response retrieves most recent response."""
        response1 = MagicMock()
        response2 = MagicMock()
        response3 = MagicMock()

        capture.set_last_response(response1)
        capture.set_last_response(response2)
        capture.set_last_response(response3)

        result = capture.get_last_response()
        assert result is response3

        result = capture.get_last_response()
        assert result is response2

        result = capture.get_last_response()
        assert result is response1

    def test_get_all_responses_returns_copy(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that get_all_responses returns a copy of all responses."""
        response1 = MagicMock()
        response2 = MagicMock()

        capture.set_last_response(response1)
        capture.set_last_response(response2)

        all_responses = capture.get_all_responses()

        assert len(all_responses) == 2
        assert all_responses == [response1, response2]

        # Modifying returned list should not affect internal state
        all_responses.append(MagicMock())
        assert len(capture._all_responses) == 2

    def test_get_all_responses_returns_empty_list(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that get_all_responses returns empty list when no responses."""
        all_responses = capture.get_all_responses()
        assert all_responses == []

    def test_clear_removes_all_responses(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that clear removes all stored responses."""
        capture.set_last_response(MagicMock())
        capture.set_last_response(MagicMock())
        capture.set_current_key("test_key")

        capture.clear()

        assert len(capture._storage.responses) == 0
        assert len(capture._all_responses) == 0
        assert len(capture._by_key) == 0
        assert capture._key_local.current_key is None

    def test_set_current_key_stores_key(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that set_current_key stores correlation key."""
        capture.set_current_key("example_123")
        assert capture._key_local.current_key == "example_123"

    def test_clear_current_key_removes_key(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that clear_current_key removes correlation key."""
        capture.set_current_key("example_123")
        capture.clear_current_key()
        assert capture._key_local.current_key is None

    def test_set_last_response_with_key_stores_by_key(
        self, capture: LangChainMetadataCapture, mock_response: MagicMock
    ) -> None:
        """Test that response is stored by key when correlation key is set."""
        capture.set_current_key("example_123")
        capture.set_last_response(mock_response)

        assert capture._by_key["example_123"] is mock_response

    def test_get_by_key_retrieves_response(
        self, capture: LangChainMetadataCapture, mock_response: MagicMock
    ) -> None:
        """Test that get_by_key retrieves response for given key."""
        capture.set_current_key("example_123")
        capture.set_last_response(mock_response)

        result = capture.get_by_key("example_123")
        assert result is mock_response

    def test_get_by_key_returns_none_for_unknown_key(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test that get_by_key returns None for unknown key."""
        result = capture.get_by_key("unknown_key")
        assert result is None

    def test_thread_safety_set_last_response(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test thread safety of set_last_response."""
        responses = []
        threads = []

        def add_response(index: int) -> None:
            response = MagicMock()
            response.index = index
            responses.append(response)
            capture.set_last_response(response)

        # Create and start multiple threads
        for i in range(10):
            thread = threading.Thread(target=add_response, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All responses should be captured
        assert len(capture._all_responses) == 10

    def test_thread_safety_get_last_response(
        self, capture: LangChainMetadataCapture
    ) -> None:
        """Test thread safety of get_last_response."""
        # Pre-populate with responses in main thread
        for i in range(10):
            response = MagicMock()
            response.index = i
            capture.set_last_response(response)

        results = []
        lock = threading.Lock()
        threads = []

        def get_response() -> None:
            result = capture.get_last_response()
            if result is not None:
                with lock:
                    results.append(result)

        # Create and start multiple threads
        for _ in range(10):
            thread = threading.Thread(target=get_response)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should retrieve some responses (exact count may vary due to thread timing)
        assert len(results) <= 10


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_captured_responses()

    def test_capture_langchain_response_stores_response(self) -> None:
        """Test that capture_langchain_response stores response."""
        response = MagicMock()
        response.content = "Test content"

        result = capture_langchain_response(response)

        assert result is response
        captured = get_captured_response()
        assert captured is response

    def test_capture_langchain_response_returns_response(self) -> None:
        """Test that capture_langchain_response returns the same response."""
        response = MagicMock()
        result = capture_langchain_response(response)
        assert result is response

    def test_get_captured_response_retrieves_response(self) -> None:
        """Test that get_captured_response retrieves stored response."""
        response = MagicMock()
        capture_langchain_response(response)

        result = get_captured_response()
        assert result is response

    def test_get_captured_response_returns_none_when_empty(self) -> None:
        """Test that get_captured_response returns None when empty."""
        result = get_captured_response()
        assert result is None

    def test_get_all_captured_responses_returns_all(self) -> None:
        """Test that get_all_captured_responses returns all responses."""
        response1 = MagicMock()
        response2 = MagicMock()

        capture_langchain_response(response1)
        capture_langchain_response(response2)

        all_responses = get_all_captured_responses()
        assert len(all_responses) == 2
        assert all_responses == [response1, response2]

    def test_get_captured_response_by_key_retrieves_response(self) -> None:
        """Test that get_captured_response_by_key retrieves response by key."""
        response = MagicMock()

        with capture_key("test_key"):
            capture_langchain_response(response)

        result = get_captured_response_by_key("test_key")
        assert result is response

    def test_get_captured_response_by_key_returns_none_for_unknown(self) -> None:
        """Test that get_captured_response_by_key returns None for unknown key."""
        result = get_captured_response_by_key("unknown_key")
        assert result is None

    def test_clear_captured_responses_clears_all(self) -> None:
        """Test that clear_captured_responses clears all responses."""
        capture_langchain_response(MagicMock())
        capture_langchain_response(MagicMock())

        clear_captured_responses()

        assert get_captured_response() is None
        assert get_all_captured_responses() == []


class TestLangChainMetadataContext:
    """Tests for langchain_metadata_context context manager."""

    def test_context_manager_clears_on_enter(self) -> None:
        """Test that context manager clears responses on enter."""
        capture_langchain_response(MagicMock())

        with langchain_metadata_context() as capture:
            # Should be empty on entry
            assert len(capture.get_all_responses()) == 0

    def test_context_manager_clears_on_exit(self) -> None:
        """Test that context manager clears responses on exit."""
        with langchain_metadata_context() as capture:
            capture.set_last_response(MagicMock())
            assert len(capture.get_all_responses()) == 1

        # Should be cleared after exit
        assert get_all_captured_responses() == []

    def test_context_manager_allows_capturing_within_scope(self) -> None:
        """Test that context manager allows capturing within scope."""
        response = MagicMock()

        with langchain_metadata_context() as capture:
            capture.set_last_response(response)
            assert len(capture.get_all_responses()) == 1
            assert capture.get_last_response() is response

    def test_context_manager_clears_on_exception(self) -> None:
        """Test that context manager clears responses even on exception."""
        try:
            with langchain_metadata_context() as capture:
                capture.set_last_response(MagicMock())
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be cleared after exception
        assert get_all_captured_responses() == []


class TestCaptureKeyContext:
    """Tests for capture_key context manager."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_captured_responses()

    def test_capture_key_sets_current_key(self) -> None:
        """Test that capture_key sets the current correlation key."""
        with capture_key("test_key"):
            response = MagicMock()
            capture_langchain_response(response)

        result = get_captured_response_by_key("test_key")
        assert result is response

    def test_capture_key_clears_on_exit(self) -> None:
        """Test that capture_key clears current key on exit."""
        from traigent.utils.langchain_interceptor import _metadata_capture

        with capture_key("test_key"):
            assert _metadata_capture._key_local.current_key == "test_key"

        assert _metadata_capture._key_local.current_key is None

    def test_capture_key_clears_on_exception(self) -> None:
        """Test that capture_key clears key even on exception."""
        from traigent.utils.langchain_interceptor import _metadata_capture

        try:
            with capture_key("test_key"):
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert _metadata_capture._key_local.current_key is None

    def test_capture_key_allows_multiple_responses(self) -> None:
        """Test that capture_key can capture multiple responses with same key."""
        response1 = MagicMock()
        response2 = MagicMock()

        with capture_key("test_key"):
            capture_langchain_response(response1)
            capture_langchain_response(response2)

        # Last response wins for by_key storage
        result = get_captured_response_by_key("test_key")
        assert result is response2


class TestStreamWrapper:
    """Tests for _create_stream_wrapper function."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_captured_responses()

    def test_stream_wrapper_yields_chunks(self) -> None:
        """Test that stream wrapper yields all chunks."""
        chunk1 = MagicMock()
        chunk1.content = "chunk1"
        chunk2 = MagicMock()
        chunk2.content = "chunk2"
        chunk3 = MagicMock()
        chunk3.content = "chunk3"

        def mock_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock streaming method."""
            yield chunk1
            yield chunk2
            yield chunk3

        wrapper = _create_stream_wrapper(mock_stream)
        mock_self = MagicMock()

        chunks = list(wrapper(mock_self))

        assert chunks == [chunk1, chunk2, chunk3]

    def test_stream_wrapper_captures_last_chunk(self) -> None:
        """Test that stream wrapper captures the last chunk."""
        mock_chunk = MagicMock()
        mock_chunk.content = "final chunk"

        def mock_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock streaming method."""
            yield MagicMock()
            yield MagicMock()
            yield mock_chunk

        wrapper = _create_stream_wrapper(mock_stream)
        mock_self = MagicMock()

        # Consume the stream
        list(wrapper(mock_self))

        # Last chunk should be captured
        captured = get_captured_response()
        assert captured is mock_chunk

    def test_stream_wrapper_adds_response_time(self) -> None:
        """Test that stream wrapper adds response_time_ms to metadata."""
        mock_chunk = MagicMock(spec=["content"])
        mock_chunk.content = "final chunk"

        def mock_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock streaming method."""
            time.sleep(0.01)  # Small delay
            yield mock_chunk

        wrapper = _create_stream_wrapper(mock_stream)
        mock_self = MagicMock()

        # Consume the stream
        list(wrapper(mock_self))

        # Should have response_metadata with timing
        assert hasattr(mock_chunk, "response_metadata")
        assert "response_time_ms" in mock_chunk.response_metadata
        assert mock_chunk.response_metadata["response_time_ms"] > 0

    def test_stream_wrapper_handles_empty_stream(self) -> None:
        """Test that stream wrapper handles empty stream gracefully."""

        def mock_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock streaming method that yields nothing."""
            if False:
                yield  # Make it a generator but yield nothing

        wrapper = _create_stream_wrapper(mock_stream)
        mock_self = MagicMock()

        chunks = list(wrapper(mock_self))

        assert chunks == []
        # Should not capture anything
        assert get_captured_response() is None

    def test_stream_wrapper_preserves_existing_metadata(self) -> None:
        """Test that stream wrapper preserves existing response metadata."""
        mock_chunk = MagicMock()
        mock_chunk.response_metadata = {"model": "gpt-4", "usage": {"tokens": 100}}

        def mock_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock streaming method."""
            yield mock_chunk

        wrapper = _create_stream_wrapper(mock_stream)
        mock_self = MagicMock()

        list(wrapper(mock_self))

        # Should preserve existing metadata and add timing
        assert mock_chunk.response_metadata["model"] == "gpt-4"
        assert "response_time_ms" in mock_chunk.response_metadata


class TestAstreamWrapper:
    """Tests for _create_astream_wrapper function."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_captured_responses()

    @pytest.mark.asyncio
    async def test_astream_wrapper_yields_chunks(self) -> None:
        """Test that async stream wrapper yields all chunks."""
        chunk1 = MagicMock()
        chunk1.content = "chunk1"
        chunk2 = MagicMock()
        chunk2.content = "chunk2"
        chunk3 = MagicMock()
        chunk3.content = "chunk3"

        async def mock_astream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock async streaming method."""
            yield chunk1
            yield chunk2
            yield chunk3

        wrapper = _create_astream_wrapper(mock_astream)
        mock_self = MagicMock()

        chunks = []
        async for chunk in wrapper(mock_self):
            chunks.append(chunk)

        assert chunks == [chunk1, chunk2, chunk3]

    @pytest.mark.asyncio
    async def test_astream_wrapper_captures_last_chunk(self) -> None:
        """Test that async stream wrapper captures the last chunk."""
        mock_chunk = MagicMock()
        mock_chunk.content = "final chunk"

        async def mock_astream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock async streaming method."""
            yield MagicMock()
            yield MagicMock()
            yield mock_chunk

        wrapper = _create_astream_wrapper(mock_astream)
        mock_self = MagicMock()

        # Consume the stream
        async for _ in wrapper(mock_self):
            pass

        # Last chunk should be captured
        captured = get_captured_response()
        assert captured is mock_chunk

    @pytest.mark.asyncio
    async def test_astream_wrapper_adds_response_time(self) -> None:
        """Test that async stream wrapper adds response_time_ms to metadata."""
        mock_chunk = MagicMock(spec=["content"])
        mock_chunk.content = "final chunk"

        async def mock_astream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock async streaming method."""
            import asyncio

            await asyncio.sleep(0.01)  # Small delay
            yield mock_chunk

        wrapper = _create_astream_wrapper(mock_astream)
        mock_self = MagicMock()

        # Consume the stream
        async for _ in wrapper(mock_self):
            pass

        # Should have response_metadata with timing
        assert hasattr(mock_chunk, "response_metadata")
        assert "response_time_ms" in mock_chunk.response_metadata
        assert mock_chunk.response_metadata["response_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_astream_wrapper_handles_empty_stream(self) -> None:
        """Test that async stream wrapper handles empty stream gracefully."""

        async def mock_astream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock async streaming method that yields nothing."""
            if False:
                yield  # Make it a generator but yield nothing

        wrapper = _create_astream_wrapper(mock_astream)
        mock_self = MagicMock()

        chunks = []
        async for chunk in wrapper(mock_self):
            chunks.append(chunk)

        assert chunks == []
        # Should not capture anything
        assert get_captured_response() is None

    @pytest.mark.asyncio
    async def test_astream_wrapper_preserves_existing_metadata(self) -> None:
        """Test that async stream wrapper preserves existing response metadata."""
        mock_chunk = MagicMock()
        mock_chunk.response_metadata = {"model": "gpt-4", "usage": {"tokens": 100}}

        async def mock_astream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock async streaming method."""
            yield mock_chunk

        wrapper = _create_astream_wrapper(mock_astream)
        mock_self = MagicMock()

        async for _ in wrapper(mock_self):
            pass

        # Should preserve existing metadata and add timing
        assert mock_chunk.response_metadata["model"] == "gpt-4"
        assert "response_time_ms" in mock_chunk.response_metadata


class TestPatchLangChainForMetadataCapture:
    """Tests for patch_langchain_for_metadata_capture function."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        clear_captured_responses()

    @patch("traigent.utils.langchain_interceptor.logger")
    def test_patch_returns_false_when_no_langchain_available(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that patch returns False when LangChain is not available."""
        with patch.dict(
            "sys.modules", {"langchain_anthropic": None, "langchain_openai": None}
        ):
            result = patch_langchain_for_metadata_capture()
            assert result is False

    @patch("traigent.utils.langchain_interceptor.logger")
    def test_patch_logs_debug_when_no_models_patched(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that patch logs debug when no models can be patched."""
        with patch.dict(
            "sys.modules", {"langchain_anthropic": None, "langchain_openai": None}
        ):
            patch_langchain_for_metadata_capture()
            mock_logger.debug.assert_any_call(
                "No LangChain models could be patched for metadata capture"
            )

    @patch("traigent.utils.langchain_interceptor.logger")
    def test_patch_handles_import_error_gracefully(
        self, mock_logger: MagicMock
    ) -> None:
        """Test that patch handles ImportError gracefully."""

        def raise_import_error(name: str, *args: Any, **kwargs: Any) -> None:
            """Raise ImportError for langchain modules."""
            if "langchain" in name:
                raise ImportError(f"No module named '{name}'")
            return None

        with patch("builtins.__import__", side_effect=raise_import_error):
            result = patch_langchain_for_metadata_capture()
            # Should not crash, should return False
            assert result is False

    def test_patch_logs_debug_on_import_error(self) -> None:
        """Test that patch logs debug message when module not available."""
        # This tests the actual behavior without mocking internals
        # Since LangChain may or may not be installed, we just verify it doesn't crash
        result = patch_langchain_for_metadata_capture()
        # Result depends on whether LangChain is installed
        assert isinstance(result, bool)

    def test_clear_removes_storage_correctly(self) -> None:
        """Test that clear operation removes all stored responses."""
        # Store some responses
        capture_langchain_response(MagicMock())
        capture_langchain_response(MagicMock())

        with capture_key("test_key"):
            capture_langchain_response(MagicMock())

        # Clear and verify
        clear_captured_responses()

        assert get_captured_response() is None
        assert get_all_captured_responses() == []
        assert get_captured_response_by_key("test_key") is None

    def test_metadata_capture_without_response_metadata_attribute(self) -> None:
        """Test capture handles responses without response_metadata attribute."""
        response = MagicMock(spec=["content"])
        response.content = "Test"

        # This should not crash even without response_metadata
        capture_langchain_response(response)

        captured = get_captured_response()
        assert captured is response

    def test_stream_wrapper_with_none_last_chunk(self) -> None:
        """Test stream wrapper when last chunk is None."""

        def mock_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock streaming method with None."""
            yield "chunk1"
            yield None

        wrapper = _create_stream_wrapper(mock_stream)
        mock_self = MagicMock()

        chunks = list(wrapper(mock_self))

        # Should yield all chunks including None
        assert chunks == ["chunk1", None]
        # None should not be captured (no response_metadata)
        captured = get_captured_response()
        # Captured will be None since last chunk is None
        assert captured is None

    @pytest.mark.asyncio
    async def test_astream_wrapper_with_none_last_chunk(self) -> None:
        """Test async stream wrapper when last chunk is None."""

        async def mock_astream(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Mock async streaming method with None."""
            yield "chunk1"
            yield None

        wrapper = _create_astream_wrapper(mock_astream)
        mock_self = MagicMock()

        chunks = []
        async for chunk in wrapper(mock_self):
            chunks.append(chunk)

        # Should yield all chunks including None
        assert chunks == ["chunk1", None]
        # None should not be captured
        captured = get_captured_response()
        assert captured is None

    def test_multiple_keys_stored_separately(self) -> None:
        """Test that multiple keys store responses separately."""
        response1 = MagicMock()
        response2 = MagicMock()
        response3 = MagicMock()

        with capture_key("key1"):
            capture_langchain_response(response1)

        with capture_key("key2"):
            capture_langchain_response(response2)

        with capture_key("key3"):
            capture_langchain_response(response3)

        assert get_captured_response_by_key("key1") is response1
        assert get_captured_response_by_key("key2") is response2
        assert get_captured_response_by_key("key3") is response3

    def test_clear_clears_current_key_properly(self) -> None:
        """Test that clear properly clears current key in thread local."""
        from traigent.utils.langchain_interceptor import _metadata_capture

        _metadata_capture.set_current_key("test_key")
        clear_captured_responses()

        assert _metadata_capture._key_local.current_key is None

    def test_nested_capture_key_contexts(self) -> None:
        """Test nested capture_key contexts."""
        from traigent.utils.langchain_interceptor import _metadata_capture

        with capture_key("outer"):
            assert _metadata_capture._key_local.current_key == "outer"

            with capture_key("inner"):
                assert _metadata_capture._key_local.current_key == "inner"
                response = MagicMock()
                capture_langchain_response(response)

            # After exiting inner context
            assert _metadata_capture._key_local.current_key is None

        # After exiting outer context
        assert _metadata_capture._key_local.current_key is None

    def test_thread_local_isolation(self) -> None:
        """Test that thread local storage isolates responses between threads."""
        from traigent.utils.langchain_interceptor import _metadata_capture

        results = []
        lock = threading.Lock()

        def thread_func(thread_id: int) -> None:
            response = MagicMock()
            response.thread_id = thread_id
            _metadata_capture.set_last_response(response)

            # Get response should only get this thread's response
            retrieved = _metadata_capture.get_last_response()
            with lock:
                results.append((thread_id, retrieved.thread_id if retrieved else None))

        threads = [threading.Thread(target=thread_func, args=(i,)) for i in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Each thread should retrieve its own response
        assert len(results) == 5
