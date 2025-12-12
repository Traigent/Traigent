"""Comprehensive tests for streaming invoker and streaming types.

Tests cover:
- StreamingChunk dataclass
- StreamingInvocationResult dataclass
- StreamingInvoker class
- Streaming response handling
- Error handling and edge cases
"""

from __future__ import annotations

import asyncio

import pytest

from traigent.invokers.base import (
    BaseInvoker,
    InvocationResult,
    StreamingChunk,
    StreamingInvocationResult,
)
from traigent.invokers.streaming import StreamingInvoker
from traigent.utils.exceptions import InvocationError


class TestStreamingChunk:
    """Test StreamingChunk dataclass."""

    def test_streaming_chunk_creation_basic(self):
        """Test creating basic StreamingChunk."""
        chunk = StreamingChunk(content="Hello")

        assert chunk.content == "Hello"
        assert chunk.index == 0
        assert chunk.is_final is False
        assert chunk.metadata == {}

    def test_streaming_chunk_creation_with_index(self):
        """Test creating StreamingChunk with index."""
        chunk = StreamingChunk(content="World", index=5)

        assert chunk.content == "World"
        assert chunk.index == 5
        assert chunk.is_final is False

    def test_streaming_chunk_creation_final(self):
        """Test creating final StreamingChunk."""
        chunk = StreamingChunk(content="", index=10, is_final=True)

        assert chunk.content == ""
        assert chunk.index == 10
        assert chunk.is_final is True

    def test_streaming_chunk_with_metadata(self):
        """Test StreamingChunk with metadata."""
        metadata = {"elapsed_time": 0.5, "token_count": 15}
        chunk = StreamingChunk(content="Test", index=1, metadata=metadata)

        assert chunk.metadata["elapsed_time"] == 0.5
        assert chunk.metadata["token_count"] == 15

    def test_streaming_chunk_content_types(self):
        """Test StreamingChunk with various content types."""
        # String content
        string_chunk = StreamingChunk(content="text")
        assert string_chunk.content == "text"

        # Dict content
        dict_chunk = StreamingChunk(content={"key": "value"})
        assert dict_chunk.content == {"key": "value"}

        # List content
        list_chunk = StreamingChunk(content=[1, 2, 3])
        assert list_chunk.content == [1, 2, 3]

        # None content
        none_chunk = StreamingChunk(content=None)
        assert none_chunk.content is None


class TestStreamingInvocationResult:
    """Test StreamingInvocationResult dataclass."""

    def test_streaming_result_creation_empty(self):
        """Test creating empty StreamingInvocationResult."""
        result = StreamingInvocationResult()

        assert result.chunks == []
        assert result.execution_time == 0.0
        assert result.metadata == {}
        assert result.error is None
        assert result.is_successful is True
        assert result.is_complete is False

    def test_streaming_result_with_chunks(self):
        """Test StreamingInvocationResult with chunks."""
        chunks = [
            StreamingChunk(content="Hello", index=0),
            StreamingChunk(content=" ", index=1),
            StreamingChunk(content="World", index=2),
        ]
        result = StreamingInvocationResult(chunks=chunks)

        assert len(result.chunks) == 3
        assert result.chunk_count == 3

    def test_streaming_result_accumulated_content(self):
        """Test accumulated_content property."""
        result = StreamingInvocationResult()
        result.add_chunk(StreamingChunk(content="Hello", index=0))
        result.add_chunk(StreamingChunk(content=" ", index=1))
        result.add_chunk(StreamingChunk(content="World", index=2))

        assert result.accumulated_content == "Hello World"

    def test_streaming_result_add_chunk(self):
        """Test add_chunk method."""
        result = StreamingInvocationResult()

        result.add_chunk(StreamingChunk(content="First", index=0))
        assert result.chunk_count == 1
        assert result.is_complete is False

        result.add_chunk(StreamingChunk(content="Last", index=1, is_final=True))
        assert result.chunk_count == 2
        assert result.is_complete is True

    def test_streaming_result_with_error(self):
        """Test StreamingInvocationResult with error."""
        result = StreamingInvocationResult(error="Connection failed")

        assert result.error == "Connection failed"
        assert result.is_successful is False

    def test_streaming_result_to_invocation_result(self):
        """Test conversion to standard InvocationResult."""
        result = StreamingInvocationResult(execution_time=1.5)
        result.add_chunk(StreamingChunk(content="Test", index=0))
        result.add_chunk(StreamingChunk(content=" Content", index=1, is_final=True))

        invocation_result = result.to_invocation_result()

        assert isinstance(invocation_result, InvocationResult)
        assert invocation_result.result == "Test Content"
        assert invocation_result.execution_time == 1.5
        assert invocation_result.is_successful is True
        assert invocation_result.metadata["streaming"] is True
        assert invocation_result.metadata["chunk_count"] == 2

    def test_streaming_result_to_invocation_result_with_error(self):
        """Test conversion with error."""
        result = StreamingInvocationResult(error="Stream failed")

        invocation_result = result.to_invocation_result()

        assert invocation_result.is_successful is False
        assert invocation_result.error == "Stream failed"


class TestStreamingInvokerCreation:
    """Test StreamingInvoker creation and configuration."""

    def test_streaming_invoker_default_creation(self):
        """Test creating StreamingInvoker with defaults."""
        invoker = StreamingInvoker()

        assert invoker.timeout == 120.0
        assert invoker.max_retries == 0
        assert invoker.chunk_timeout == 30.0
        assert invoker.buffer_size == 100

    def test_streaming_invoker_custom_timeout(self):
        """Test StreamingInvoker with custom timeout."""
        invoker = StreamingInvoker(timeout=60.0)

        assert invoker.timeout == 60.0

    def test_streaming_invoker_custom_chunk_timeout(self):
        """Test StreamingInvoker with custom chunk timeout."""
        invoker = StreamingInvoker(chunk_timeout=15.0)

        assert invoker.chunk_timeout == 15.0

    def test_streaming_invoker_custom_buffer_size(self):
        """Test StreamingInvoker with custom buffer size."""
        invoker = StreamingInvoker(buffer_size=50)

        assert invoker.buffer_size == 50

    def test_streaming_invoker_supports_streaming(self):
        """Test that StreamingInvoker supports streaming."""
        invoker = StreamingInvoker()

        assert invoker.supports_streaming() is True

    def test_streaming_invoker_supports_batch(self):
        """Test that StreamingInvoker supports batch processing."""
        invoker = StreamingInvoker()

        assert invoker.supports_batch() is True


class TestStreamingInvokerValidation:
    """Test StreamingInvoker validation methods."""

    def test_invalid_chunk_timeout_type(self):
        """Test that invalid chunk_timeout type raises error."""
        with pytest.raises(InvocationError, match="chunk_timeout must be numeric"):
            StreamingInvoker(chunk_timeout="invalid")

    def test_invalid_chunk_timeout_zero(self):
        """Test that zero chunk_timeout raises error."""
        with pytest.raises(
            InvocationError, match="chunk_timeout must be greater than zero"
        ):
            StreamingInvoker(chunk_timeout=0)

    def test_invalid_chunk_timeout_negative(self):
        """Test that negative chunk_timeout raises error."""
        with pytest.raises(
            InvocationError, match="chunk_timeout must be greater than zero"
        ):
            StreamingInvoker(chunk_timeout=-5)

    def test_invalid_chunk_timeout_too_large(self):
        """Test that chunk_timeout exceeding maximum raises error."""
        with pytest.raises(InvocationError, match="exceeds maximum allowed"):
            StreamingInvoker(chunk_timeout=4000)  # > MAX_TIMEOUT_SECONDS

    def test_invalid_buffer_size_type(self):
        """Test that invalid buffer_size type raises error."""
        with pytest.raises(InvocationError, match="buffer_size must be an integer"):
            StreamingInvoker(buffer_size=5.5)

    def test_invalid_buffer_size_zero(self):
        """Test that zero buffer_size raises error."""
        with pytest.raises(InvocationError, match="buffer_size must be at least 1"):
            StreamingInvoker(buffer_size=0)

    def test_invalid_buffer_size_negative(self):
        """Test that negative buffer_size raises error."""
        with pytest.raises(InvocationError, match="buffer_size must be at least 1"):
            StreamingInvoker(buffer_size=-10)

    def test_invalid_buffer_size_too_large(self):
        """Test that buffer_size exceeding maximum raises error."""
        with pytest.raises(InvocationError, match="exceeds maximum allowed"):
            StreamingInvoker(buffer_size=50000)


class TestStreamingInvokerFunctionDetection:
    """Test streaming function detection."""

    def test_is_streaming_function_by_name_stream(self):
        """Test detection by function name containing 'stream'."""
        invoker = StreamingInvoker()

        async def stream_response():
            yield "chunk"

        assert invoker._is_streaming_function(stream_response) is True

    def test_is_streaming_function_by_name_generate(self):
        """Test detection by function name containing 'generate'."""
        invoker = StreamingInvoker()

        async def generate_tokens():
            yield "token"

        assert invoker._is_streaming_function(generate_tokens) is True

    def test_is_streaming_function_by_name_iter(self):
        """Test detection by function name containing 'iter'."""
        invoker = StreamingInvoker()

        async def iterate_results():
            yield "result"

        assert invoker._is_streaming_function(iterate_results) is True

    def test_is_not_streaming_function_regular_name(self):
        """Test that regular functions are not detected as streaming."""
        invoker = StreamingInvoker()

        def regular_function():
            return "result"

        assert invoker._is_streaming_function(regular_function) is False

    def test_is_streaming_function_by_annotation(self):
        """Test detection by return type annotation."""
        from collections.abc import AsyncIterator, Iterator

        invoker = StreamingInvoker()

        async def async_iterator_func() -> AsyncIterator[str]:
            yield "chunk"

        def sync_iterator_func() -> Iterator[str]:
            yield "chunk"

        assert invoker._is_streaming_function(async_iterator_func) is True
        assert invoker._is_streaming_function(sync_iterator_func) is True


class TestStreamingInvokerExecution:
    """Test StreamingInvoker execution methods."""

    @pytest.mark.asyncio
    async def test_invoke_streaming_async_generator(self):
        """Test streaming with async generator function."""
        invoker = StreamingInvoker()

        async def async_generator(**kwargs):
            for i in range(3):
                yield f"chunk{i}"

        chunks = []
        async for chunk in invoker._stream_response(async_generator, {}, {}):
            chunks.append(chunk)

        # 3 content chunks + 1 final marker
        assert len(chunks) == 4
        assert chunks[0].content == "chunk0"
        assert chunks[1].content == "chunk1"
        assert chunks[2].content == "chunk2"
        assert chunks[3].is_final is True

    @pytest.mark.asyncio
    async def test_invoke_streaming_sync_generator(self):
        """Test streaming with sync generator function."""
        invoker = StreamingInvoker()

        def sync_generator(**kwargs):
            for i in range(3):
                yield f"item{i}"

        chunks = []
        async for chunk in invoker._stream_response(sync_generator, {}, {}):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert chunks[0].content == "item0"
        assert chunks[3].is_final is True

    @pytest.mark.asyncio
    async def test_invoke_streaming_single_value(self):
        """Test streaming with function returning single value."""
        invoker = StreamingInvoker()

        def single_value_func(**kwargs):
            return "single_result"

        chunks = []
        async for chunk in invoker._stream_response(single_value_func, {}, {}):
            chunks.append(chunk)

        # 1 content chunk + 1 final marker
        assert len(chunks) == 2
        assert chunks[0].content == "single_result"
        assert chunks[1].is_final is True

    @pytest.mark.asyncio
    async def test_invoke_streaming_to_result(self):
        """Test invoke_streaming_to_result collects all chunks."""
        invoker = StreamingInvoker()

        async def streaming_func(**kwargs):
            yield "Hello"
            yield " "
            yield "World"

        result = await invoker.invoke_streaming_to_result(streaming_func, {}, {})

        assert isinstance(result, StreamingInvocationResult)
        assert result.accumulated_content == "Hello World"
        assert result.is_complete is True
        assert result.chunk_count >= 3

    @pytest.mark.asyncio
    async def test_invoke_streaming_metadata(self):
        """Test that streaming chunks contain metadata."""
        invoker = StreamingInvoker()

        async def generator(**kwargs):
            yield "test"

        chunks = []
        async for chunk in invoker._stream_response(generator, {}, {}):
            chunks.append(chunk)

        # Check first chunk has elapsed_time metadata
        assert "elapsed_time" in chunks[0].metadata

        # Check final chunk has total_time and total_chunks
        final_chunk = chunks[-1]
        assert final_chunk.is_final is True
        assert "total_time" in final_chunk.metadata
        assert "total_chunks" in final_chunk.metadata


class TestStreamingInvokerErrorHandling:
    """Test error handling in StreamingInvoker."""

    @pytest.mark.asyncio
    async def test_streaming_exception_handling(self):
        """Test that exceptions during streaming are handled."""
        invoker = StreamingInvoker()

        async def failing_generator(**kwargs):
            yield "first"
            raise ValueError("Stream error")

        chunks = []
        async for chunk in invoker._stream_response(failing_generator, {}, {}):
            chunks.append(chunk)

        # Should have first chunk + error chunk
        assert len(chunks) >= 2
        final_chunk = chunks[-1]
        assert final_chunk.is_final is True
        assert "error" in final_chunk.metadata
        assert "Streaming failed" in final_chunk.metadata["error"]

    @pytest.mark.asyncio
    async def test_invoke_streaming_validation_function(self):
        """Test that invoke_streaming validates function."""
        invoker = StreamingInvoker()

        with pytest.raises(InvocationError, match="must be callable"):
            # Need to consume the iterator to trigger validation
            async for _ in invoker.invoke_streaming("not_a_function", {}, {}):
                pass

    @pytest.mark.asyncio
    async def test_invoke_streaming_validation_config(self):
        """Test that invoke_streaming validates config."""
        invoker = StreamingInvoker()

        def func():
            yield "test"

        with pytest.raises(InvocationError, match="must be a dictionary"):
            async for _ in invoker.invoke_streaming(func, "invalid_config", {}):
                pass

    @pytest.mark.asyncio
    async def test_invoke_streaming_validation_input(self):
        """Test that invoke_streaming validates input."""
        invoker = StreamingInvoker()

        def func():
            yield "test"

        with pytest.raises(InvocationError, match="must be a dictionary"):
            async for _ in invoker.invoke_streaming(func, {}, "invalid_input"):
                pass


class TestStreamingInvokerEdgeCases:
    """Test edge cases for StreamingInvoker."""

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test handling of empty stream."""
        invoker = StreamingInvoker()

        async def empty_generator(**kwargs):
            return
            yield  # Makes it a generator

        chunks = []
        async for chunk in invoker._stream_response(empty_generator, {}, {}):
            chunks.append(chunk)

        # Should have no content chunks but might have final marker logic
        # Empty generator returns no chunks
        assert len(chunks) == 0 or all(c.is_final for c in chunks)

    @pytest.mark.asyncio
    async def test_large_stream(self):
        """Test handling of large number of chunks."""
        invoker = StreamingInvoker(buffer_size=50)

        async def large_generator(**kwargs):
            for i in range(100):
                yield f"chunk{i}"

        chunks = []
        async for chunk in invoker._stream_response(large_generator, {}, {}):
            chunks.append(chunk)

        # Should process all 100 chunks + final marker
        assert len(chunks) == 101
        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_streaming_with_delays(self):
        """Test streaming with delays between chunks."""
        invoker = StreamingInvoker()

        async def delayed_generator(**kwargs):
            yield "first"
            await asyncio.sleep(0.01)
            yield "second"
            await asyncio.sleep(0.01)
            yield "third"

        chunks = []
        async for chunk in invoker._stream_response(delayed_generator, {}, {}):
            chunks.append(chunk)

        assert len(chunks) == 4
        # Check elapsed time increases
        assert chunks[2].metadata.get("elapsed_time", 0) > 0


class TestBaseInvokerStreamingInterface:
    """Test streaming interface in BaseInvoker."""

    def test_base_invoker_invoke_streaming_not_supported(self):
        """Test that BaseInvoker without streaming support raises error."""

        class NonStreamingInvoker(BaseInvoker):
            async def invoke(self, func, config, input_data):
                return InvocationResult(result="test")

            async def invoke_batch(self, func, config, input_batch):
                return []

            def supports_streaming(self):
                return False

            def supports_batch(self):
                return True

        invoker = NonStreamingInvoker()

        with pytest.raises(InvocationError, match="does not support streaming"):
            invoker.invoke_streaming(lambda: None, {}, {})

    def test_streaming_invoker_has_streaming_interface(self):
        """Test that StreamingInvoker has proper streaming interface."""
        invoker = StreamingInvoker()

        # Should have invoke_streaming method
        assert hasattr(invoker, "invoke_streaming")
        assert callable(invoker.invoke_streaming)

        # Should have invoke_streaming_to_result method
        assert hasattr(invoker, "invoke_streaming_to_result")
        assert callable(invoker.invoke_streaming_to_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
