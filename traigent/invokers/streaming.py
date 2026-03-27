"""Streaming function invocation strategy for LLM responses."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-INVOKERS REQ-INV-006 REQ-INJ-002 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from traigent.invokers.base import InvocationResult, StreamingChunk
from traigent.invokers.local import LocalInvoker
from traigent.utils.exceptions import InvocationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class StreamingInvoker(LocalInvoker):
    """Streaming function invocation strategy.

    Extends LocalInvoker to support streaming responses from LLM functions.
    Functions must return an async iterator or generator to use streaming.

    Example:
        >>> invoker = StreamingInvoker(timeout=60.0)  # doctest: +SKIP
        >>> async for chunk in invoker.invoke_streaming(llm_func, config, input_data):  # doctest: +SKIP
        ...     print(chunk.content, end="")  # doctest: +SKIP

        >>> # Or collect all chunks into a result
        >>> result = await invoker.invoke_streaming_to_result(llm_func, config, input_data)  # doctest: +SKIP
        >>> print(result.accumulated_content)  # doctest: +SKIP
    """

    def __init__(
        self,
        timeout: float = 120.0,
        max_retries: int = 0,
        chunk_timeout: float = 30.0,
        buffer_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize streaming invoker.

        Args:
            timeout: Timeout for entire streaming operation (seconds)
            max_retries: Maximum number of retries for failed invocations
            chunk_timeout: Timeout for receiving individual chunks (seconds)
            buffer_size: Maximum number of chunks to buffer
            **kwargs: Additional configuration passed to LocalInvoker
        """
        super().__init__(timeout, max_retries, **kwargs)
        self.chunk_timeout = self._validate_chunk_timeout(chunk_timeout)
        self.buffer_size = self._validate_buffer_size(buffer_size)

        logger.debug(
            f"StreamingInvoker configured: chunk_timeout={chunk_timeout}s, "
            f"buffer_size={buffer_size}"
        )

    def supports_streaming(self) -> bool:
        """Streaming invoker supports streaming responses."""
        return True

    def supports_batch(self) -> bool:
        """Streaming invoker supports batch processing via parent class."""
        return True

    def invoke_streaming(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> AsyncIterator[StreamingChunk]:
        """Invoke function and stream response chunks.

        Args:
            func: Function to invoke (should return an async iterator)
            config: Configuration parameters
            input_data: Input data for the function

        Returns:
            AsyncIterator yielding StreamingChunk objects

        Raises:
            InvocationError: If function cannot be streamed
        """
        self.validate_function(func)
        self.validate_config(config)
        self.validate_input(input_data)

        return self._stream_response(func, config, input_data)

    async def _stream_response(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> AsyncIterator[StreamingChunk]:
        """Internal streaming implementation.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_data: Input data

        Yields:
            StreamingChunk objects
        """
        start_time = time.time()
        chunk_index = 0

        try:
            # Inject configuration into function
            configured_func = self._provider.inject_config(
                func, config, self.config_param
            )

            # Get the response iterator
            response = configured_func(**input_data)

            # Await plain awaitables, but do not await generators/iterators.
            if inspect.isawaitable(response) and not inspect.isgenerator(response):
                response = await response

            # Stream chunks from the response
            async for chunk_content in self._iterate_response(response):
                is_final = False  # Will be set after loop completes

                chunk = StreamingChunk(
                    content=chunk_content,
                    index=chunk_index,
                    is_final=is_final,
                    metadata={
                        "elapsed_time": time.time() - start_time,
                    },
                )
                yield chunk
                chunk_index += 1

                # Check buffer size limit
                if chunk_index >= self.buffer_size:
                    logger.warning(
                        f"Buffer size limit ({self.buffer_size}) reached, "
                        "some chunks may be dropped"
                    )

            # Yield final chunk marker if we got any content
            if chunk_index > 0:
                yield StreamingChunk(
                    content="",
                    index=chunk_index,
                    is_final=True,
                    metadata={
                        "total_chunks": chunk_index,
                        "total_time": time.time() - start_time,
                    },
                )

        except TimeoutError:
            error_msg = f"Streaming timed out after {self.timeout}s"
            logger.warning(error_msg)
            yield StreamingChunk(
                content="",
                index=chunk_index,
                is_final=True,
                metadata={"error": error_msg, "timeout": True},
            )

        except Exception as e:
            error_msg = f"Streaming failed: {e}"
            logger.error(error_msg)
            yield StreamingChunk(
                content="",
                index=chunk_index,
                is_final=True,
                metadata={"error": error_msg, "exception_type": type(e).__name__},
            )

    async def _iterate_response(self, response: Any) -> AsyncIterator[Any]:
        """Iterate over response, handling both sync and async iterators.

        Args:
            response: The response to iterate over

        Yields:
            Individual chunks from the response
        """
        # Handle async iterators
        if hasattr(response, "__aiter__"):
            async_iter = response.__aiter__()
            while True:
                try:
                    if self.chunk_timeout:
                        chunk = await asyncio.wait_for(
                            async_iter.__anext__(), timeout=self.chunk_timeout
                        )
                    else:
                        chunk = await async_iter.__anext__()
                except StopAsyncIteration:
                    break
                except TimeoutError as e:
                    raise InvocationError(
                        f"Chunk timeout after {self.chunk_timeout}s"
                    ) from e
                yield chunk

        # Handle sync iterators (wrap in async)
        elif hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
            for chunk in response:
                yield chunk
                # Allow other tasks to run
                await asyncio.sleep(0)

        # Handle generators
        elif inspect.isgenerator(response):
            for chunk in response:
                yield chunk
                await asyncio.sleep(0)

        # Handle async generators
        elif inspect.isasyncgen(response):
            async for chunk in response:
                yield chunk

        # Single value - yield as one chunk
        else:
            yield response

    async def invoke(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> InvocationResult:
        """Invoke function and collect streaming response into a single result.

        For streaming functions, this collects all chunks and returns the
        accumulated content. For non-streaming functions, delegates to parent.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_data: Input data for the function

        Returns:
            InvocationResult with accumulated content
        """
        # Check if the function appears to be a streaming function
        if self._is_streaming_function(func):
            result = await self.invoke_streaming_to_result(func, config, input_data)
            return result.to_invocation_result()

        # Fall back to parent implementation for non-streaming functions
        return await super().invoke(func, config, input_data)

    def _is_streaming_function(self, func: Callable[..., Any]) -> bool:
        """Check if function appears to return a streaming response.

        Args:
            func: Function to check

        Returns:
            True if function appears to be streaming
        """
        # Check function annotations
        if hasattr(func, "__annotations__"):
            return_type = func.__annotations__.get("return", None)
            if return_type:
                type_str = str(return_type)
                streaming_indicators = [
                    "AsyncIterator",
                    "AsyncGenerator",
                    "Iterator",
                    "Generator",
                    "Stream",
                ]
                return any(ind in type_str for ind in streaming_indicators)

        # Check function name patterns
        func_name = getattr(func, "__name__", "")
        streaming_patterns = ["stream", "generate", "iter"]
        return any(pattern in func_name.lower() for pattern in streaming_patterns)

    def _validate_chunk_timeout(self, timeout: float | None) -> float | None:
        """Validate chunk timeout setting."""
        if timeout is None:
            return None
        if not isinstance(timeout, (int, float)):
            raise InvocationError("chunk_timeout must be numeric or None")
        timeout_value = float(timeout)
        if timeout_value <= 0:
            raise InvocationError("chunk_timeout must be greater than zero seconds")
        if timeout_value > self.MAX_TIMEOUT_SECONDS:
            raise InvocationError(
                f"chunk_timeout {timeout_value}s exceeds maximum allowed "
                f"{self.MAX_TIMEOUT_SECONDS}s"
            )
        return timeout_value

    def _validate_buffer_size(self, buffer_size: int) -> int:
        """Validate buffer size setting."""
        if not isinstance(buffer_size, int):
            raise InvocationError("buffer_size must be an integer")
        if buffer_size < 1:
            raise InvocationError("buffer_size must be at least 1")
        if buffer_size > 10000:
            raise InvocationError(
                f"buffer_size {buffer_size} exceeds maximum allowed 10000"
            )
        return buffer_size
