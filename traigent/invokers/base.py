"""Base classes for function invocation strategies."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-INVOKERS REQ-INV-006 REQ-INJ-002 SYNC-OptimizationFlow

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from traigent.utils.exceptions import InvocationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InvocationResult:
    """Result from a single function invocation.

    This class contains the output and metadata from invoking a function
    with a specific configuration and input.

    Attributes:
        result: The result returned by the function
        execution_time: Time taken to execute the function (seconds)
        metadata: Additional metadata about the invocation
        error: Error message if invocation failed
        is_successful: Whether the invocation was successful
    """

    result: Any | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    is_successful: bool = True

    def __post_init__(self) -> None:
        """Set success based on error status."""
        if self.error is not None:
            self.is_successful = False

    @property
    def output(self) -> Any:
        """Backward compatibility: alias for result."""
        return self.result

    @property
    def success(self) -> bool:
        """Backward compatibility: alias for is_successful."""
        return self.is_successful

    @property
    def failed(self) -> bool:
        """Check if invocation failed."""
        return not self.is_successful


@dataclass
class StreamingChunk:
    """A single chunk from a streaming response.

    Attributes:
        content: The content of this chunk
        index: Position of this chunk in the stream
        is_final: Whether this is the last chunk
        metadata: Additional chunk-specific metadata
    """

    content: Any
    index: int = 0
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingInvocationResult:
    """Result from a streaming function invocation.

    This class wraps streaming responses and provides accumulated results
    after the stream completes.

    Attributes:
        chunks: List of received chunks
        execution_time: Total time taken for streaming (seconds)
        metadata: Additional metadata about the invocation
        error: Error message if invocation failed
        is_successful: Whether the invocation was successful
        is_complete: Whether streaming has completed
    """

    chunks: list[StreamingChunk] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    is_successful: bool = True
    is_complete: bool = False

    def __post_init__(self) -> None:
        """Set success based on error status."""
        if self.error is not None:
            self.is_successful = False

    @property
    def accumulated_content(self) -> str:
        """Get all chunk contents concatenated as a string."""
        return "".join(str(chunk.content) for chunk in self.chunks)

    @property
    def chunk_count(self) -> int:
        """Get the number of chunks received."""
        return len(self.chunks)

    def add_chunk(self, chunk: StreamingChunk) -> None:
        """Add a chunk to the result.

        Args:
            chunk: The chunk to add
        """
        self.chunks.append(chunk)
        if chunk.is_final:
            self.is_complete = True

    def to_invocation_result(self) -> InvocationResult:
        """Convert streaming result to standard InvocationResult.

        Returns:
            InvocationResult with accumulated content as result
        """
        return InvocationResult(
            result=self.accumulated_content,
            execution_time=self.execution_time,
            metadata={
                **self.metadata,
                "streaming": True,
                "chunk_count": self.chunk_count,
            },
            error=self.error,
            is_successful=self.is_successful,
        )


class BaseInvoker(ABC):
    """Abstract base class for function invocation strategies.

    This class defines the interface for invoking functions with different
    configurations and managing the execution environment.
    """

    MAX_TIMEOUT_SECONDS = 60 * 60  # 1 hour upper bound for a single invocation
    MAX_RETRIES = 10

    def __init__(
        self, timeout: float | None = None, max_retries: int = 0, **kwargs: Any
    ) -> None:
        """Initialize invoker.

        Args:
            timeout: Timeout for individual invocations (seconds)
            max_retries: Maximum number of retries for failed invocations
            **kwargs: Additional configuration
        """
        self.timeout = self._validate_timeout(timeout)
        self.max_retries = self._validate_retries(max_retries)
        self.config = kwargs

        logger.debug(f"Initialized {self.__class__.__name__} with timeout={timeout}")

    @abstractmethod
    async def invoke(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> InvocationResult:
        """Invoke function with configuration and input data.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_data: Input data for the function

        Returns:
            InvocationResult with output and metadata

        Raises:
            InvocationError: If invocation fails after retries
        """
        pass

    @abstractmethod
    async def invoke_batch(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_batch: list[dict[str, Any]],
    ) -> list[InvocationResult]:
        """Invoke function on multiple inputs with same configuration.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_batch: List of input data dictionaries

        Returns:
            List of InvocationResult objects

        Raises:
            InvocationError: If batch invocation fails
        """
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if invoker supports streaming responses.

        Returns:
            True if streaming is supported
        """
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        """Check if invoker supports batch processing.

        Returns:
            True if batch processing is supported
        """
        pass

    def invoke_streaming(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> AsyncIterator[StreamingChunk]:
        """Invoke function with streaming response.

        This method returns an async iterator that yields chunks as they
        become available. Override in subclasses that support streaming.

        Args:
            func: Function to invoke (must return an async iterator or generator)
            config: Configuration parameters
            input_data: Input data for the function

        Returns:
            AsyncIterator yielding StreamingChunk objects

        Raises:
            InvocationError: If invoker does not support streaming
        """
        if not self.supports_streaming():
            raise InvocationError(
                f"{self.__class__.__name__} does not support streaming. "
                "Use invoke() instead or switch to a streaming-capable invoker."
            )
        # Default implementation - subclasses should override
        raise NotImplementedError(
            f"{self.__class__.__name__} has supports_streaming=True but "
            "invoke_streaming is not implemented"
        )

    async def invoke_streaming_to_result(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> StreamingInvocationResult:
        """Invoke function with streaming and collect all chunks into a result.

        Convenience method that consumes the streaming iterator and returns
        a complete StreamingInvocationResult.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_data: Input data for the function

        Returns:
            StreamingInvocationResult with all collected chunks
        """
        import time

        start_time = time.time()
        result = StreamingInvocationResult(
            metadata=self.create_metadata(start_time, start_time)
        )

        try:
            chunk_index = 0
            async for chunk in self.invoke_streaming(func, config, input_data):
                chunk.index = chunk_index
                result.add_chunk(chunk)
                chunk_index += 1

            end_time = time.time()
            result.execution_time = end_time - start_time
            result.metadata = self.create_metadata(
                start_time,
                end_time,
                streaming=True,
                chunk_count=result.chunk_count,
            )
            result.is_complete = True

        except Exception as e:
            end_time = time.time()
            result.execution_time = end_time - start_time
            result.error = str(e)
            result.is_successful = False
            result.metadata = self.create_metadata(
                start_time,
                end_time,
                streaming=True,
                error_type=type(e).__name__,
            )

        return result

    def validate_function(self, func: Callable[..., Any]) -> None:
        """Validate that function can be invoked by this invoker.

        Args:
            func: Function to validate

        Raises:
            InvocationError: If function is invalid
        """
        if not callable(func):
            raise InvocationError("Function must be callable")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration parameters.

        Args:
            config: Configuration to validate

        Raises:
            InvocationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise InvocationError("Configuration must be a dictionary")

    def validate_input(self, input_data: dict[str, Any]) -> None:
        """Validate input data.

        Args:
            input_data: Input data to validate

        Raises:
            InvocationError: If input data is invalid
        """
        if not isinstance(input_data, dict):
            raise InvocationError("Input data must be a dictionary")

    @classmethod
    def _validate_timeout(cls, timeout: float | None) -> float | None:
        """Validate timeout parameter."""
        if timeout is None:
            return None

        if not isinstance(timeout, (int, float)):
            raise InvocationError("Timeout must be a numeric value in seconds")

        timeout_value = float(timeout)
        if timeout_value <= 0:
            raise InvocationError("Timeout must be greater than zero seconds")

        if timeout_value > cls.MAX_TIMEOUT_SECONDS:
            raise InvocationError(
                f"Timeout {timeout_value}s exceeds maximum allowed "
                f"{cls.MAX_TIMEOUT_SECONDS}s"
            )

        return timeout_value

    @classmethod
    def _validate_retries(cls, max_retries: int) -> int:
        """Validate retry count."""
        if not isinstance(max_retries, int):
            raise InvocationError("max_retries must be an integer")

        if max_retries < 0:
            raise InvocationError("max_retries cannot be negative")

        if max_retries > cls.MAX_RETRIES:
            raise InvocationError(
                f"max_retries {max_retries} exceeds maximum allowed {cls.MAX_RETRIES}"
            )

        return max_retries

    async def invoke_with_retry(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> InvocationResult:
        """Invoke function with retry logic.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_data: Input data for the function

        Returns:
            InvocationResult from successful invocation

        Raises:
            InvocationError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self.invoke(func, config, input_data)
                if result.is_successful:
                    if attempt > 0:
                        logger.info(f"Invocation succeeded on attempt {attempt + 1}")
                    return result
                else:
                    last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Invocation attempt {attempt + 1} failed: {e}")

            if attempt < self.max_retries:
                logger.debug(f"Retrying invocation (attempt {attempt + 2})")

        # All attempts failed
        error_msg = (
            f"Invocation failed after {self.max_retries + 1} attempts: {last_error}"
        )
        logger.error(error_msg)
        return InvocationResult(
            error=error_msg, is_successful=False, execution_time=0.0
        )

    def create_metadata(
        self, start_time: float, end_time: float, **additional: Any
    ) -> dict[str, Any]:
        """Create metadata dictionary for invocation result.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            **additional: Additional metadata fields

        Returns:
            Metadata dictionary
        """
        metadata = {
            "invoker": self.__class__.__name__,
            "start_time": start_time,
            "end_time": end_time,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        metadata.update(additional)
        return metadata
