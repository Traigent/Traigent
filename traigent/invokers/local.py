"""Local function invocation strategy."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Performance FUNC-INVOKERS REQ-INV-006 REQ-INJ-002 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any

from traigent.config.providers import get_provider
from traigent.invokers.base import BaseInvoker, InvocationResult
from traigent.utils.error_handler import APIKeyError
from traigent.utils.exceptions import InvocationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class LocalInvoker(BaseInvoker):
    """Local function invocation strategy.

    Invokes functions locally in the current process with configuration injection.
    Supports both synchronous and asynchronous functions with timeout handling.

    Example:
        >>> invoker = LocalInvoker(timeout=30.0)
        >>> result = await invoker.invoke(my_func, config, input_data)
    """

    def __init__(
        self,
        timeout: float = 60.0,
        max_retries: int = 0,
        injection_mode: str = "context",
        config_param: str = "config",
        **kwargs: Any,
    ) -> None:
        """Initialize local invoker.

        Args:
            timeout: Timeout for individual invocations (seconds)
            max_retries: Maximum number of retries for failed invocations
            injection_mode: Configuration injection mode
            config_param: Parameter name for injection_mode="parameter"
            **kwargs: Additional configuration
        """
        normalized_mode = self._validate_injection_mode(injection_mode)
        normalized_param = (
            self._validate_config_param(config_param)
            if normalized_mode == "parameter"
            else config_param
        )

        super().__init__(timeout, max_retries, **kwargs)
        self.injection_mode = normalized_mode
        self.config_param = normalized_param

        # Get configuration provider
        self._provider = get_provider(
            self.injection_mode, config_param=self.config_param
        )

        logger.debug(f"LocalInvoker using {injection_mode} injection mode")

    async def invoke(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_data: dict[str, Any],
    ) -> InvocationResult:
        """Invoke function locally with configuration.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_data: Input data for the function

        Returns:
            InvocationResult with output and metadata
        """
        self.validate_function(func)
        self.validate_config(config)
        self.validate_input(input_data)

        start_time = time.time()

        try:
            # Inject configuration into function
            configured_func = self._provider.inject_config(
                func, config, self.config_param
            )

            # Invoke function
            if asyncio.iscoroutinefunction(configured_func):
                output = await self._invoke_async(
                    configured_func, input_data, start_time
                )
            else:
                output = await self._invoke_sync(
                    configured_func, input_data, start_time
                )

            end_time = time.time()
            execution_time = end_time - start_time

            metadata = self.create_metadata(
                start_time,
                end_time,
                injection_mode=self.injection_mode,
                function_name=func.__name__,
                async_function=asyncio.iscoroutinefunction(configured_func),
            )

            return InvocationResult(
                result=output,
                execution_time=execution_time,
                metadata=metadata,
                is_successful=True,
            )

        except TimeoutError:
            end_time = time.time()
            execution_time = end_time - start_time
            error_msg = f"Function call timed out after {self.timeout}s"

            logger.warning(error_msg)

            return InvocationResult(
                error=error_msg,
                execution_time=execution_time,
                metadata=self.create_metadata(
                    start_time, end_time, timeout_occurred=True
                ),
                is_successful=False,
            )

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            error_msg = f"Function call failed: {e}"

            # Fail fast on API key errors - don't retry 309 times
            lowered = str(e).lower()
            if any(
                token in lowered
                for token in ("api key", "api_key", "authentication", "openai_api_key")
            ):
                raise APIKeyError(
                    f"API key error detected. Set the required API key environment "
                    f"variable or use TRAIGENT_MOCK_LLM=true for testing. "
                    f"Original error: {e}"
                ) from e

            logger.warning(error_msg)

            return InvocationResult(
                error=error_msg,
                execution_time=execution_time,
                metadata=self.create_metadata(
                    start_time, end_time, exception_type=type(e).__name__
                ),
                is_successful=False,
            )

    async def _invoke_async(
        self, func: Callable[..., Any], input_data: dict[str, Any], start_time: float
    ) -> Any:
        """Invoke async function with timeout."""
        if self.timeout:
            return await asyncio.wait_for(func(**input_data), timeout=self.timeout)
        else:
            return await func(**input_data)

    async def _invoke_sync(
        self, func: Callable[..., Any], input_data: dict[str, Any], start_time: float
    ) -> Any:
        """Invoke sync function in thread pool with timeout."""
        loop = asyncio.get_event_loop()

        if self.timeout:
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(**input_data)),
                timeout=self.timeout,
            )
        else:
            return await loop.run_in_executor(None, lambda: func(**input_data))

    async def invoke_batch(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        input_batch: list[dict[str, Any]],
    ) -> list[InvocationResult]:
        """Invoke function on multiple inputs sequentially.

        Args:
            func: Function to invoke
            config: Configuration parameters
            input_batch: List of input data dictionaries

        Returns:
            List of InvocationResult objects
        """
        if not input_batch:
            return []

        logger.debug(f"Invoking function on batch of {len(input_batch)} inputs")

        results = []
        for i, input_data in enumerate(input_batch):
            try:
                result = await self.invoke(func, config, input_data)
                results.append(result)

                if not result.is_successful:
                    logger.warning(f"Batch item {i} failed: {result.error}")

            except Exception as e:
                logger.error(f"Batch item {i} raised exception: {e}")
                results.append(
                    InvocationResult(
                        error=str(e),
                        is_successful=False,
                        metadata={"batch_index": i},
                    )
                )

        successful = sum(1 for r in results if r.is_successful)
        logger.info(f"Batch completed: {successful}/{len(results)} successful")

        return results

    def supports_streaming(self) -> bool:
        """Local invoker does not support streaming by default."""
        return False

    def supports_batch(self) -> bool:
        """Local invoker supports batch processing."""
        return True

    @staticmethod
    def _validate_injection_mode(injection_mode: str) -> str:
        """Ensure injection mode is a non-empty string."""
        if not isinstance(injection_mode, str) or not injection_mode.strip():
            raise InvocationError("injection_mode must be a non-empty string")
        return injection_mode.strip()

    @staticmethod
    def _validate_config_param(config_param: str) -> str:
        """Ensure parameter injection uses a valid parameter name."""
        if not isinstance(config_param, str) or not config_param.strip():
            raise InvocationError("config_param must be a non-empty string")
        return config_param.strip()

    def validate_function(self, func: Callable[..., Any]) -> None:
        """Validate function can be invoked locally."""
        super().validate_function(func)

        # Check if provider can handle this function
        if not self._provider.supports_function(func):
            if self.injection_mode == "parameter":
                # For parameter injection, check if function has the required parameter
                sig = inspect.signature(func)
                if self.config_param not in sig.parameters:
                    raise InvocationError(
                        f"Function {func.__name__} does not have parameter '{self.config_param}' "
                        f"required for parameter injection. Available parameters: {list(sig.parameters.keys())}"
                    )
            else:
                # Other modes should work with any function
                pass
