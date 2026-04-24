"""LiteLLM response interceptor for capturing token metadata.

This module monkey-patches litellm.completion and litellm.acompletion to
capture response metadata (tokens, cost, timing) that would otherwise be
lost when user functions return only strings.

Without this interceptor, Traigent falls back to estimating tokens via
len(text) // 4 — producing inaccurate cost data.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import time
from typing import Any

from traigent.utils.langchain_interceptor import capture_langchain_response
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def patch_litellm_for_metadata_capture() -> bool:
    """Monkey-patch LiteLLM to automatically capture response metadata.

    Patches litellm.completion and litellm.acompletion to intercept
    ModelResponse objects before the user's function discards them.
    Captured responses flow through the existing OpenAIResponseHandler
    since LiteLLM uses OpenAI-compatible response format.

    Returns:
        True if at least one function was patched, False otherwise.
    """
    patched_any = False

    try:
        import litellm
    except ImportError:
        logger.debug("LiteLLM not available, skipping interceptor")
        return False

    # Patch litellm.completion (sync)
    if not getattr(litellm, "_traigent_patched_completion", False):
        original_completion = litellm.completion

        def completion_with_capture(*args: Any, **kwargs: Any) -> Any:
            """Wrapped completion that captures response with timing."""
            from traigent.integrations.utils.mock_adapter import MockAdapter

            if MockAdapter.is_mock_enabled("litellm"):
                mock_data = MockAdapter.get_mock_response(
                    "litellm", model=kwargs.get("model", args[0] if args else "mock-model")
                )
                return mock_data

            start_time = time.perf_counter()
            response = original_completion(*args, **kwargs)
            response_time_ms = (time.perf_counter() - start_time) * 1000

            # Skip capture for streaming responses — usage is not populated
            # until the stream is fully consumed in user code after we return.
            if kwargs.get("stream", False):
                return response

            # Inject timing into response for the handler chain
            response.response_time_ms = response_time_ms

            capture_langchain_response(response)
            logger.debug(
                "Captured litellm.completion response: model=%s, "
                "tokens=%s, response_time_ms=%.2f",
                getattr(response, "model", "unknown"),
                getattr(response, "usage", None),
                response_time_ms,
            )
            return response

        litellm.completion = completion_with_capture
        litellm._traigent_patched_completion = True
        logger.info("Patched litellm.completion for metadata capture")
        patched_any = True

    # Patch litellm.acompletion (async)
    if not getattr(litellm, "_traigent_patched_acompletion", False):
        original_acompletion = litellm.acompletion

        async def acompletion_with_capture(*args: Any, **kwargs: Any) -> Any:
            """Wrapped async completion that captures response with timing."""
            from traigent.integrations.utils.mock_adapter import MockAdapter

            if MockAdapter.is_mock_enabled("litellm"):
                mock_data = MockAdapter.get_mock_response(
                    "litellm", model=kwargs.get("model", args[0] if args else "mock-model")
                )
                return mock_data

            start_time = time.perf_counter()
            response = await original_acompletion(*args, **kwargs)
            response_time_ms = (time.perf_counter() - start_time) * 1000

            # Skip capture for streaming responses — usage is not populated
            # until the stream is fully consumed in user code after we return.
            if kwargs.get("stream", False):
                return response

            # Inject timing into response for the handler chain
            response.response_time_ms = response_time_ms

            capture_langchain_response(response)
            logger.debug(
                "Captured litellm.acompletion response: model=%s, "
                "tokens=%s, response_time_ms=%.2f",
                getattr(response, "model", "unknown"),
                getattr(response, "usage", None),
                response_time_ms,
            )
            return response

        litellm.acompletion = acompletion_with_capture
        litellm._traigent_patched_acompletion = True
        logger.info("Patched litellm.acompletion for metadata capture")
        patched_any = True

    if not patched_any:
        logger.debug("LiteLLM functions already patched, skipping")

    return patched_any
