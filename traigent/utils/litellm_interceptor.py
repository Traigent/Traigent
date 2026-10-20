"""LiteLLM response interceptor for capturing token metadata.

This module monkey-patches litellm.completion and litellm.acompletion to
capture response metadata (tokens, cost, timing) that would otherwise be
lost when user functions return only strings.

Without this interceptor, Traigent falls back to estimating tokens via
len(text) // 4 — producing inaccurate cost data.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import sys
import time
from typing import Any

from traigent.utils.langchain_interceptor import (
    capture_langchain_response,
    capture_observed_response,
    instrumented_provider_call,
)
from traigent.utils.logging import configure_litellm_logging, get_logger

logger = get_logger(__name__)


def _litellm_provider(response: Any) -> str:
    """The upstream provider LiteLLM routed a call to, when it reports one."""
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        provider = hidden.get("custom_llm_provider")
        if isinstance(provider, str) and provider:
            return provider
    return "litellm"


def _rebind_imported_names(original: Any, wrapper: Any) -> int:
    """Point module-level names that still hold ``original`` at ``wrapper``.

    ``from litellm import completion`` copies the function object into the
    importing module at import time, so replacing ``litellm.completion``
    afterwards does not reach it (#2515). The scan is bounded: it looks only
    at the top-level namespace of already-imported modules, matches by object
    identity (never by name), and skips litellm's own submodules so its
    internals keep their original call graph. Names bound after the patch
    already receive the wrapper.

    Returns the number of names rebound.
    """
    rebound = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod_name.startswith("litellm."):
            continue
        namespace = getattr(mod, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for attr, value in list(namespace.items()):
            if value is original:
                try:
                    namespace[attr] = wrapper
                    rebound += 1
                except Exception:  # noqa: BLE001 - read-only namespaces are skipped
                    logger.debug("Could not rebind an imported litellm name")
    return rebound


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

    configure_litellm_logging(litellm_module=litellm)

    # Patch litellm.completion (sync)
    if not getattr(litellm, "_traigent_patched_completion", False):
        original_completion = getattr(litellm, "completion", None)
        if original_completion is None:
            logger.debug("litellm.completion not available, skipping sync interceptor")
        else:

            def completion_with_capture(*args: Any, **kwargs: Any) -> Any:
                """Wrapped completion that captures response with timing."""
                from traigent.integrations.utils.mock_adapter import MockAdapter

                if MockAdapter.is_mock_enabled("litellm"):
                    mock_data = MockAdapter.get_mock_response(
                        "litellm",
                        model=kwargs.get("model", args[0] if args else "mock-model"),
                    )
                    mock_data.response_time_ms = 0.0
                    capture_langchain_response(mock_data)
                    return mock_data

                start_time = time.perf_counter()
                with instrumented_provider_call():
                    response = original_completion(*args, **kwargs)
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Skip capture for streaming responses — usage is not populated
                # until the stream is fully consumed in user code after we return.
                if kwargs.get("stream", False):
                    return response

                # Inject timing into response for the handler chain
                response.response_time_ms = response_time_ms

                capture_observed_response(
                    response,
                    provider=_litellm_provider(response),
                    requested_model=kwargs.get("model", args[0] if args else None),
                )
                logger.debug(
                    "Captured litellm.completion response: model=%s, "
                    "tokens=%s, response_time_ms=%.2f",
                    getattr(response, "model", "unknown"),
                    getattr(response, "usage", None),
                    response_time_ms,
                )
                return response

            completion_with_capture._traigent_original = original_completion  # type: ignore[attr-defined]
            litellm.completion = completion_with_capture
            litellm._traigent_patched_completion = True
            logger.info("Patched litellm.completion for metadata capture")
            patched_any = True

    # Patch litellm.acompletion (async)
    if not getattr(litellm, "_traigent_patched_acompletion", False):
        original_acompletion = getattr(litellm, "acompletion", None)
        if original_acompletion is None:
            logger.debug(
                "litellm.acompletion not available, skipping async interceptor"
            )
        else:

            async def acompletion_with_capture(*args: Any, **kwargs: Any) -> Any:
                """Wrapped async completion that captures response with timing."""
                from traigent.integrations.utils.mock_adapter import MockAdapter

                if MockAdapter.is_mock_enabled("litellm"):
                    mock_data = MockAdapter.get_mock_response(
                        "litellm",
                        model=kwargs.get("model", args[0] if args else "mock-model"),
                    )
                    mock_data.response_time_ms = 0.0
                    capture_langchain_response(mock_data)
                    return mock_data

                start_time = time.perf_counter()
                with instrumented_provider_call():
                    response = await original_acompletion(*args, **kwargs)
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Skip capture for streaming responses — usage is not populated
                # until the stream is fully consumed in user code after we return.
                if kwargs.get("stream", False):
                    return response

                # Inject timing into response for the handler chain
                response.response_time_ms = response_time_ms

                capture_observed_response(
                    response,
                    provider=_litellm_provider(response),
                    requested_model=kwargs.get("model", args[0] if args else None),
                )
                logger.debug(
                    "Captured litellm.acompletion response: model=%s, "
                    "tokens=%s, response_time_ms=%.2f",
                    getattr(response, "model", "unknown"),
                    getattr(response, "usage", None),
                    response_time_ms,
                )
                return response

            acompletion_with_capture._traigent_original = original_acompletion  # type: ignore[attr-defined]
            litellm.acompletion = acompletion_with_capture
            litellm._traigent_patched_acompletion = True
            logger.info("Patched litellm.acompletion for metadata capture")
            patched_any = True

    # Names bound by ``from litellm import completion`` before (or between)
    # patch calls keep the original function; rebind them on every call.
    for attr in ("completion", "acompletion"):
        wrapper = getattr(litellm, attr, None)
        original = getattr(wrapper, "_traigent_original", None)
        if original is not None and _rebind_imported_names(original, wrapper):
            logger.debug("Rebound imported litellm.%s names to the interceptor", attr)

    if not patched_any:
        logger.debug("LiteLLM functions already patched, skipping")

    return patched_any
