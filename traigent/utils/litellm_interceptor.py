"""LiteLLM response interceptor for capturing token metadata.

This module monkey-patches litellm.completion and litellm.acompletion to
capture response metadata (tokens, cost, timing) that would otherwise be
lost when user functions return only strings.

Without this interceptor, Traigent falls back to estimating tokens via
len(text) // 4 — producing inaccurate cost data.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import functools
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from traigent.utils.langchain_interceptor import (
    capture_langchain_response,
    capture_observed_response,
    classify_import_failure,
    instrumented_provider_call,
)
from traigent.utils.logging import configure_litellm_logging, get_logger
from traigent.utils.mock_provenance import note_real_call

logger = get_logger(__name__)


def _litellm_provider(response: Any) -> str:
    """The upstream provider LiteLLM routed a call to, when it reports one."""
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        provider = hidden.get("custom_llm_provider")
        if isinstance(provider, str) and provider:
            return provider
    return "litellm"


class MockModeBypassError(RuntimeError):
    """A LiteLLM call reached the provider path under mock mode without mock."""


def _mock_mode_pre_call_guard(_input: Any = None) -> bool:
    """LiteLLM ``pre_call_rules`` entry that fails closed under mock mode.

    LiteLLM runs every registered pre-call rule inside ``completion`` /
    ``acompletion`` before any provider transport, for every reference to the
    function (closures, default arguments, stored callables) -- unlike the
    module-global rebinding, which only reaches names still in a namespace
    (#2515). Calls made through the Traigent wrapper return their mock earlier
    and never get here, so a call that does arrive while mock mode is on is a
    bypass and must not reach the provider. The message is content-free.
    """
    _raise_if_mock_bypass()
    return True


_GUARDED_CALL_TYPES = frozenset(
    {"completion", "acompletion", "text_completion", "atext_completion"}
)


def _raise_if_mock_bypass() -> None:
    from traigent.integrations.utils.mock_adapter import MockAdapter

    if MockAdapter.is_mock_enabled("litellm"):
        raise MockModeBypassError(
            "A LiteLLM call bypassed Traigent mock mode and was blocked before "
            "reaching the provider. Call litellm.completion/acompletion after "
            "mock mode is enabled, or re-fetch the function reference."
        )


_guard_install_lock = threading.RLock()


def _install_function_setup_guard(litellm: Any) -> None:
    """Guard ``litellm.utils.function_setup``, the unconditional entry point.

    Every ``@client``-decorated ``completion``/``acompletion`` call looks up
    ``function_setup`` by module-global name before any provider transport,
    whatever the message shape and whether or not ``pre_call_rules`` was
    replaced or skipped (LiteLLM only runs rules for messages that carry a
    ``content`` key). Wrapping it covers stale references of every kind. An
    exception raised here propagates to the caller (verified on 1.100.5 and
    1.103.5). Raises ``RuntimeError`` when the hook point is missing so the
    caller can fail closed.
    """
    utils_mod = getattr(litellm, "utils", None)
    current = getattr(utils_mod, "function_setup", None)
    if utils_mod is None or not callable(current):
        raise RuntimeError(
            "litellm.utils.function_setup is unavailable; cannot guarantee "
            "that mock mode blocks provider calls."
        )
    original_setup: Callable[..., Any] = current
    if getattr(current, "_traigent_mock_guard", False):
        return

    @functools.wraps(current)
    def function_setup_with_mock_guard(
        original_function: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if original_function in _GUARDED_CALL_TYPES:
            _raise_if_mock_bypass()
        return original_setup(original_function, *args, **kwargs)

    function_setup_with_mock_guard._traigent_mock_guard = True  # type: ignore[attr-defined]
    function_setup_with_mock_guard._traigent_original = original_setup  # type: ignore[attr-defined]
    utils_mod.function_setup = function_setup_with_mock_guard


def _install_mock_guard(litellm: Any) -> None:
    """Install the fail-closed mock guards (idempotent).

    Primary: the ``function_setup`` boundary (unconditional). Second layer:
    a ``pre_call_rules`` entry, when LiteLLM exposes the list.
    """
    # Serialized: the check-then-wrap in the guard installers must not run
    # concurrently, or a second wrapper is stacked over the first.
    with _guard_install_lock:
        _install_function_setup_guard(litellm)
        rules = getattr(litellm, "pre_call_rules", None)
        if isinstance(rules, list) and _mock_mode_pre_call_guard not in rules:
            rules.append(_mock_mode_pre_call_guard)


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


def patch_litellm_for_metadata_capture(
    outcomes: dict[str, str] | None = None,
) -> bool:
    """Monkey-patch LiteLLM to automatically capture response metadata.

    Patches litellm.completion and litellm.acompletion to intercept
    ModelResponse objects before the user's function discards them.
    Captured responses flow through the existing OpenAIResponseHandler
    since LiteLLM uses OpenAI-compatible response format.

    Args:
        outcomes: Optional dict filled with ``outcomes["litellm"]`` set to
            ``installed`` / ``already-installed`` / ``not-applicable`` (package
            absent) / ``failed`` (the fail-closed mock guard could not be
            installed). When given, a guard failure is reported there instead
            of raised, so activation code can fail closed itself.

    Returns:
        True if at least one function was patched, False otherwise.
    """
    patched_any = False
    out: dict[str, str] = outcomes if outcomes is not None else {}

    # Env-var mock mode (TRAIGENT_MOCK_LLM) never goes through
    # enable_mock_mode_for_quickstart(); install the provider-host transport
    # guard here so raw openai/anthropic clients are blocked too (#2418).
    # Failure is not swallowed: mock mode must fail closed.
    from traigent.integrations.utils.mock_adapter import MockAdapter

    if MockAdapter.is_mock_enabled("litellm"):
        from traigent.utils.mock_transport_guard import install_mock_transport_guard

        install_mock_transport_guard()

    try:
        import litellm
    except ImportError:
        # Absent only when the package itself cannot be found; a present but
        # broken/incompatible install must not be reported as "not applicable".
        out["litellm"] = classify_import_failure("litellm")
        if out["litellm"] == "failed":
            logger.error("LiteLLM is installed but could not be imported")
        else:
            logger.debug("LiteLLM not available, skipping interceptor")
        return False

    configure_litellm_logging(litellm_module=litellm)

    # Safety net for references the rebinding cannot reach (closures, default
    # arguments, stored callables): fail closed inside LiteLLM itself.
    guard_failed = False
    try:
        _install_mock_guard(litellm)
    except Exception:
        guard_failed = True
        out["litellm"] = "failed"
        from traigent.integrations.utils.mock_adapter import MockAdapter

        if outcomes is None and MockAdapter.is_mock_enabled("litellm"):
            raise
        logger.warning("LiteLLM mock guard could not be installed")

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

                note_real_call()
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

                note_real_call()
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

    # This invocation's final outcome, assigned explicitly so a successful
    # retry with a reused dict does not keep a stale "failed".
    if guard_failed:
        out["litellm"] = "failed"
    else:
        out["litellm"] = "installed" if patched_any else "already-installed"
    if not patched_any:
        logger.debug("LiteLLM functions already patched, skipping")

    return patched_any
