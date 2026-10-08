"""LangChain response interceptor for capturing token metadata.

This module provides utilities to capture LangChain response metadata
that would otherwise be lost when functions return only strings.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _CaptureBucket:
    """One trial's captured responses.

    Deliberately a MUTABLE object held by a ``ContextVar``: sync agent and
    metric functions run through ``copy_context().run(...)``
    (``evaluators/base.py``), and a copied context does not propagate
    *rebinding* a ContextVar back to the caller -- but it does share this
    object, so appends made in the worker thread are visible to the trial that
    opened the scope.  Storing the list directly (rather than an id to look up
    in a registry) is also self-cleaning: the bucket dies with the scope, so
    concurrent trials cannot accumulate entries in a process-global map.
    """

    responses: list[Any] = field(default_factory=list)
    by_key: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Provider/model versions observed from REAL provider responses in this
    #: trial (content identity v1, ``ObservedProviderVersionV1``), keyed by
    #: ``(provider, requested_model, response_model, system_fingerprint)``.
    #: Deliberately NOT emptied by :meth:`LangChainMetadataCapture.clear`,
    #: which drains per-example spend; observations describe the whole trial.
    observed: dict[tuple[str, str, str | None, str | None], int] = field(
        default_factory=dict
    )

    def observed_provider_versions(self) -> list[dict[str, Any]]:
        """This trial's observations as an ``ObservedProviderVersionV1`` list."""
        from traigent.identity.provider_versions import observations_payload

        with self.lock:
            return observations_payload(dict(self.observed))


#: The capture scope for the trial running in this context, or ``None`` when no
#: scope is active.  ``None`` keeps the pre-#2387 process-global behaviour so
#: callers outside a trial (and any third-party use of the interceptor) are
#: unaffected.
_capture_scope: ContextVar[_CaptureBucket | None] = ContextVar(
    "traigent_capture_scope", default=None
)


#: True while an instrumented wrapper (LangChain ``ChatOpenAI.invoke``,
#: ``litellm.completion``/``acompletion``) is running and will capture its own
#: response.  The ``openai`` SDK call such a wrapper makes underneath must not
#: be captured a second time, or the example is charged twice (Traigent#2441).
_instrumented_call: ContextVar[bool] = ContextVar(
    "traigent_instrumented_provider_call", default=False
)


@contextmanager
def instrumented_provider_call() -> Iterator[None]:
    """Mark the enclosed provider call as captured by an outer wrapper."""
    token = _instrumented_call.set(True)
    try:
        yield
    finally:
        _instrumented_call.reset(token)


def inside_instrumented_provider_call() -> bool:
    """True when an outer wrapper will capture the current provider call."""
    return _instrumented_call.get()


class capture_scope:
    """Give the enclosing trial its own capture buffer (Traigent#2387).

    Concurrent trials are coroutines gathered on ONE event loop, so before this
    existed they all appended to, and drained, a single process-global list: a
    judge call made by one trial could be charged to another, or -- when the
    neighbour drained first -- a trial with real judge spend could be charged
    ``0.0``.  ``threading.local()`` cannot fix that, because those coroutines
    share a thread; ownership has to be per-context.

    Usable as ``with`` or ``async with``.  Nesting is safe: the innermost scope
    wins and the outer one is restored on exit.
    """

    __slots__ = ("_token",)

    def __enter__(self) -> _CaptureBucket:
        bucket = _CaptureBucket()
        self._token = _capture_scope.set(bucket)
        return bucket

    def __exit__(self, *exc_info: Any) -> None:
        _capture_scope.reset(self._token)

    async def __aenter__(self) -> _CaptureBucket:
        return self.__enter__()

    async def __aexit__(self, *exc_info: Any) -> None:
        self.__exit__(*exc_info)


#: How many recent responses the per-thread ``get_last_response`` history
#: keeps. Trial cost never reads it (it reads the capture scope), so this only
#: bounds memory on long-lived worker threads (#2444).
_LAST_RESPONSE_HISTORY_LIMIT = 64


class LangChainMetadataCapture:
    """Thread-safe storage for LangChain response metadata."""

    def __init__(self) -> None:
        self._storage = threading.local()
        self._lock = threading.Lock()
        # Keep all responses in order
        self._all_responses: list[Any] = []
        self._response_lock = threading.Lock()
        # Map correlation keys (e.g., example_id) to responses
        self._by_key: dict[str, Any] = {}
        self._by_key_lock = threading.Lock()
        # Current correlation key in thread local
        self._key_local = threading.local()

    def set_last_response(self, response: Any) -> None:
        """Store the last LangChain response."""
        with self._lock:
            if not hasattr(self._storage, "responses"):
                self._storage.responses = []
            self._storage.responses.append(response)
            # This per-thread list feeds only ``get_last_response`` (which pops
            # one element) and is otherwise drained only by ``clear``; trial
            # spend is read from the scoped bucket below. Unbounded, it held a
            # reference to every response a long-lived worker thread ever saw
            # (#2444), so keep only the most recent ones.
            overflow = len(self._storage.responses) - _LAST_RESPONSE_HISTORY_LIMIT
            if overflow > 0:
                del self._storage.responses[:overflow]
            logger.debug("Captured LangChain response with metadata")

        key = getattr(self._key_local, "current_key", None)

        # Route into the trial's own buffer when one is active (Traigent#2387).
        # Without a scope this falls back to the process-global list, which is
        # the historical behaviour for callers outside a trial.
        bucket = _capture_scope.get()
        if bucket is not None:
            with bucket.lock:
                bucket.responses.append(response)
                if key is not None:
                    bucket.by_key[key] = response
            return

        with self._response_lock:
            self._all_responses.append(response)

        if key is not None:
            with self._by_key_lock:
                self._by_key[key] = response

    def get_last_response(self) -> Any | None:
        """Get and clear the last LangChain response."""
        with self._lock:
            if hasattr(self._storage, "responses") and self._storage.responses:
                response = self._storage.responses.pop()
                logger.debug("Retrieved captured LangChain response")
                return response
            return None

    def get_all_responses(self) -> list[Any]:
        """Get all captured responses (for batch processing).

        Reads only THIS trial's responses when a capture scope is active, so a
        concurrently running trial's spend can never be drained here.
        """
        bucket = _capture_scope.get()
        if bucket is not None:
            with bucket.lock:
                return bucket.responses.copy()
        with self._response_lock:
            return self._all_responses.copy()

    def clear(self) -> None:
        """Clear stored responses for the active scope (or globally).

        Scoped to this trial when a capture scope is active: clearing must not
        discard a concurrently running trial's not-yet-folded spend.
        """
        with self._lock:
            if hasattr(self._storage, "responses"):
                self._storage.responses.clear()

        bucket = _capture_scope.get()
        if bucket is not None:
            with bucket.lock:
                bucket.responses.clear()
                bucket.by_key.clear()
        else:
            with self._response_lock:
                self._all_responses.clear()
            with self._by_key_lock:
                self._by_key.clear()

        if hasattr(self._key_local, "current_key"):
            self._key_local.current_key = None

    # Key management
    def set_current_key(self, key: Any) -> None:
        self._key_local.current_key = key

    def clear_current_key(self) -> None:
        self._key_local.current_key = None

    def get_by_key(self, key: Any) -> Any:
        # Example ids are not unique ACROSS trials -- every trial evaluates the
        # same dataset -- so an unscoped map also collided between concurrent
        # trials, handing one trial another's response for "its" example.
        bucket = _capture_scope.get()
        if bucket is not None:
            with bucket.lock:
                return bucket.by_key.get(key)
        with self._by_key_lock:
            return self._by_key.get(key)


# Global instance for metadata capture
_metadata_capture = LangChainMetadataCapture()


def capture_langchain_response(response: Any) -> Any:
    """Capture a LangChain response for metadata extraction.

    This should be called immediately after getting a response from LangChain
    but before returning just the string content.
    """
    _metadata_capture.set_last_response(response)
    return response


def capture_observed_response(
    response: Any, *, provider: str, requested_model: Any
) -> Any:
    """Capture a REAL provider response and record its observed model version.

    Call this only on the non-mock path: a mock response witnesses nothing
    about what a provider served. The observation lands in the active trial
    capture scope; outside a scope it is dropped (there is no trial to
    attribute it to). Recording never raises into the caller's LLM call.
    """
    bucket = _capture_scope.get()
    if bucket is not None:
        try:
            from traigent.identity.provider_versions import observation_key

            key = observation_key(
                response, provider=provider, requested_model=requested_model
            )
            if key is not None:
                with bucket.lock:
                    if key in bucket.observed or len(bucket.observed) < 256:
                        bucket.observed[key] = bucket.observed.get(key, 0) + 1
        except Exception:  # noqa: BLE001 - observation is best-effort metadata
            logger.debug("Could not record an observed provider version")
    return capture_langchain_response(response)


def requested_model_of(client: Any) -> Any:
    """The model a LangChain chat client was configured to request, if exposed."""
    for attr in ("model_name", "model", "model_id"):
        try:
            value = getattr(client, attr, None)
        except Exception:  # noqa: BLE001
            value = None
        if isinstance(value, str) and value:
            return value
    return None


def get_captured_response() -> Any | None:
    """Retrieve the last captured LangChain response."""
    return _metadata_capture.get_last_response()


def get_all_captured_responses() -> list[Any]:
    """Get all captured responses for batch processing."""
    return _metadata_capture.get_all_responses()


def get_captured_response_by_key(key: Any) -> Any | None:
    """Get captured response by correlation key (e.g., example_id)."""
    return _metadata_capture.get_by_key(key)


def clear_captured_responses() -> None:
    """Clear all captured responses."""
    _metadata_capture.clear()


@contextmanager
def langchain_metadata_context():
    """Context manager for capturing LangChain metadata within a scope."""
    try:
        clear_captured_responses()
        yield _metadata_capture
    finally:
        clear_captured_responses()


@contextmanager
def capture_key(key: Any):
    """Context manager to associate subsequent captured responses with a key."""
    try:
        _metadata_capture.set_current_key(key)
        yield
    finally:
        _metadata_capture.clear_current_key()


def _mock_enabled(provider: str) -> bool:
    from traigent.integrations.utils.mock_adapter import MockAdapter

    return MockAdapter.is_mock_enabled(provider)


def _mock_ai_message(provider: str, client: Any, *, chunk: bool = False) -> Any:
    """Build the canned LangChain message mock mode returns for ``provider``.

    Shared by every patched entry point (``invoke``, ``ainvoke``, ``stream``,
    ``astream``) so they cannot drift apart (#2424). ``chunk=True`` returns an
    ``AIMessageChunk`` for the streaming methods. The response is captured for
    metadata extraction, the same as the historical ``invoke`` mock branch.
    """
    from traigent.integrations.utils.mock_adapter import MockAdapter

    if chunk:
        from langchain_core.messages import AIMessageChunk as message_cls
    else:
        from langchain_core.messages import AIMessage as message_cls

    model_name = requested_model_of(client) or "mock-model"
    mock_data = MockAdapter.get_mock_response(provider, model=str(model_name))
    usage = mock_data["usage"]
    if provider == "openai":
        content = mock_data["choices"][0]["message"]["content"]
        response_metadata = {
            "model_name": mock_data["model"],
            "finish_reason": mock_data["choices"][0]["finish_reason"],
        }
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]
    else:  # anthropic and bedrock share the Messages response shape
        content = mock_data["content"][0]["text"]
        response_metadata = {
            "model_name": mock_data["model"],
            "stop_reason": mock_data["stop_reason"],
        }
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
    response_metadata["response_time_ms"] = 0.0
    response = message_cls(
        content=content,
        response_metadata=response_metadata,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        },
    )
    capture_langchain_response(response)
    return response


def _create_ainvoke_wrapper(
    original_meth: Any, provider: str = "unknown", *, instrumented: bool = False
) -> Any:
    """Mock ``ainvoke`` in mock mode (#2424); otherwise capture it the way the
    patched ``invoke`` does (#2445).

    ``abatch`` and other async entry points that go through ``ainvoke`` are
    covered by the same wrapper.
    """

    async def ainvoke_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _mock_enabled(provider):
            return _mock_ai_message(provider, self)

        start_time = time.perf_counter()
        if instrumented:
            with instrumented_provider_call():
                response = await original_meth(self, *args, **kwargs)
        else:
            response = await original_meth(self, *args, **kwargs)
        response_time_ms = (time.perf_counter() - start_time) * 1000

        if not hasattr(response, "response_metadata"):
            response.response_metadata = {}
        response.response_metadata["response_time_ms"] = response_time_ms
        capture_observed_response(
            response, provider=provider, requested_model=requested_model_of(self)
        )
        return response

    return ainvoke_wrapper


def _create_stream_wrapper(original_meth: Any, provider: str = "unknown") -> Any:
    def stream_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _mock_enabled(provider):
            yield _mock_ai_message(provider, self, chunk=True)
            return
        start_time = time.perf_counter()
        last = None
        for chunk in original_meth(self, *args, **kwargs):
            last = chunk
            yield chunk
        if last is not None:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            if not hasattr(last, "response_metadata"):
                last.response_metadata = {}
            last.response_metadata["response_time_ms"] = response_time_ms
            capture_observed_response(
                last, provider=provider, requested_model=requested_model_of(self)
            )

    return stream_wrapper


def _create_astream_wrapper(original_meth: Any, provider: str = "unknown") -> Any:
    async def astream_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _mock_enabled(provider):
            yield _mock_ai_message(provider, self, chunk=True)
            return
        start_time = time.perf_counter()
        last = None
        async for chunk in original_meth(self, *args, **kwargs):
            last = chunk
            yield chunk
        if last is not None:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            if not hasattr(last, "response_metadata"):
                last.response_metadata = {}
            last.response_metadata["response_time_ms"] = response_time_ms
            capture_observed_response(
                last, provider=provider, requested_model=requested_model_of(self)
            )

    return astream_wrapper


_METHOD_WRAPPERS: dict[str, Callable[..., Any]] = {
    "ainvoke": _create_ainvoke_wrapper,
    "stream": _create_stream_wrapper,
    "astream": _create_astream_wrapper,
}
#: Patched alongside ``invoke`` on every supported chat model class.
_EXTRA_METHODS = (
    ("ainvoke", "_traigent_patched_ainvoke"),
    ("stream", "_traigent_patched_stream"),
    ("astream", "_traigent_patched_astream"),
)


def _patch_langchain_bedrock_model(model_cls: Any, class_name: str) -> bool:
    """Patch one langchain_aws Bedrock chat class for response capture."""
    patched_any = False

    if getattr(model_cls, "_traigent_patched_invoke", False) is False:
        original_invoke = model_cls.invoke

        def invoke_with_capture_bedrock(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Wrapped invoke method that captures Bedrock usage metadata."""
            from traigent.integrations.utils.mock_adapter import MockAdapter

            if MockAdapter.is_mock_enabled("bedrock"):
                return _mock_ai_message("bedrock", self)

            start_time = time.perf_counter()
            response = original_invoke(self, *args, **kwargs)
            response_time_ms = (time.perf_counter() - start_time) * 1000

            if not hasattr(response, "response_metadata"):
                response.response_metadata = {}
            response.response_metadata["response_time_ms"] = response_time_ms

            capture_observed_response(
                response, provider="bedrock", requested_model=requested_model_of(self)
            )
            logger.debug(
                "Captured %s invoke usage: %s, response_time_ms: %.2f",
                class_name,
                getattr(response, "usage_metadata", None),
                response_time_ms,
            )
            return response

        model_cls.invoke = invoke_with_capture_bedrock
        model_cls._traigent_patched_invoke = True
        logger.info("✅ Patched %s.invoke for metadata capture", class_name)
        patched_any = True

    for meth_name, flag in _EXTRA_METHODS:
        if hasattr(model_cls, meth_name) and not getattr(model_cls, flag, False):
            original_meth = getattr(model_cls, meth_name)
            setattr(
                model_cls,
                meth_name,
                _METHOD_WRAPPERS[meth_name](original_meth, provider="bedrock"),
            )
            setattr(model_cls, flag, True)
            logger.info("✅ Patched %s.%s for metadata capture", class_name, meth_name)
            patched_any = True

    return patched_any


def patch_langchain_for_metadata_capture() -> bool:
    """Monkey-patch LangChain to automatically capture response metadata.

    This patches both ChatAnthropic and ChatOpenAI invoke methods to capture responses.
    """
    patched_any = False

    # Patch ChatAnthropic
    try:
        from langchain_anthropic import ChatAnthropic

        # Store original method
        if getattr(ChatAnthropic, "_traigent_patched_invoke", False) is False:
            original_invoke = ChatAnthropic.invoke

            def invoke_with_capture_anthropic(
                self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                """Wrapped invoke method that captures response with timing."""
                # Check if mock mode is enabled — return mock response without API call
                from traigent.integrations.utils.mock_adapter import MockAdapter

                if MockAdapter.is_mock_enabled("anthropic"):
                    return _mock_ai_message("anthropic", self)

                start_time = time.perf_counter()
                response = original_invoke(self, *args, **kwargs)
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Inject timing into response metadata
                if not hasattr(response, "response_metadata"):
                    response.response_metadata = {}
                response.response_metadata["response_time_ms"] = response_time_ms

                capture_observed_response(
                    response,
                    provider="anthropic",
                    requested_model=requested_model_of(self),
                )
                logger.debug(
                    f"Captured ChatAnthropic invoke usage: {getattr(response, 'usage_metadata', None)}, "
                    f"response_time_ms: {response_time_ms:.2f}"
                )
                return response

            ChatAnthropic.invoke = invoke_with_capture_anthropic
            ChatAnthropic._traigent_patched_invoke = True
            logger.info("✅ Patched ChatAnthropic.invoke for metadata capture")
            patched_any = True

        # Patch stream methods if available
        for meth_name, flag in _EXTRA_METHODS:
            if hasattr(ChatAnthropic, meth_name) and not getattr(
                ChatAnthropic, flag, False
            ):
                original_meth = getattr(ChatAnthropic, meth_name)
                setattr(
                    ChatAnthropic,
                    meth_name,
                    _METHOD_WRAPPERS[meth_name](original_meth, provider="anthropic"),
                )
                setattr(ChatAnthropic, flag, True)
                logger.info(
                    f"✅ Patched ChatAnthropic.{meth_name} for metadata capture"
                )

    except ImportError:
        logger.debug("ChatAnthropic not available, skipping")
    except Exception as e:
        logger.error(f"Failed to patch ChatAnthropic: {e}")

    # Patch ChatOpenAI
    try:
        from langchain_openai import ChatOpenAI

        if getattr(ChatOpenAI, "_traigent_patched_invoke", False) is False:
            original_invoke_openai = ChatOpenAI.invoke

            def invoke_with_capture_openai(self: Any, *args: Any, **kwargs: Any) -> Any:
                """Wrapped invoke method that captures response with timing."""
                # Check if mock mode is enabled — return mock response without API call
                from traigent.integrations.utils.mock_adapter import MockAdapter

                if MockAdapter.is_mock_enabled("openai"):
                    return _mock_ai_message("openai", self)

                start_time = time.perf_counter()
                with instrumented_provider_call():
                    response = original_invoke_openai(self, *args, **kwargs)
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Inject timing into response metadata
                if not hasattr(response, "response_metadata"):
                    response.response_metadata = {}
                response.response_metadata["response_time_ms"] = response_time_ms

                capture_observed_response(
                    response,
                    provider="openai",
                    requested_model=requested_model_of(self),
                )
                logger.debug(
                    f"Captured ChatOpenAI invoke usage: {getattr(response, 'usage_metadata', None)}, "
                    f"response_time_ms: {response_time_ms:.2f}"
                )
                return response

            ChatOpenAI.invoke = invoke_with_capture_openai
            ChatOpenAI._traigent_patched_invoke = True
            logger.info("✅ Patched ChatOpenAI.invoke for metadata capture")
            patched_any = True

        for meth_name, flag in _EXTRA_METHODS:
            if hasattr(ChatOpenAI, meth_name) and not getattr(ChatOpenAI, flag, False):
                original_meth = getattr(ChatOpenAI, meth_name)
                setattr(
                    ChatOpenAI,
                    meth_name,
                    _METHOD_WRAPPERS[meth_name](
                        original_meth,
                        provider="openai",
                        # Mirrors invoke: the openai call LangChain makes
                        # underneath must not be captured a second time.
                        **({"instrumented": True} if meth_name == "ainvoke" else {}),
                    ),
                )
                setattr(ChatOpenAI, flag, True)
                logger.info(f"✅ Patched ChatOpenAI.{meth_name} for metadata capture")

    except ImportError:
        logger.debug("ChatOpenAI not available, skipping")
    except Exception as e:
        logger.error(f"Failed to patch ChatOpenAI: {e}")

    # Patch langchain_aws Bedrock chat models
    try:
        import langchain_aws

        for class_name in ("ChatBedrock", "ChatBedrockConverse"):
            model_cls = getattr(langchain_aws, class_name, None)
            if model_cls is None:
                logger.debug("langchain_aws.%s not available, skipping", class_name)
                continue
            if _patch_langchain_bedrock_model(model_cls, f"langchain_aws.{class_name}"):
                patched_any = True

    except ImportError:
        logger.debug("langchain_aws not available, skipping Bedrock")
    except Exception as e:
        logger.error(f"Failed to patch langchain_aws Bedrock models: {e}")

    if not patched_any:
        logger.debug("No LangChain models could be patched for metadata capture")
        return False

    return True
