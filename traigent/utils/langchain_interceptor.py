"""LangChain response interceptor for capturing token metadata.

This module provides utilities to capture LangChain response metadata
that would otherwise be lost when functions return only strings.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import importlib.util
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future
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
    #: Owner (example index) of each entry in ``responses``, same order and
    #: length; ``None`` when the response was captured outside any example
    #: ownership scope (:func:`capture_owner`).  Ownership is recorded at
    #: capture time from the capturing context, never inferred from counts.
    owners: list[int | None] = field(default_factory=list)
    #: Model the capturing call REQUESTED (client configuration at the call
    #: site), same order/length as ``responses``; ``None`` when unknown.
    #: Provenance for pricing a response that reports no model of its own.
    requested: list[str | None] = field(default_factory=list)
    by_key: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    sealed: bool = False
    # Futures remain observable after timeout; completion before sealing clears
    # uncertainty about workers still able to incur spend.
    timed_out_workers: list[Future[Any]] = field(default_factory=list)
    incomplete: bool = False
    late: list[tuple[Any, str | None]] = field(default_factory=list)
    finished: bool = False
    #: Provider/model versions observed from REAL provider responses in this
    #: trial (content identity v1, ``ObservedProviderVersionV1``), keyed by
    #: ``(provider, requested_model, response_model, system_fingerprint)``.
    #: Deliberately NOT emptied by :meth:`LangChainMetadataCapture.clear`,
    #: which drains per-example spend; observations describe the whole trial.
    observed: dict[tuple[str, str, str | None, str | None], int] = field(
        default_factory=dict
    )
    #: Provider-call attempt ledger (SDK-41). It lives ONLY on the ROOT
    #: (outermost) bucket and is guarded by the root's lock: nested scopes
    #: share their root's ledger, so there is nothing to merge or forward and
    #: ids are unique across nested and ended scopes. See
    #: ``traigent.utils.vendor_retry``. Counters are independent of the capped
    #: record list.
    root: Any = None
    vendor_attempts: list[dict[str, Any]] = field(default_factory=list)
    vendor_call_count: int = 0
    vendor_attempt_total: int = 0  # counted when an attempt is OPENED
    vendor_done_total: int = 0  # attempts that reached a final outcome
    vendor_retry_total: int = 0
    vendor_known_usage_total: int = 0
    vendor_records_truncated: bool = False

    def observed_provider_versions(self) -> list[dict[str, Any]]:
        """This trial's observations as an ``ObservedProviderVersionV1`` list."""
        from traigent.identity.provider_versions import observations_payload

        with self.lock:
            return observations_payload(dict(self.observed))


_MAX_VENDOR_ATTEMPT_RECORDS = 10_000


def _root_of(bucket: _CaptureBucket | None) -> _CaptureBucket | None:
    return None if bucket is None else (bucket.root or bucket)


def begin_vendor_call() -> int:
    """Number a new logical provider call, from the root bucket (0 outside one)."""
    root = _root_of(_capture_scope.get())
    if root is None:
        return 0
    with root.lock:
        root.vendor_call_count += 1
        return root.vendor_call_count


def current_capture_bucket() -> _CaptureBucket | None:
    """The active trial's capture bucket, if any."""
    return _capture_scope.get()


class VendorAttemptHandle:
    """One opened provider-call attempt, updated in place until it finishes."""

    __slots__ = ("root", "record", "done")

    def __init__(self, root: _CaptureBucket | None, record: dict[str, Any] | None):
        self.root = root
        self.record = record
        self.done = False


def open_vendor_attempt(
    record: dict[str, Any], bucket: _CaptureBucket | None = None
) -> VendorAttemptHandle:
    """Count an attempt the moment it is opened (outcome ``in_progress``).

    An attempt that is abandoned (a stream never consumed, a cancelled
    consumer) therefore stays counted, with unknown usage.
    """
    root = _root_of(bucket if bucket is not None else _capture_scope.get())
    if root is None:
        return VendorAttemptHandle(None, None)
    stored: dict[str, Any] | None = None
    with root.lock:
        root.vendor_attempt_total += 1
        if len(root.vendor_attempts) < _MAX_VENDOR_ATTEMPT_RECORDS:
            stored = {"outcome": "in_progress", "usage": None, **record}
            root.vendor_attempts.append(stored)
        else:
            root.vendor_records_truncated = True
    return VendorAttemptHandle(root, stored)


def finish_vendor_attempt(handle: VendorAttemptHandle, **updates: Any) -> None:
    """Give an opened attempt its final outcome (idempotent, atomic)."""
    root = handle.root
    if root is None:
        return
    with root.lock:
        if handle.done:
            return
        handle.done = True
        root.vendor_done_total += 1
        if updates.get("retried"):
            root.vendor_retry_total += 1
        if updates.get("usage_known"):
            root.vendor_known_usage_total += 1
        if handle.record is not None:
            handle.record.update(updates)


def record_vendor_attempt(record: dict[str, Any]) -> None:
    """Open and immediately finish one attempt (dropped outside a scope)."""
    handle = open_vendor_attempt(record)
    updates = dict(record)
    updates.setdefault("outcome", "success")
    finish_vendor_attempt(handle, **updates)


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


_capture_owned: ContextVar[frozenset[int]] = ContextVar(
    "traigent_langchain_capture_owned", default=frozenset()
)


def _owned_by(model: Any) -> Any:
    """Mark ``model`` as owning the current capture; returns the reset token."""
    return _capture_owned.set(_capture_owned.get() | {id(model)})


@contextmanager
def _capture_ownership(model: Any) -> Iterator[bool]:
    """Yield True when this invoke wrapper should record its own response.

    Normally always True: ``invoke``/``ainvoke`` wrappers record exactly once
    per call, and nested independent calls (even a callback re-invoking the
    SAME model instance) are separate provider calls that must each be
    recorded. The stream wrappers avoid double counting instead by asking
    LangChain up front whether a stream call falls back to invoke/ainvoke
    (:func:`_stream_falls_back`) and not recording in that case.

    Only when ``_should_stream`` is unavailable (``_stream_falls_back``
    returns None) do the stream wrappers mark the instance as owning the
    capture while the wrapped stream runs; this wrapper then yields False so
    the fallback invoke is recorded once, by the stream wrapper. That legacy
    path is instance-scoped and may drop a same-instance re-entrant call.
    """
    yield id(model) not in _capture_owned.get()


def _stream_falls_back(model: Any, *, async_api: bool, kwargs: dict[str, Any]) -> Any:
    """Ask LangChain whether ``stream``/``astream`` will fall back to invoke.

    Mirrors the decision in ``BaseChatModel.stream``/``astream``:
    ``not self._should_stream(async_api=..., **{**kwargs, "stream": True})``.
    Returns True (falls back; the inner invoke wrapper records), False (really
    streams) or None when the check is unavailable or fails (legacy
    instance-scoped ownership applies).
    """
    should_stream = getattr(model, "_should_stream", None)
    if not callable(should_stream):
        return None
    try:
        return not should_stream(async_api=async_api, **{**kwargs, "stream": True})
    except Exception:  # noqa: BLE001 - private LangChain API; degrade, don't fail
        return None


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
        parent = _capture_scope.get()
        bucket = _CaptureBucket()
        bucket.root = parent.root or parent if parent is not None else bucket
        self._token = _capture_scope.set(bucket)
        return bucket

    def __exit__(self, *exc_info: Any) -> None:
        seal_captured_responses()
        finish_captured_responses(discard=True)
        _capture_scope.reset(self._token)

    async def __aenter__(self) -> _CaptureBucket:
        return self.__enter__()

    async def __aexit__(self, *exc_info: Any) -> None:
        self.__exit__(*exc_info)


#: The example (by index) whose evaluation is running in this context, or
#: ``None``.  A ContextVar, so concurrent example tasks on one event loop and
#: sync agent functions run via ``copy_context().run`` (worker threads) each
#: see their own owner.
_capture_owner: ContextVar[int | None] = ContextVar(
    "traigent_capture_owner", default=None
)


@contextmanager
def capture_owner(owner: int) -> Iterator[None]:
    """Attribute every response captured in this context to ``owner`` (#2444).

    Wrap one example's evaluation in this so its LLM calls are recorded as
    owned by that example, whatever the number of calls and whatever order
    concurrent examples complete in.
    """
    token = _capture_owner.set(owner)
    try:
        yield
    finally:
        _capture_owner.reset(token)


# Cap on the per-thread "last response" stack (see set_last_response).
_MAX_THREAD_LOCAL_RESPONSES = 64


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

    def set_last_response(self, response: Any, *, requested_model: Any = None) -> None:
        """Store the last LangChain response.

        ``requested_model`` is the model THIS call asked for, when the call
        site knows it; it is recorded next to the response in the active scope.
        """
        with self._lock:
            if not hasattr(self._storage, "responses"):
                # Bounded: this per-thread stack only serves get_last_response()
                # (newest first). Cost accounting reads the capture-scope bucket
                # / global list below, never this one, so evicting its oldest
                # entries cannot drop a response still needed for spend. (#2444)
                self._storage.responses = deque(maxlen=_MAX_THREAD_LOCAL_RESPONSES)
            self._storage.responses.append(response)
            logger.debug("Captured LangChain response with metadata")

        key = getattr(self._key_local, "current_key", None)

        # Route into the trial's own buffer when one is active (Traigent#2387).
        # Without a scope this falls back to the process-global list, which is
        # the historical behaviour for callers outside a trial.
        bucket = _capture_scope.get()
        if bucket is not None:
            with bucket.lock:
                if bucket.sealed:
                    if not bucket.finished:
                        bucket.late.append(
                            (
                                response,
                                requested_model
                                if isinstance(requested_model, str)
                                else None,
                            )
                        )
                    # Captures after evaluate's final handoff cannot update its
                    # returned result. Live timed-out workers already make that
                    # trial unmeasured; never assume their remaining spend is $0.
                    return
                bucket.responses.append(response)
                bucket.owners.append(_capture_owner.get())
                bucket.requested.append(
                    requested_model
                    if isinstance(requested_model, str) and requested_model
                    else None
                )
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

    def get_response_owners(self) -> list[int | None] | None:
        """Owner of each response ``get_all_responses`` returns, same order.

        ``None`` outside a capture scope (the process-global list records no
        ownership).
        """
        bucket = _capture_scope.get()
        if bucket is None:
            return None
        with bucket.lock:
            return bucket.owners.copy()

    def drain(
        self,
    ) -> tuple[list[Any], list[int | None] | None, list[str | None] | None]:
        """Take every response with its owner and requested model, atomically.

        Snapshot and clear happen under ONE lock, so a call that lands while
        the evaluator folds spend is either in this drain or left for the next,
        never cleared unaccounted and never misaligned with its owner.  Owners
        and requested models are ``None`` outside a capture scope.
        """
        with self._lock:
            if hasattr(self._storage, "responses"):
                self._storage.responses.clear()
        bucket = _capture_scope.get()
        if bucket is not None:
            with bucket.lock:
                taken = (
                    bucket.responses.copy(),
                    bucket.owners.copy(),
                    bucket.requested.copy(),
                )
                bucket.responses.clear()
                bucket.owners.clear()
                bucket.requested.clear()
                bucket.by_key.clear()
            return taken
        with self._response_lock:
            responses = self._all_responses.copy()
            self._all_responses.clear()
        return responses, None, None

    def snapshot(
        self,
    ) -> tuple[list[Any], list[int | None] | None, list[str | None] | None]:
        """Atomic non-consuming view for ownership-aware early progress."""
        bucket = _capture_scope.get()
        if bucket is not None:
            with bucket.lock:
                return (
                    bucket.responses.copy(),
                    bucket.owners.copy(),
                    bucket.requested.copy(),
                )
        with self._response_lock:
            return self._all_responses.copy(), None, None

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
                bucket.owners.clear()
                bucket.requested.clear()
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
    _metadata_capture.set_last_response(response, requested_model=requested_model)
    return response


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


def get_captured_response_owners() -> list[int | None] | None:
    """Owners (example indexes) of ``get_all_captured_responses()``, in order."""
    return _metadata_capture.get_response_owners()


def drain_captured_responses() -> tuple[
    list[Any], list[int | None] | None, list[str | None] | None
]:
    """Atomically take ``(responses, owners, requested_models)`` and clear them."""
    return _metadata_capture.drain()


def snapshot_captured_responses() -> tuple[
    list[Any], list[int | None] | None, list[str | None] | None
]:
    """Read calls and provenance together without consuming final accounting."""
    return _metadata_capture.snapshot()


def mark_capture_incomplete(worker: Future[Any]) -> None:
    """A timed-out live worker leaves this trial's eventual spend unknown."""
    bucket = _capture_scope.get()
    if bucket is not None:
        with bucket.lock:
            bucket.timed_out_workers.append(worker)
            bucket.incomplete = True


def seal_captured_responses() -> bool:
    """Close appends before the final drain; return unresolved-spend state."""
    bucket = _capture_scope.get()
    if bucket is None:
        return False
    with bucket.lock:
        if not bucket.sealed:
            bucket.incomplete = any(
                not worker.done() for worker in bucket.timed_out_workers
            )
        bucket.sealed = True
        return bucket.incomplete


def finish_captured_responses(*, discard: bool = False) -> list[tuple[Any, str | None]]:
    """Take late captures; close only at an empty final handoff.

    Pricing a drained batch may let another worker capture, so callers fold
    batches until an empty take atomically closes appends. Scope exit discards
    any remaining batch when unwinding without a result.
    """
    bucket = _capture_scope.get()
    if bucket is None:
        return []
    with bucket.lock:
        late, bucket.late = bucket.late, []
        if discard or not late:
            bucket.finished = True
        return late


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


def _mock_chat_message(client: Any, provider: str, *, chunk: bool = False) -> Any:
    """Build the canned LangChain message mock mode returns for ``provider``."""
    from langchain_core.messages import AIMessage, AIMessageChunk

    from traigent.integrations.utils.mock_adapter import MockAdapter

    message_cls = AIMessageChunk if chunk else AIMessage
    if provider == "openai":
        model_name = getattr(client, "model_name", None) or getattr(
            client, "model", "mock-model"
        )
        data = MockAdapter.get_mock_response("openai", model=model_name)
        usage = data["usage"]
        return message_cls(
            content=data["choices"][0]["message"]["content"],
            response_metadata={
                "model_name": data["model"],
                "finish_reason": data["choices"][0]["finish_reason"],
                "response_time_ms": 0.0,
            },
            usage_metadata={
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        )
    if provider == "bedrock":
        model_name = (
            getattr(client, "model_id", None)
            or getattr(client, "model", None)
            or "mock-model"
        )
        data = MockAdapter.get_mock_response("bedrock", model=str(model_name))
    else:
        data = MockAdapter.get_mock_response(
            "anthropic", model=getattr(client, "model", "mock-model")
        )
    usage = data["usage"]
    return message_cls(
        content=data["content"][0]["text"],
        response_metadata={
            "model_name": data["model"],
            "stop_reason": data["stop_reason"],
            "response_time_ms": 0.0,
        },
        usage_metadata={
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        },
    )


_MOCKABLE_PROVIDERS = frozenset({"openai", "anthropic", "bedrock"})


def _mock_enabled(provider: str) -> bool:
    if provider not in _MOCKABLE_PROVIDERS:
        return False
    from traigent.integrations.utils.mock_adapter import MockAdapter

    return MockAdapter.is_mock_enabled(provider)


def _record_stream_capture(
    model: Any, provider: str, last: Any, start_time: float
) -> None:
    """Record the final stream chunk once, if any chunk was produced."""
    if last is None:
        return
    response_time_ms = (time.perf_counter() - start_time) * 1000
    if not hasattr(last, "response_metadata"):
        last.response_metadata = {}
    last.response_metadata["response_time_ms"] = response_time_ms
    # A stream chunk without usage_metadata is captured as-is: the downstream
    # metrics treat missing usage as unmeasured, not $0.
    capture_observed_response(
        last, provider=provider, requested_model=requested_model_of(model)
    )


def _create_stream_wrapper(original_meth: Any, provider: str = "unknown") -> Any:
    def stream_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _mock_enabled(provider):
            # Mock mode: never reach the provider (#2424).
            response = _mock_chat_message(self, provider, chunk=True)
            capture_langchain_response(response)
            yield response
            return
        start_time = time.perf_counter()
        last = None
        fallback = _stream_falls_back(self, async_api=False, kwargs=kwargs)
        owns = fallback is not True and id(self) not in _capture_owned.get()
        iterator = iter(original_meth(self, *args, **kwargs))
        try:
            while True:
                # Legacy only (``_should_stream`` unavailable): own the
                # capture while the wrapped stream runs, so a stream->invoke
                # fallback is captured once.
                token = _owned_by(self) if fallback is None else None
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                finally:
                    if token is not None:
                        _capture_owned.reset(token)
                last = chunk
                yield chunk
        except GeneratorExit:
            # close() after the first chunk: the provider call completed
            # (e.g. a stream->invoke fallback), so its usage still counts.
            if owns:
                _record_stream_capture(self, provider, last, start_time)
            raise
        if owns:
            _record_stream_capture(self, provider, last, start_time)

    return stream_wrapper


def _create_astream_wrapper(original_meth: Any, provider: str = "unknown") -> Any:
    async def astream_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _mock_enabled(provider):
            response = _mock_chat_message(self, provider, chunk=True)
            capture_langchain_response(response)
            yield response
            return
        start_time = time.perf_counter()
        last = None
        fallback = _stream_falls_back(self, async_api=True, kwargs=kwargs)
        owns = fallback is not True and id(self) not in _capture_owned.get()
        iterator = original_meth(self, *args, **kwargs).__aiter__()
        try:
            while True:
                # Legacy only: see stream_wrapper.
                token = _owned_by(self) if fallback is None else None
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                finally:
                    if token is not None:
                        _capture_owned.reset(token)
                last = chunk
                yield chunk
        except GeneratorExit:
            # aclose() after the first chunk: see stream_wrapper.
            if owns:
                _record_stream_capture(self, provider, last, start_time)
            raise
        if owns:
            _record_stream_capture(self, provider, last, start_time)

    return astream_wrapper


def _create_ainvoke_wrapper(original_meth: Any, provider: str = "unknown") -> Any:
    """Wrap ``ainvoke`` so mock mode covers it and its cost is captured (#2424, #2445).

    ``abatch`` is covered through this wrapper: LangChain's default
    ``abatch`` dispatches every item to ``self.ainvoke``.
    """

    async def ainvoke_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _mock_enabled(provider):
            response = _mock_chat_message(self, provider)
            capture_langchain_response(response)
            return response
        start_time = time.perf_counter()
        with _capture_ownership(self) as owns, instrumented_provider_call():
            response = await original_meth(self, *args, **kwargs)
        response_time_ms = (time.perf_counter() - start_time) * 1000
        if not hasattr(response, "response_metadata"):
            response.response_metadata = {}
        response.response_metadata["response_time_ms"] = response_time_ms
        if owns:
            capture_observed_response(
                response, provider=provider, requested_model=requested_model_of(self)
            )
        return response

    return ainvoke_wrapper


def _patch_ainvoke(model_cls: Any, class_name: str, provider: str) -> bool:
    """Patch ``model_cls.ainvoke`` once; returns True if newly patched."""
    flag = "_traigent_patched_ainvoke"
    # Check the class's own namespace: a subclass must not inherit "patched".
    if flag in vars(model_cls) or not hasattr(model_cls, "ainvoke"):
        return False
    model_cls.ainvoke = _create_ainvoke_wrapper(model_cls.ainvoke, provider)
    setattr(model_cls, flag, True)
    logger.info("Patched %s.ainvoke for metadata capture", class_name)
    return True


def _patch_langchain_bedrock_model(model_cls: Any, class_name: str) -> bool:
    """Patch one langchain_aws Bedrock chat class for response capture."""
    patched_any = False

    if getattr(model_cls, "_traigent_patched_invoke", False) is False:
        original_invoke = model_cls.invoke

        def invoke_with_capture_bedrock(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Wrapped invoke method that captures Bedrock usage metadata."""
            from traigent.integrations.utils.mock_adapter import MockAdapter

            if MockAdapter.is_mock_enabled("bedrock"):
                from langchain_core.messages import AIMessage

                model_name = (
                    getattr(self, "model_id", None)
                    or getattr(self, "model", None)
                    or "mock-model"
                )
                mock_data = MockAdapter.get_mock_response(
                    "bedrock", model=str(model_name)
                )
                usage = mock_data["usage"]
                response = AIMessage(
                    content=mock_data["content"][0]["text"],
                    response_metadata={
                        "model_name": mock_data["model"],
                        "stop_reason": mock_data["stop_reason"],
                        "response_time_ms": 0.0,
                    },
                    usage_metadata={
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "total_tokens": usage["total_tokens"],
                    },
                )
                capture_langchain_response(response)
                return response

            start_time = time.perf_counter()
            with _capture_ownership(self) as owns:
                response = original_invoke(self, *args, **kwargs)
            response_time_ms = (time.perf_counter() - start_time) * 1000

            if not hasattr(response, "response_metadata"):
                response.response_metadata = {}
            response.response_metadata["response_time_ms"] = response_time_ms

            if owns:
                capture_observed_response(
                    response,
                    provider="bedrock",
                    requested_model=requested_model_of(self),
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

    if _patch_ainvoke(model_cls, class_name, "bedrock"):
        patched_any = True

    for meth_name, flag in [
        ("stream", "_traigent_patched_stream"),
        ("astream", "_traigent_patched_astream"),
    ]:
        if hasattr(model_cls, meth_name) and not getattr(model_cls, flag, False):
            original_meth = getattr(model_cls, meth_name)
            if meth_name == "stream":
                setattr(
                    model_cls,
                    meth_name,
                    _create_stream_wrapper(original_meth, provider="bedrock"),
                )
            else:
                setattr(
                    model_cls,
                    meth_name,
                    _create_astream_wrapper(original_meth, provider="bedrock"),
                )
            setattr(model_cls, flag, True)
            logger.info("✅ Patched %s.%s for metadata capture", class_name, meth_name)
            patched_any = True

    return patched_any


INSTALL_INSTALLED = "installed"
INSTALL_ALREADY = "already-installed"
INSTALL_NOT_APPLICABLE = "not-applicable"
INSTALL_FAILED = "failed"


def classify_import_failure(top_level_package: str) -> str:
    """Outcome for an ``ImportError`` raised while installing one provider.

    ``not-applicable`` only when the optional top-level package is provably
    absent (``find_spec`` returns None). A package that is present but fails
    to import (a broken transitive dependency, an incompatible version) is a
    ``failed`` install: the provider would otherwise reach the real API.
    """
    try:
        absent = importlib.util.find_spec(top_level_package) is None
    except (ImportError, ValueError):
        absent = False
    return INSTALL_NOT_APPLICABLE if absent else INSTALL_FAILED


def patch_langchain_for_metadata_capture(
    outcomes: dict[str, str] | None = None,
) -> bool:
    """Monkey-patch LangChain to automatically capture response metadata.

    This patches ChatAnthropic, ChatOpenAI and the langchain_aws Bedrock
    models to capture responses. Patch failures are logged, not raised, so
    callers that must know (mock-mode activation) pass ``outcomes``: it is
    filled with one of ``installed`` / ``already-installed`` /
    ``not-applicable`` (package absent) / ``failed`` per provider key
    (``anthropic``, ``openai``, ``bedrock``).
    """
    out: dict[str, str] = outcomes if outcomes is not None else {}
    patched_any = False
    before = False

    # Patch ChatAnthropic
    before, patched_any = patched_any, False
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
                    from langchain_core.messages import AIMessage

                    model_name = getattr(self, "model", "mock-model")
                    mock_data = MockAdapter.get_mock_response(
                        "anthropic", model=model_name
                    )
                    content = mock_data["content"][0]["text"]
                    response = AIMessage(
                        content=content,
                        response_metadata={
                            "model_name": mock_data["model"],
                            "stop_reason": mock_data["stop_reason"],
                            "response_time_ms": 0.0,
                        },
                        usage_metadata={
                            "input_tokens": mock_data["usage"]["input_tokens"],
                            "output_tokens": mock_data["usage"]["output_tokens"],
                            "total_tokens": mock_data["usage"]["input_tokens"]
                            + mock_data["usage"]["output_tokens"],
                        },
                    )
                    capture_langchain_response(response)
                    return response

                start_time = time.perf_counter()
                with _capture_ownership(self) as owns:
                    response = original_invoke(self, *args, **kwargs)
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Inject timing into response metadata
                if not hasattr(response, "response_metadata"):
                    response.response_metadata = {}
                response.response_metadata["response_time_ms"] = response_time_ms

                if owns:
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

        if _patch_ainvoke(ChatAnthropic, "ChatAnthropic", "anthropic"):
            patched_any = True

        # Patch stream methods if available
        for meth_name, flag in [
            ("stream", "_traigent_patched_stream"),
            ("astream", "_traigent_patched_astream"),
        ]:
            if hasattr(ChatAnthropic, meth_name) and not getattr(
                ChatAnthropic, flag, False
            ):
                original_meth = getattr(ChatAnthropic, meth_name)

                if meth_name == "stream":
                    setattr(
                        ChatAnthropic,
                        meth_name,
                        _create_stream_wrapper(original_meth, provider="anthropic"),
                    )
                else:
                    setattr(
                        ChatAnthropic,
                        meth_name,
                        _create_astream_wrapper(original_meth, provider="anthropic"),
                    )
                setattr(ChatAnthropic, flag, True)
                logger.info(
                    f"✅ Patched ChatAnthropic.{meth_name} for metadata capture"
                )

        out["anthropic"] = INSTALL_INSTALLED if patched_any else INSTALL_ALREADY
    except ImportError:
        out["anthropic"] = classify_import_failure("langchain_anthropic")
        if out["anthropic"] == INSTALL_FAILED:
            logger.error(
                "Failed to patch langchain_anthropic: import error in an installed package"
            )
        else:
            logger.debug("ChatAnthropic not available, skipping")
    except Exception as e:
        out["anthropic"] = INSTALL_FAILED
        logger.error(f"Failed to patch ChatAnthropic: {e}")
    patched_any = patched_any or before

    # Patch ChatOpenAI
    before, patched_any = patched_any, False
    try:
        from langchain_openai import ChatOpenAI

        if getattr(ChatOpenAI, "_traigent_patched_invoke", False) is False:
            original_invoke_openai = ChatOpenAI.invoke

            def invoke_with_capture_openai(self: Any, *args: Any, **kwargs: Any) -> Any:
                """Wrapped invoke method that captures response with timing."""
                # Check if mock mode is enabled — return mock response without API call
                from traigent.integrations.utils.mock_adapter import MockAdapter

                if MockAdapter.is_mock_enabled("openai"):
                    from langchain_core.messages import AIMessage

                    model_name = getattr(self, "model_name", None) or getattr(
                        self, "model", "mock-model"
                    )
                    mock_data = MockAdapter.get_mock_response(
                        "openai", model=model_name
                    )
                    content = mock_data["choices"][0]["message"]["content"]
                    response = AIMessage(
                        content=content,
                        response_metadata={
                            "model_name": mock_data["model"],
                            "finish_reason": mock_data["choices"][0]["finish_reason"],
                            "response_time_ms": 0.0,
                        },
                        usage_metadata={
                            "input_tokens": mock_data["usage"]["prompt_tokens"],
                            "output_tokens": mock_data["usage"]["completion_tokens"],
                            "total_tokens": mock_data["usage"]["total_tokens"],
                        },
                    )
                    capture_langchain_response(response)
                    return response

                start_time = time.perf_counter()
                with _capture_ownership(self) as owns, instrumented_provider_call():
                    response = original_invoke_openai(self, *args, **kwargs)
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Inject timing into response metadata
                if not hasattr(response, "response_metadata"):
                    response.response_metadata = {}
                response.response_metadata["response_time_ms"] = response_time_ms

                if owns:
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

        if _patch_ainvoke(ChatOpenAI, "ChatOpenAI", "openai"):
            patched_any = True

        for meth_name, flag in [
            ("stream", "_traigent_patched_stream"),
            ("astream", "_traigent_patched_astream"),
        ]:
            if hasattr(ChatOpenAI, meth_name) and not getattr(ChatOpenAI, flag, False):
                original_meth = getattr(ChatOpenAI, meth_name)
                if meth_name == "stream":
                    setattr(
                        ChatOpenAI,
                        meth_name,
                        _create_stream_wrapper(original_meth, provider="openai"),
                    )
                else:
                    setattr(
                        ChatOpenAI,
                        meth_name,
                        _create_astream_wrapper(original_meth, provider="openai"),
                    )
                setattr(ChatOpenAI, flag, True)
                logger.info(f"✅ Patched ChatOpenAI.{meth_name} for metadata capture")

        out["openai"] = INSTALL_INSTALLED if patched_any else INSTALL_ALREADY
    except ImportError:
        out["openai"] = classify_import_failure("langchain_openai")
        if out["openai"] == INSTALL_FAILED:
            logger.error(
                "Failed to patch langchain_openai: import error in an installed package"
            )
        else:
            logger.debug("ChatOpenAI not available, skipping")
    except Exception as e:
        out["openai"] = INSTALL_FAILED
        logger.error(f"Failed to patch ChatOpenAI: {e}")
    patched_any = patched_any or before

    # Patch langchain_aws Bedrock chat models
    before, patched_any = patched_any, False
    try:
        import langchain_aws

        for class_name in ("ChatBedrock", "ChatBedrockConverse"):
            model_cls = getattr(langchain_aws, class_name, None)
            if model_cls is None:
                logger.debug("langchain_aws.%s not available, skipping", class_name)
                continue
            if _patch_langchain_bedrock_model(model_cls, f"langchain_aws.{class_name}"):
                patched_any = True

        out["bedrock"] = INSTALL_INSTALLED if patched_any else INSTALL_ALREADY
    except ImportError:
        out["bedrock"] = classify_import_failure("langchain_aws")
        if out["bedrock"] == INSTALL_FAILED:
            logger.error(
                "Failed to patch langchain_aws: import error in an installed package"
            )
        else:
            logger.debug("langchain_aws not available, skipping Bedrock")
    except Exception as e:
        out["bedrock"] = INSTALL_FAILED
        logger.error(f"Failed to patch langchain_aws Bedrock models: {e}")
    patched_any = patched_any or before

    if not patched_any:
        logger.debug("No LangChain models could be patched for metadata capture")
        return False

    return True
