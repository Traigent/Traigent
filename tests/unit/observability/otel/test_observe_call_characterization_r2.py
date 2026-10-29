"""Characterize decorator completion and stream control without network I/O."""

import asyncio
from types import SimpleNamespace

import pytest
from opentelemetry.trace import StatusCode

from traigent.observability.otel import api
from traigent.observability.otel import contract as C


@pytest.fixture
def spans(monkeypatch, memory_provider):
    provider, exporter = memory_provider
    monkeypatch.setattr(
        api,
        "_handle",
        SimpleNamespace(content_mode="record", tracer=provider.get_tracer("test")),
    )
    yield exporter
    monkeypatch.setattr(api, "_handle", None)


def assert_span(exporter, error=None, output=None):
    (span,) = exporter.get_finished_spans()
    assert span.attributes[C.ATTR_OBSERVATION_TYPE] == "span"
    if error:
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == error
        assert span.events[0].attributes["exception.type"] == error
    else:
        assert span.status.status_code == StatusCode.UNSET
    if output is not None:
        assert span.attributes[C.ATTR_OUTPUT] == output


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
def test_call_result_and_exception(spans, fail, is_async):
    def action():
        if fail:
            raise ValueError("broken")
        return 42

    if is_async:

        @api.observe("call")
        async def invoke():
            return action()

        def call():
            return asyncio.run(invoke())
    else:
        call = api.observe("call")(action)
    if fail:
        with pytest.raises(ValueError, match="broken"):
            call()
        assert_span(spans, error="ValueError")
    else:
        assert call() == 42
        assert_span(spans, output="42")


@pytest.mark.parametrize(
    "operation", ["exhaust", "close", "throw", "send", "unhandled"]
)
def test_generator_controls(spans, operation):
    events = []

    @api.observe("stream")
    def stream():
        try:
            try:
                value = yield "first"
                yield value
            except ValueError:
                yield "handled"
            return "done"
        finally:
            events.append("closed")

    gen = stream()
    assert next(gen) == "first"
    if operation == "close":
        gen.close()
    elif operation == "unhandled":
        with pytest.raises(KeyError):
            gen.throw(KeyError("bad"))
    else:
        if operation == "throw":
            assert gen.throw(ValueError()) == "handled"
        else:
            assert gen.send(7 if operation == "send" else None) == (
                7 if operation == "send" else None
            )
        with pytest.raises(StopIteration) as stop:
            next(gen)
        assert stop.value.value == "done"
    assert events == ["closed"]
    assert_span(
        spans,
        error="KeyError" if operation == "unhandled" else None,
        output='"done"' if operation in {"exhaust", "throw", "send"} else None,
    )


@pytest.mark.parametrize(
    "operation", ["exhaust", "close", "throw", "send", "unhandled"]
)
def test_async_generator_controls(spans, operation):
    events = []

    @api.observe("stream")
    async def stream():
        try:
            try:
                value = yield "first"
                yield value
            except ValueError:
                yield "handled"
        finally:
            events.append("closed")

    async def consume():
        gen = stream()
        assert await gen.__anext__() == "first"
        if operation == "close":
            await gen.aclose()
        elif operation == "unhandled":
            with pytest.raises(KeyError):
                await gen.athrow(KeyError("bad"))
        else:
            if operation == "throw":
                assert await gen.athrow(ValueError()) == "handled"
            else:
                assert await gen.asend(7 if operation == "send" else None) == (
                    7 if operation == "send" else None
                )
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()

    asyncio.run(consume())
    assert events == ["closed"]
    assert_span(spans, error="KeyError" if operation == "unhandled" else None)


def test_disabled_tracing_and_start_failure(monkeypatch):
    monkeypatch.setattr(api, "_handle", None)
    assert api.observe("disabled")(lambda: 3)() == 3

    class BrokenTracer:
        def start_span(self, *args, **kwargs):
            raise RuntimeError("start failed")

    monkeypatch.setattr(api, "_tracer", lambda: BrokenTracer())
    with pytest.raises(RuntimeError, match="start failed"):
        api.observe("broken")(lambda: 3)()
