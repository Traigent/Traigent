"""Response-timing side map (#1812): no response mutation, GC-reentrancy safe."""

import asyncio
import gc
import threading
import types
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict

from traigent.evaluators.metrics_tracker import OpenAIResponseHandler
from traigent.utils import response_timing as rt
from traigent.utils.langchain_interceptor import clear_captured_responses


class Resp(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "gpt-4o-mini"


def _drain():
    with rt._LOCK:
        rt._drain_dead_locked()


def test_pydantic_not_mutated_and_time_recoverable():
    resp = Resp()
    before = dict(resp.__pydantic_extra__ or {})
    rt.record_response_time_ms(resp, 12.5)
    assert (resp.__pydantic_extra__ or {}) == before
    assert not hasattr(resp, "response_time_ms")
    assert "response_time_ms" not in resp.model_dump()
    assert rt.recorded_response_time_ms(resp) == 12.5
    assert OpenAIResponseHandler().extract_response_time(resp) == 12.5


def test_plain_object_still_gets_attribute():
    obj = types.SimpleNamespace()
    rt.record_response_time_ms(obj, 3.0)
    assert obj.response_time_ms == 3.0


def test_weakref_cleanup_removes_entry():
    resp = Resp()
    key = id(resp)
    rt.record_response_time_ms(resp, 1.0)
    assert key in rt._TIMES
    del resp
    gc.collect()
    _drain()
    assert key not in rt._TIMES


def test_stale_id_is_rejected():
    resp = Resp()
    rt.record_response_time_ms(resp, 1.0)
    other = Resp()
    # Simulate id reuse: other's id maps to resp's (live, different) entry.
    with rt._LOCK:
        rt._TIMES[id(other)] = rt._TIMES[id(resp)]
    assert rt.recorded_response_time_ms(other) is None
    with rt._LOCK:
        del rt._TIMES[id(other)]


def test_object_without_weakref_support_falls_back_safely():
    class Slotted:
        __slots__ = ()
        model_dump = None
        __pydantic_serializer__ = None

    obj = Slotted()
    rt.record_response_time_ms(obj, 1.0)  # must not raise
    assert rt.recorded_response_time_ms(obj) is None


def test_gc_callback_while_lock_held_does_not_deadlock():
    resp = Resp()
    rt.record_response_time_ms(resp, 1.0)
    del resp
    done = threading.Event()

    def work():
        with rt._LOCK:  # a GC pass here fires the weakref callback
            gc.collect()
        rt.recorded_response_time_ms(Resp())  # drains the queue
        done.set()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    assert done.wait(10), "weakref callback deadlocked on _LOCK"


def test_concurrent_record_and_read():
    errors = []

    def worker(n):
        try:
            for i in range(300):
                r = Resp()
                rt.record_response_time_ms(r, float(n * 1000 + i))
                assert rt.recorded_response_time_ms(r) == float(n * 1000 + i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    gc.collect()
    _drain()
    assert not errors


@pytest.fixture()
def real_litellm_module():
    litellm = pytest.importorskip("litellm")
    mod = types.ModuleType("litellm")

    def completion(*a, **kw):
        return litellm.ModelResponse(model=kw.get("model", "gpt-4o"))

    async def acompletion(*a, **kw):
        return litellm.ModelResponse(model=kw.get("model", "gpt-4o"))

    mod.completion = completion
    mod.acompletion = acompletion
    mod.pre_call_rules = []
    mod.utils = types.SimpleNamespace(function_setup=lambda f, *a, **kw: (None, kw))
    mod._traigent_patched_completion = False
    mod._traigent_patched_acompletion = False
    clear_captured_responses()
    with (
        patch.dict("sys.modules", {"litellm": mod}),
        patch(
            "traigent.integrations.utils.mock_adapter.MockAdapter.is_mock_enabled",
            return_value=False,
        ),
    ):
        yield mod
    clear_captured_responses()


def _assert_not_mutated_and_timed(resp):
    assert "response_time_ms" not in (resp.__pydantic_extra__ or {})
    assert "response_time_ms" not in resp.__dict__
    assert "response_time_ms" not in resp.model_dump()
    assert OpenAIResponseHandler().extract_response_time(resp) > 0


def test_sync_interceptor_with_real_model_response(real_litellm_module):
    from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

    patch_litellm_for_metadata_capture()
    resp = real_litellm_module.completion(model="gpt-4o", messages=[])
    _assert_not_mutated_and_timed(resp)


def test_async_interceptor_with_real_model_response(real_litellm_module):
    from traigent.utils.litellm_interceptor import patch_litellm_for_metadata_capture

    patch_litellm_for_metadata_capture()
    resp = asyncio.run(real_litellm_module.acompletion(model="gpt-4o", messages=[]))
    _assert_not_mutated_and_timed(resp)
