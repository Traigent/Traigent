"""HybridAPIOptions numeric fields reject 0, negative, NaN, bool and str."""

from __future__ import annotations

import pytest

import traigent
from traigent.api.decorators import HybridAPIOptions

NAN = float("nan")
INF = float("inf")

# --- hybrid options ------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, NAN, INF, True, "5"])
def test_hybrid_timeout_rejected(bad):
    with pytest.raises(Exception, match="timeout"):
        HybridAPIOptions(timeout=bad)


def test_hybrid_timeout_valid():
    assert HybridAPIOptions(timeout=5).timeout == 5.0
    assert HybridAPIOptions().timeout is None


@pytest.mark.parametrize("field", ["batch_size", "batch_parallelism"])
def test_hybrid_int_fields_reject_bool(field):
    with pytest.raises(Exception, match=field):
        HybridAPIOptions(**{field: True})


def test_hybrid_heartbeat_rejects_bool_and_nan():
    for bad in (True, NAN):
        with pytest.raises(Exception, match="heartbeat_interval"):
            HybridAPIOptions(heartbeat_interval=bad)


def test_hybrid_flat_kwarg_timeout_rejected_at_decoration():
    with pytest.raises(Exception, match="timeout"):

        @traigent.optimize(
            execution_mode="hybrid_api",
            hybrid_api_endpoint="http://localhost:1",
            hybrid_api_timeout=0,
            eval_dataset=[{"input": {"q": "a"}, "output": "a"}],
            objectives=["accuracy"],
        )
        def fn(q):
            return q
