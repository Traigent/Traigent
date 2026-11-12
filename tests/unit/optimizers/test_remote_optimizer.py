from __future__ import annotations

import asyncio

import pytest

from traigent.optimizers.registry import get_optimizer, list_optimizers
from traigent.optimizers.remote import MockRemoteSuggestionClient, RemoteOptimizer
from traigent.utils.exceptions import OptimizationError


def test_remote_optimizer_fail_loud_without_client_and_without_remote_enabled():
    """Issue #872 regression: constructing without a remote_client used to
    silently fall back to local random search, masquerading as 'remote'.
    The construction now MUST fail loud with a migration message instead.
    """
    with pytest.raises(OptimizationError) as exc_info:
        RemoteOptimizer({"x": [1, 2]}, ["accuracy"], remote_enabled=False)
    msg = str(exc_info.value)
    assert "remote_client" in msg
    assert "remote_enabled=True" in msg
    assert "algorithm='random'" in msg
    assert "execution_mode='hybrid'" in msg


def test_remote_optimizer_fail_loud_with_remote_enabled_but_no_client():
    """remote_enabled=True without a client is still invalid — same gate."""
    with pytest.raises(OptimizationError) as exc_info:
        RemoteOptimizer({"x": [1, 2]}, ["accuracy"], remote_enabled=True)
    assert "remote_client" in str(exc_info.value)


def test_remote_optimizer_fail_loud_with_client_but_remote_not_enabled():
    """A client alone is not enough; the opt-in flag must be explicit too."""
    client = MockRemoteSuggestionClient()

    with pytest.raises(OptimizationError) as exc_info:
        RemoteOptimizer({"x": [1, 2]}, ["accuracy"], remote_client=client)

    msg = str(exc_info.value)
    assert "remote_client" in msg
    assert "remote_enabled=True" in msg


def test_remote_still_registered_so_users_get_a_clear_error_not_keyerror():
    """We keep "remote" in the registry so users who pass algorithm='remote'
    receive the OptimizationError migration message via construction, not
    a confusing KeyError from a missing registry entry.
    """
    assert "remote" in list_optimizers()

    # The registry path (algorithm='remote' in @traigent.optimize) instantiates
    # via get_optimizer(); that path must surface the same migration message.
    with pytest.raises(OptimizationError) as exc_info:
        get_optimizer("remote", {"x": [1, 2]}, ["accuracy"])
    # get_optimizer re-wraps; the original migration text is preserved in the chain.
    assert "remote_client" in str(exc_info.value) or "remote_client" in str(
        exc_info.value.__cause__ or ""
    )


def test_remote_optimizer_uses_remote_single_suggestion():
    client = MockRemoteSuggestionClient()
    opt = RemoteOptimizer(
        {"x": [1, 2]}, ["accuracy"], remote_enabled=True, remote_client=client
    )

    async def run():
        # force single-suggestion path by calling suggest_next_trial_async directly
        cfg = await opt.suggest_next_trial_async(
            [], remote_context={"privacy_enabled": True}
        )
        return cfg, client

    cfg, client_used = asyncio.run(run())
    assert cfg in [{"x": 1}, {"x": 2}]
    assert (
        client_used.last_context
        and client_used.last_context.get("privacy_enabled") is True
    )


def test_remote_optimizer_sync_suggestion_fails_loud_instead_of_random_fallback():
    client = MockRemoteSuggestionClient()
    opt = RemoteOptimizer(
        {"x": [1, 2]}, ["accuracy"], remote_enabled=True, remote_client=client
    )

    with pytest.raises(OptimizationError) as exc_info:
        opt.suggest_next_trial([])

    msg = str(exc_info.value)
    assert "does not support synchronous suggestions" in msg
    assert "algorithm='random'" in msg
    assert client.calls["single"] == 0


def test_remote_optimizer_uses_remote_batch_suggestions():
    client = MockRemoteSuggestionClient()
    opt = RemoteOptimizer(
        {"x": [1, 2, 3]}, ["accuracy"], remote_enabled=True, remote_client=client
    )

    async def run():
        cands = await opt.generate_candidates_async(
            3, remote_context={"privacy_enabled": False}
        )
        return cands, client

    cands, client_used = asyncio.run(run())
    assert isinstance(cands, list) and len(cands) == 3
    assert all("x" in c for c in cands)
    assert (
        client_used.last_context
        and client_used.last_context.get("privacy_enabled") is False
    )


class _RecordingClient:
    """Stub remote client recording the history it receives (#3153)."""

    def __init__(self, batch=None, single=None):
        self.batch = batch
        self.single = single
        self.histories: list = []

    async def suggest_batch(self, config_space, n, history=None, ctx=None):
        self.histories.append(history)
        return self.batch

    async def suggest_next(self, config_space, history=None, ctx=None):
        return self.single


def _remote(client, space=None):
    return RemoteOptimizer(
        space or {"model": ["a", "b"], "t": [0.1, 0.2]},
        ["accuracy"],
        remote_enabled=True,
        remote_client=client,
    )


def test_remote_batch_receives_caller_history_3153():
    good = [{"model": "a", "t": 0.1}]
    client = _RecordingClient(batch=good)
    trial = {"trial_id": "t1", "config": {"model": "a"}, "metrics": {"score": 0.5}}
    out = asyncio.run(_remote(client).generate_candidates_async(1, history=[trial]))
    assert out == good
    # No explicit privacy_enabled=False: indices and statuses only.
    assert client.histories == [[{"index": 0, "status": None}]]


def test_remote_batch_history_carries_config_only_when_privacy_is_off_3153():
    client = _RecordingClient(batch=[{"model": "a", "t": 0.1}])
    trial = {
        "trial_id": "t1",
        "config": {"model": "a"},
        "metrics": {"score": 0.5, "note": "free text"},
        "status": None,
    }
    asyncio.run(
        _remote(client).generate_candidates_async(
            1, remote_context={"privacy_enabled": False}, history=[trial]
        )
    )
    (sent,) = client.histories
    assert sent == [
        {
            "index": 0,
            "status": None,
            "trial_id": "t1",
            "config": {"model": "a"},
            "metrics": {"score": 0.5},
        }
    ]


def test_orchestrator_batch_path_sends_completed_trials_as_closed_history_3153():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from traigent.api.types import TrialResult, TrialStatus
    from traigent.core.orchestrator import OptimizationOrchestrator

    client = _RecordingClient(batch=[{"model": "a", "t": 0.1}])
    trials = [
        TrialResult(
            trial_id=f"t{i}",
            config={"model": "a", "t": 0.1},
            metrics={"accuracy": 0.1 * i, "note": "free text"},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime.now(UTC),
            metadata={"example_output": "SECRET CONTENT"},
        )
        for i in range(3)
    ]
    fake_self = SimpleNamespace(
        traigent_config=SimpleNamespace(privacy_enabled=False),
        optimizer=_remote(client),
        _trials=trials,
    )
    out = asyncio.run(
        OptimizationOrchestrator._try_hybrid_batch_generation(fake_self, [1, 2], 1)
    )
    assert out == [{"model": "a", "t": 0.1}]
    (sent,) = client.histories
    assert [e["trial_id"] for e in sent] == ["t0", "t1", "t2"]
    assert set(sent[0]) == {"index", "trial_id", "config", "metrics", "status"}
    assert sent[0]["status"] == "completed"
    assert "note" not in sent[0]["metrics"]
    assert "SECRET" not in repr(sent)


@pytest.mark.parametrize(
    "bad", [{"model": "x"}, None, [], [{"model": "zzz", "t": 0.1}], ["nope"]]
)
def test_remote_batch_malformed_fails_closed_3153(bad):
    client = _RecordingClient(batch=bad)
    with pytest.raises(OptimizationError, match="Remote batch rejected"):
        asyncio.run(_remote(client).generate_candidates_async(2))


@pytest.mark.parametrize(
    "bad", ["not-a-dict", {"model": "q", "t": 0.1}, {"model": "a"}, None]
)
def test_remote_single_suggestion_fails_closed_3153(bad):
    client = _RecordingClient(single=bad)
    with pytest.raises(OptimizationError, match="Remote suggestion rejected"):
        asyncio.run(_remote(client).suggest_next_trial_async([]))


def test_remote_client_exception_still_falls_back_with_warning_3153(caplog):
    class Boom(_RecordingClient):
        async def suggest_batch(self, *a, **k):
            raise RuntimeError("down")

    with caplog.at_level("WARNING", logger="traigent.optimizers.remote"):
        out = asyncio.run(_remote(Boom()).generate_candidates_async(2))
    assert out
    assert any("falling back" in r.getMessage() for r in caplog.records)


def test_remote_suggestions_validated_against_range_specs_3153():
    from traigent.api.parameter_ranges import Choices, IntRange, LogRange, Range

    space = {
        "t": Range(0.0, 1.0),
        "n": IntRange(1, 9, step=2),
        "lr": LogRange(1e-4, 1e-1),
        "m": Choices(["a", "b"]),
        "tup": (0, 10),
        "d": {"type": "float", "low": 0.0, "high": 2.0},
    }
    good = {"t": 0.5, "n": 3, "lr": 0.01, "m": "a", "tup": 4, "d": 1.5}
    assert asyncio.run(
        _remote(_RecordingClient(batch=[good]), space).generate_candidates_async(1)
    ) == [good]
    for patch in (
        {"t": 1.5},
        {"t": float("nan")},
        {"t": True},
        {"n": 4},
        {"n": 3.5},
        {"n": 11},
        {"lr": 0.5},
        {"m": "c"},
        {"tup": 11},
        {"d": -1.0},
    ):
        with pytest.raises(OptimizationError):
            asyncio.run(
                _remote(
                    _RecordingClient(batch=[{**good, **patch}]), space
                ).generate_candidates_async(1)
            )


def test_remote_unsupported_spec_fails_closed_3153():
    client = _RecordingClient(batch=[{"x": 1}])
    with pytest.raises(OptimizationError, match="not supported"):
        asyncio.run(_remote(client, {"x": object()}).generate_candidates_async(1))
