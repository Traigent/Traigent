from __future__ import annotations

import asyncio

from traigent.optimizers.remote import MockRemoteSuggestionClient, RemoteOptimizer


def test_remote_optimizer_fallback_sync():
    opt = RemoteOptimizer({"x": [1, 2]}, ["accuracy"], remote_enabled=False)
    cfg = opt.suggest_next_trial([])
    assert cfg in [{"x": 1}, {"x": 2}]


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
