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
