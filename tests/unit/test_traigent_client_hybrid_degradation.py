"""Regression tests for hybrid-mode graceful degradation in TraigentClient.

Issue #1265: when the Traigent backend is unreachable, hybrid mode must
gracefully fall back to local results instead of crashing. Previously
``_optimize_hybrid`` called ``finalize_hybrid_session`` unguarded, so a
``CloudServiceError`` raised by an unreachable backend at finalize propagated
out of the whole ``optimize()`` call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from traigent.cloud.client import CloudServiceError
from traigent.traigent_client import TraigentClient


def _build_hybrid_client() -> TraigentClient:
    client = object.__new__(TraigentClient)
    backend = Mock()
    backend.__aenter__ = AsyncMock(return_value=backend)
    backend.__aexit__ = AsyncMock(return_value=None)
    backend.create_hybrid_session = AsyncMock(
        return_value=("sess-1", "token", "https://endpoint")
    )
    client.backend_client = backend
    client.agent_builder = Mock()
    return client


def _patch_optimizer_loop(mp, *, has_next: bool = False) -> None:
    """Patch the direct optimizer client + adapter so the trial loop exits
    immediately (no real trials) and control reaches finalize."""
    fake_optimizer = AsyncMock()
    fake_optimizer.__aenter__.return_value = fake_optimizer
    fake_optimizer.__aexit__.return_value = None
    fake_optimizer.get_next_configuration = AsyncMock(
        return_value={"has_next": has_next}
    )
    fake_optimizer.submit_metrics = AsyncMock()

    direct_client_factory = Mock(return_value=fake_optimizer)
    # raising=False: the optional cloud-optimizer extra may be absent in the
    # unit-test environment (then `_CLOUD_AVAILABLE` is False and these names
    # aren't bound on the module); we still inject the mocks the patched
    # `_optimize_hybrid` resolves from module globals.
    mp.setattr(
        "traigent.traigent_client.OptimizerDirectClient",
        direct_client_factory,
        raising=False,
    )
    mp.setattr(
        "traigent.traigent_client.LocalExecutionAdapter",
        lambda *a, **k: Mock(),
        raising=False,
    )


@pytest.mark.asyncio
async def test_hybrid_finalize_backend_unreachable_degrades_to_local(caplog):
    """A CloudServiceError at finalize must NOT crash the run; it returns a
    local result marked source='local'."""
    client = _build_hybrid_client()
    client.backend_client.finalize_hybrid_session = AsyncMock(
        side_effect=CloudServiceError("backend unreachable")
    )

    with pytest.MonkeyPatch.context() as mp:
        _patch_optimizer_loop(mp, has_next=False)
        result = await client._optimize_hybrid(
            function=lambda **_kwargs: "ok",
            dataset={"examples": []},
            configuration_space={"model": ["gpt-4o-mini"]},
            objectives=["accuracy"],
            max_trials=2,
            optimization_config={},
            config_defaults={},
        )

    assert result["source"] == "local"
    assert result["backend_finalized"] is False
    assert result["execution_mode"] == "hybrid"


@pytest.mark.asyncio
async def test_hybrid_finalize_success_marks_source_backend():
    """When the backend finalizes normally, the result is source='backend'."""
    client = _build_hybrid_client()
    client.backend_client.finalize_hybrid_session = AsyncMock(
        return_value={"best_configuration": {"model": "gpt-4o-mini"}}
    )

    with pytest.MonkeyPatch.context() as mp:
        _patch_optimizer_loop(mp, has_next=False)
        result = await client._optimize_hybrid(
            function=lambda **_kwargs: "ok",
            dataset={"examples": []},
            configuration_space={"model": ["gpt-4o-mini"]},
            objectives=["accuracy"],
            max_trials=2,
            optimization_config={},
            config_defaults={},
        )

    assert result["source"] == "backend"
    assert result["execution_mode"] == "hybrid"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
