"""best_config_source="agent_head": the exact version the agent head pins, no fallback."""

from __future__ import annotations

from typing import Any

import pytest

import traigent.runtime.hosted as hosted
from traigent.core.best_config_runtime import (
    BestConfigSource,
    CloudPublishUnavailable,
    CloudPublishUnavailableReason,
)
from traigent.core.optimized_function import OptimizedFunction

from .conftest import Reactive


def answer(text: str, **kwargs: Any) -> str:
    return f"{text}:{kwargs.get('temperature')}"


@pytest.fixture
def wired(reactive: Reactive, monkeypatch: pytest.MonkeyPatch) -> Reactive:
    monkeypatch.setenv("TRAIGENT_AGENT_REF", reactive.agent_ref)
    real = hosted.ReactiveClient

    def factory(agent_ref: str, tokens: Any) -> Any:
        return real(agent_ref, tokens, transport=reactive.backend.transport)

    monkeypatch.setattr(hosted, "ReactiveClient", factory)
    monkeypatch.setenv("TRAIGENT_RUNTIME_STATE_DIR", str(reactive.state))
    return reactive


def build() -> OptimizedFunction:
    return OptimizedFunction(
        func=answer,
        configuration_space={"temperature": [0.1, 0.3]},
        config_id="answerer",
        best_config_source="agent_head",
        default_config={"temperature": 0.1},
    )


def test_loads_the_pinned_version_with_its_generation(wired: Reactive) -> None:
    wired.backend.head.spec = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "whatever-the-head-pins",
        "config": {"temperature": 0.3},
    }
    fn = build()
    snap = fn.best_config_snapshot
    assert fn.current_config["temperature"] == 0.3
    assert snap.source == BestConfigSource.AGENT_HEAD.value
    assert (
        snap.provenance["head_generation"] == 3 and snap.provenance["head_version"] == 7
    )
    assert len(wired.backend.requests_of("getAgentHeadPinnedConfig")) == 1


def test_unreadable_head_fails_closed_without_falling_back(wired: Reactive) -> None:
    wired.backend.fail_next("getAgentHeadPinnedConfig", "503")
    with pytest.raises(CloudPublishUnavailable) as err:
        build()
    assert err.value.reason is CloudPublishUnavailableReason.REQUEST_FAILED


def test_missing_agent_ref_fails_closed(
    wired: Reactive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRAIGENT_AGENT_REF")
    with pytest.raises(CloudPublishUnavailable):
        build()
    assert wired.backend.log == []
