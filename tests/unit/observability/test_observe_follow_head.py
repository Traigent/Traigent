"""Behaviour of ``observe(follow="head")`` that needs no TVL toolchain.

The Continuous Tuning tests under ``tests/unit/tuning`` skip when the TVL 1.7
toolchain is absent, so the decorator's head-following path is pinned here
without it: argument validation, the closed trace stamp, the in-code default
being served, and a bad head store never reaching the agent call.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.observability import decorators
from traigent.observability.decorators import (
    _build_observe_enrichment_metadata,
    _followed_head,
    observe,
)


def _metadata(client: MagicMock) -> dict[str, Any]:
    return client.start_trace.call_args.kwargs["metadata"]


class TestFollowArguments:
    def test_follow_must_be_none_or_head(self) -> None:
        with pytest.raises(ValueError, match="follow must be None or 'head'"):
            observe("x", follow="tail")

    def test_follow_head_needs_an_agent_or_a_store(self) -> None:
        with pytest.raises(ValueError, match="needs an agent"):
            observe("x", follow="head")

    def test_follow_head_with_only_a_store_is_accepted(self, tmp_path: Path) -> None:
        assert observe("x", follow="head", tuning_store=tmp_path) is not None


class TestFollowHeadStamp:
    def test_sync_call_runs_the_function_and_stamps_unreleased(
        self, tmp_path: Path
    ) -> None:
        client = MagicMock()

        @observe(
            "classify",
            client=client,
            agent="agent-1",
            follow="head",
            tuning_store=tmp_path,
        )
        def classify(x: int) -> int:
            return x + 1

        assert classify(1) == 2
        stamp = _metadata(client)
        assert stamp["agent"] == "agent-1"
        assert stamp["head_state"] == "unreleased"

    def test_factory_metadata_is_kept_beside_the_stamp(self, tmp_path: Path) -> None:
        client = MagicMock()

        @observe(
            "classify",
            client=client,
            agent="a",
            follow="head",
            tuning_store=tmp_path,
            metadata={"team": "blue"},
        )
        def classify() -> str:
            return "ok"

        classify()
        stamp = _metadata(client)
        assert stamp["team"] == "blue"
        assert stamp["head_state"] == "unreleased"

    def test_store_only_follow_stamps_state_without_an_agent_key(
        self, tmp_path: Path
    ) -> None:
        client = MagicMock()

        @observe("classify", client=client, follow="head", tuning_store=tmp_path)
        def classify() -> str:
            return "ok"

        classify()
        stamp = _metadata(client)
        assert "agent" not in stamp
        assert stamp["head_state"] == "unreleased"

    def test_async_call_is_awaited_and_stamped(self, tmp_path: Path) -> None:
        client = MagicMock()

        @observe(
            "classify",
            client=client,
            agent="agent-1",
            follow="head",
            tuning_store=tmp_path,
        )
        async def classify(x: int) -> int:
            return x * 2

        assert asyncio.run(classify(4)) == 8
        assert _metadata(client)["head_state"] == "unreleased"

    def test_followed_head_context_is_cleared_after_the_call(
        self, tmp_path: Path
    ) -> None:
        client = MagicMock()
        seen: list[Any] = []

        @observe(
            "classify", client=client, agent="a", follow="head", tuning_store=tmp_path
        )
        def classify() -> None:
            seen.append(_followed_head.get())

        classify()
        assert seen[0] == {"agent": "a", "head_state": "unreleased"}
        assert _followed_head.get() is None

    def test_context_is_cleared_when_the_function_raises(self, tmp_path: Path) -> None:
        client = MagicMock()

        @observe("boom", client=client, agent="a", follow="head", tuning_store=tmp_path)
        def boom() -> None:
            raise RuntimeError("agent failure")

        with pytest.raises(RuntimeError, match="agent failure"):
            boom()
        assert _followed_head.get() is None

    def test_without_follow_the_metadata_is_untouched(self) -> None:
        client = MagicMock()

        @observe("plain", client=client, metadata={"k": "v"})
        def plain() -> str:
            return "ok"

        plain()
        stamp = _metadata(client)
        assert stamp["k"] == "v"
        assert "head_state" not in stamp
        assert "agent" not in stamp


class TestUnreadableHead:
    def test_bad_head_never_crashes_and_warns_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = MagicMock()
        # a head.json whose stamp differs from the loader's: the store's
        # meta.json is missing, so reading it raises (environment is resolved)
        (tmp_path / "head.json").write_text("{not json", encoding="utf-8")

        @observe(
            "classify",
            client=client,
            agent="a",
            follow="head",
            tuning_store=tmp_path,
            environment="production",
        )
        def classify() -> str:
            return "ok"

        with caplog.at_level(logging.WARNING, logger=decorators.logger.name):
            assert classify() == "ok"
            assert classify() == "ok"

        assert _metadata(client)["head_state"] == "unreadable"
        warnings = [r for r in caplog.records if "unreadable" in r.getMessage()]
        assert len(warnings) == 1  # logged once, not on every call
        assert "serving the in-code default" in warnings[0].getMessage()


class TestEnrichmentWhileFollowingHead:
    def test_followed_head_yields_no_config_enrichment(self) -> None:
        token = _followed_head.set({"agent": "a", "head_state": "unreleased"})
        try:
            assert _build_observe_enrichment_metadata() == {}
        finally:
            _followed_head.reset(token)
