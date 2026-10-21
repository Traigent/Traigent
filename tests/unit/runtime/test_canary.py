"""Projection canary: nothing but the declaration crosses to the backend."""

from __future__ import annotations

import json
from typing import Any

import pytest

import traigent
from traigent.runtime import manifest
from traigent.runtime.apply import apply_requirement
from traigent.runtime.watch import poll_once

from .conftest import (
    AGENT,
    FIXTURES,
    PROJECT,
    FakeBackend,
    backend_parse_import,
    breach,
    head,
)

pytestmark = pytest.mark.backend_online

ENV = "staging-eu-canary7"
NAME = "SecretLatencyName"
HOOK = "secret_hook_callable_name"
LOCAL_STRINGS = (
    NAME,
    HOOK,
    f"{NAME}_monitor",
    f"{NAME}_ladder",
    "alert_opened",
    "restore",
    "fix",
)


def _paths(node: Any, needle: str, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            found += _paths(v, needle, f"{prefix}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += _paths(v, needle, f"{prefix}/{i}")
    elif isinstance(node, str) and node == needle:
        found.append(prefix)
    return found


def _hook(_event) -> None: ...


_hook.__name__ = HOOK


def test_native_route_sends_only_the_declaration(backend: FakeBackend) -> None:
    @traigent.requirement(
        latency_ms=2000,
        max_violation_rate=0.05,
        environment=ENV,
        name=NAME,
        on_breach="fix",
        max_fix_cost_usd=3.0,
        on_event=_hook,
    )
    def fn() -> None: ...

    backend.on("POST", "/production-requirements", 201, head(environment=ENV))
    apply_requirement(fn.requirements[0], project_id=PROJECT, agent_id=AGENT)
    sent = backend.requests[0]
    for local in LOCAL_STRINGS:
        assert local not in sent["raw"], local
    assert "3.0" not in json.dumps(sent["body"])  # the cost ceiling stays local
    assert _paths(sent["body"], ENV) == ["/environment"]


def test_read_requests_carry_no_body_and_no_local_names(backend: FakeBackend) -> None:
    @traigent.requirement(
        latency_ms=2000, max_violation_rate=0.05, name=NAME, on_event=_hook
    )
    def fn() -> None: ...

    backend.on(
        "GET", f"/agents/{AGENT}/production-breaches", 200, {"items": [breach()]}
    )
    poll_once(list(fn.requirements), project_id=PROJECT, agent_id=AGENT)
    for request in backend.requests:
        assert request["body"] is None
        for local in LOCAL_STRINGS:
            assert local not in request["path"]


def test_tvl_route_sends_only_refs_and_the_environment_label(
    backend: FakeBackend,
) -> None:
    pytest.importorskip("tvl.runtime_compiler", reason="needs the runtime-tvl extra")
    pytest.importorskip("scipy")
    fragment = (FIXTURES / "latency.tvl.yml").read_text().replace("production", ENV)
    path = FIXTURES.parent / f"_canary_{id(backend)}.tvl.yml"
    path.write_text(fragment)
    try:

        @traigent.requirement(tvl=str(path), max_fix_cost_usd=3.0, on_event=_hook)
        def fn() -> None: ...

        backend.on(
            "POST", "/production-requirements/tvl-import", 201, head(environment=ENV)
        )
        apply_requirement(fn.requirements[0], project_id=PROJECT, agent_id=AGENT)
    finally:
        path.unlink(missing_ok=True)
    sent = backend.requests[0]
    req = fn.requirements[0]
    assert backend_parse_import(sent["body"]) == "OK"
    for local in (
        "latency_2s",
        "latency_2s_prod",
        "latency_ladder",
        "alert",
        "restore",
        "fix",
        "ok",
        HOOK,
        "traigent.runtime",
    ):
        assert f'"{local}"' not in sent["raw"], local
    assert _paths(sent["body"], ENV) == ["/environment_binding/label"]
    # the contract carries the opaque environment ref, never the label
    ref = sent["body"]["environment_binding"]["ref"]
    assert ENV not in json.dumps(sent["body"]["contract"])
    assert ref in json.dumps(sent["body"]["contract"])
    token_map = manifest.ensure_tokens(manifest.default_state_dir(), [], persist=False)
    assert {t["local"] for t in token_map["tokens"]} >= {req.name, req.monitor_name}
    assert all(t["hosted"] not in (t["local"],) for t in token_map["tokens"])
