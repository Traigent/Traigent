"""optimization_only: a 200 ``/keys/validate`` answering ``optimization_only: true`` latches
the mode for that credential's scope before validation returns.

Recording transports only: no real network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

from .harness import capture_egress
from .scenario import BACKEND_URL, FAKE_API_KEY
from .test_optimization_only_egress import DIRTY_TYPED_SESSION, _env
from .test_optimization_only_latch_guards import REFUSING_SITES

pytestmark = pytest.mark.backend_online

OTHER_KEY = "uk_" + "z" * 43
SITES = ("hybrid_create", "dataset_upload", "workflow_traces", "hosted_optimization")


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _env(monkeypatch, tmp_path)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def _scope(key: str = FAKE_API_KEY) -> str:
    scope = oo.credential_scope(BACKEND_URL, {"X-API-Key": key})
    assert scope is not None
    return scope


def _validate(response: dict[str, Any]) -> Any:
    from traigent.cloud.auth import AuthManager, UnifiedAuthConfig

    auth = AuthManager(
        UnifiedAuthConfig(backend_base_url=BACKEND_URL, cache_credentials=False)
    )
    with capture_egress() as cap:
        cap.backend.validate_response = response
        result = asyncio.run(auth._validate_api_key_with_backend(FAKE_API_KEY))
    assert [r for r in cap.requests if r.path.endswith("/keys/validate")]
    return result


def test_true_latches_scope_and_everything_after_is_gated() -> None:
    assert _validate({"valid": True, "optimization_only": True}) is None
    assert oo.is_latched(_scope())
    for name in SITES:
        with capture_egress() as cap:
            with pytest.raises(oo.OptimizationOnlyRefusal):
                asyncio.run(REFUSING_SITES[name]())
        assert cap.requests == [], (name, [r.route for r in cap.requests])


def test_session_create_body_is_projected_after_latch() -> None:
    from .test_optimization_only_egress import _create_session_raw, _session_posts

    _validate({"valid": True, "optimization_only": True})
    expected = oo.project_for_phase(oo.PHASE_SESSION_CREATE_AUTO, DIRTY_TYPED_SESSION)
    with capture_egress() as cap:
        asyncio.run(_create_session_raw(DIRTY_TYPED_SESSION))
    (post,) = _session_posts(cap)
    assert json.loads(post.body) == expected
    assert "Alice" not in post.body.decode()


@pytest.mark.parametrize(
    "response",
    [
        {"valid": True, "optimization_only": False},
        {"valid": True},
        {"valid": True, "optimization_only": "true"},
        {"valid": True, "optimization_only": 1},
        {"valid": True, "optimization_only": None},
        {"valid": True, "optimization_only": [True]},
    ],
)
def test_anything_but_json_true_does_not_latch(response: dict[str, Any]) -> None:
    assert _validate(response) is None
    assert not oo.is_latched(_scope())
    assert oo.latched_origins() == set()


def test_false_never_unlatches() -> None:
    oo.latch_on(_scope())
    _validate({"valid": True, "optimization_only": False})
    assert oo.is_latched(_scope())


def test_explicit_false_client_does_not_override_the_latch() -> None:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    client = BackendIntegratedClient(
        api_key=FAKE_API_KEY,
        backend_config=BackendClientConfig(backend_base_url=BACKEND_URL),
        optimization_only=False,
    )
    _validate({"valid": True, "optimization_only": True})
    assert oo.is_active(oo.policy_of(client))


def test_other_credential_is_unaffected() -> None:
    _validate({"valid": True, "optimization_only": True})
    assert oo.is_latched(_scope())
    assert not oo.is_latched(_scope(OTHER_KEY))
    assert not oo.is_active(oo.Policy(None, _scope(OTHER_KEY)))


def test_invalid_key_body_does_not_latch() -> None:
    oo.reset_latch()
    assert (
        oo.note_key_validation({"valid": False, "optimization_only": True}, _scope())
        is False
    )
    assert oo.note_key_validation({"optimization_only": True}, None) is False
    assert not oo.is_latched(_scope())


def test_requests_fallback_path_latches_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("traigent.cloud.auth.AIOHTTP_AVAILABLE", False)
    assert _validate({"valid": True, "optimization_only": True}) is None
    assert oo.is_latched(_scope())
