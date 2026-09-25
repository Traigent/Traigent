"""Opt-in purpose-key grant fetch (content identity Release 1, item 8).

With ``TraigentConfig(content_identity=True)`` or ``TRAIGENT_CONTENT_IDENTITY=1``
a run that talks to a Backend fetches its tenant's grant from
``POST /api/v1/content-identity/purpose-keys`` once, before session create,
and installs it for that run only. Off by default: no request, payloads
unchanged. Every failure degrades to "no keys" (no ``content_identity`` block,
exactly the no-grant behaviour); key material never reaches logs or payloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.shared.mocks.optimizers import MockOptimizer
from traigent.cloud.backend_client import (
    BackendIntegratedClient,
    PurposeKeyGrantFetchError,
)
from traigent.cloud.client import CloudEgressBlockedError
from traigent.config.types import TraigentConfig, content_identity_enabled
from traigent.core.metadata_helpers import build_backend_metadata
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.identity.grant_fetch import fetch_content_identity_keys
from traigent.identity.keys import (
    ContentIdentityKeys,
    clear_content_identity_keys,
    get_content_identity_keys,
    set_content_identity_keys,
)

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "content_identity_v1_vectors.json").read_text(
        encoding="utf-8"
    )
)
_ROW = VECTORS["key_derivation"][0]
GRANT: dict[str, Any] = {
    "tenant_id": _ROW["tenant_id"],
    "kid": _ROW["key_id"],
    "example_id_key": _ROW["example_id_key_hex"],
    "example_version_key": _ROW["example_version_key_hex"],
    "encoding": "hex",
}
SECRETS = (GRANT["example_id_key"], GRANT["example_version_key"])
FAKE_TRAIGENT_API_KEY = "tg_" + "x" * 61  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "100.0")
    monkeypatch.delenv("TRAIGENT_CONTENT_IDENTITY", raising=False)
    clear_content_identity_keys()
    yield
    clear_content_identity_keys()


def _response(status: int, payload: Any = None, *, bad_json: bool = False) -> Any:
    response = MagicMock()
    response.status_code = status
    response.headers = {"Cache-Control": "no-store, max-age=0"}
    response.text = json.dumps(payload)
    if bad_json:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = payload
    return response


def _error(status: int, code: str) -> Any:
    return _response(status, {"success": False, "error": code, "error_code": code})


def _client(*, for_offline_run: bool = False) -> BackendIntegratedClient:
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    client.auth_manager.auth.get_headers = AsyncMock(  # type: ignore[method-assign]
        return_value={"Authorization": "Bearer test-token"}
    )
    if for_offline_run:
        # The end-to-end runs stay OFFLINE (their session create must never
        # reach a real Backend); only this client's grant request is let past
        # the egress guard, and its transport (requests.post) is mocked.
        client._raise_if_backend_egress_disabled = (  # type: ignore[method-assign]
            lambda operation: None
        )
    return client


# --------------------------------------------------------------------------
# The switch
# --------------------------------------------------------------------------


def test_switch_defaults_off() -> None:
    assert TraigentConfig().content_identity is None
    assert TraigentConfig.from_environment().content_identity is None
    assert content_identity_enabled(TraigentConfig()) is False
    assert "content_identity" not in TraigentConfig().to_dict()
    assert "content_identity" not in TraigentConfig(content_identity=True).to_dict()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_env_var_turns_the_switch_on(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TRAIGENT_CONTENT_IDENTITY", value)
    assert TraigentConfig.from_environment().content_identity is True
    assert content_identity_enabled(TraigentConfig()) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_env_var_off_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TRAIGENT_CONTENT_IDENTITY", value)
    assert content_identity_enabled(TraigentConfig()) is False


def test_explicit_config_overrides_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_CONTENT_IDENTITY", "1")
    assert content_identity_enabled(TraigentConfig(content_identity=False)) is False
    monkeypatch.setenv("TRAIGENT_CONTENT_IDENTITY", "0")
    assert content_identity_enabled(TraigentConfig(content_identity=True)) is True


@pytest.mark.parametrize("value", ["false", "False", " off ", "0", "no"])
def test_string_opt_out_beats_the_env(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # YAML/JSON configs carry strings; an opt-out string must not be ignored.
    monkeypatch.setenv("TRAIGENT_CONTENT_IDENTITY", "1")
    assert content_identity_enabled(TraigentConfig(content_identity=value)) is False
    from_dict = TraigentConfig.from_dict({"content_identity": value})
    assert content_identity_enabled(from_dict) is False


@pytest.mark.parametrize("value", ["true", "1", "YES", "on"])
def test_string_opt_in_is_honoured(value: str) -> None:
    config = TraigentConfig(content_identity=value)
    assert config.content_identity is True
    assert content_identity_enabled(config) is True


@pytest.mark.parametrize("value", ["maybe", "", 1, 0.0, ["false"]])
def test_unknown_content_identity_value_fails_loud(value: Any) -> None:
    with pytest.raises(ValueError, match="content_identity must be a boolean"):
        TraigentConfig(content_identity=value)


@pytest.mark.parametrize("assign", ["attr", "item", "merge"])
def test_string_opt_out_after_construction_beats_the_env(
    monkeypatch: pytest.MonkeyPatch, assign: str
) -> None:
    monkeypatch.setenv("TRAIGENT_CONTENT_IDENTITY", "1")
    config = TraigentConfig()
    if assign == "attr":
        config.content_identity = "false"  # type: ignore[assignment]
    elif assign == "item":
        config["content_identity"] = "off"
    else:
        config = config.merge({"content_identity": "no"})
    assert config.content_identity is False
    assert content_identity_enabled(config) is False


def test_unknown_value_assigned_after_construction_fails_loud() -> None:
    config = TraigentConfig()
    with pytest.raises(ValueError, match="content_identity must be a boolean"):
        config.content_identity = "maybe"  # type: ignore[assignment]


def _gate(config: TraigentConfig, client: Any) -> Any:
    fake_self = SimpleNamespace(traigent_config=config, backend_client=client)
    return OptimizationOrchestrator._content_identity_grant_client(fake_self)  # type: ignore[arg-type]


def test_gate_is_closed_when_the_switch_is_off() -> None:
    assert _gate(TraigentConfig(), object()) is None


def test_gate_opens_with_config_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    client = object()
    assert _gate(TraigentConfig(content_identity=True), client) is client
    monkeypatch.setenv("TRAIGENT_CONTENT_IDENTITY", "true")
    assert _gate(TraigentConfig(), client) is client
    assert _gate(TraigentConfig(content_identity=False), client) is None


def test_gate_is_closed_without_a_backend_client() -> None:
    assert _gate(TraigentConfig(content_identity=True), None) is None


def test_gate_is_closed_in_privacy_mode() -> None:
    config = TraigentConfig(content_identity=True)
    with pytest.warns(DeprecationWarning):
        config.privacy_enabled = True
    assert _gate(config, object()) is None


# --------------------------------------------------------------------------
# Backend client: the request and its failure statuses
# --------------------------------------------------------------------------


@pytest.mark.backend_online
@patch("requests.post")
def test_client_posts_to_the_purpose_keys_route(mock_post: Any) -> None:
    mock_post.return_value = _response(200, GRANT)
    grant = _client().fetch_content_identity_grant_sync()
    assert grant == GRANT
    call = mock_post.call_args
    assert call.args[0] == "https://api.test/api/v1/content-identity/purpose-keys"
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert "json" not in call.kwargs and "data" not in call.kwargs  # no body
    assert mock_post.call_count == 1


@pytest.mark.backend_online
@patch("requests.post")
def test_client_refuses_redirects_and_bounds_the_wait(mock_post: Any) -> None:
    # A redirect would carry X-API-Key to another host and accept key material
    # from it; requests follows redirects unless told not to.
    mock_post.return_value = _grant_ok = _response(200, GRANT)
    _client().fetch_content_identity_grant_sync()
    kwargs = mock_post.call_args.kwargs
    assert kwargs["allow_redirects"] is False
    connect, read = kwargs["timeout"]
    assert 0 < connect <= 5.0 and 0 < read <= 10.0
    assert _grant_ok.json.called


@pytest.mark.backend_online
@pytest.mark.parametrize("status", [301, 302, 307, 308])
@patch("requests.post")
def test_client_treats_a_redirect_as_a_failed_fetch(
    mock_post: Any, status: int
) -> None:
    redirect = _response(status, None)
    redirect.headers = {"Location": "https://attacker.test/keys"}
    mock_post.return_value = redirect
    with pytest.raises(PurposeKeyGrantFetchError) as caught:
        _client().fetch_content_identity_grant_sync()
    assert caught.value.status_code == status
    assert mock_post.call_count == 1


@pytest.mark.backend_online
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "unauthorized"),
        (403, "tenant_context_required"),
        (404, "not_found"),
        (429, "RATE_LIMIT_EXCEEDED"),
        (503, "content_identity_keys_unavailable"),
    ],
)
@patch("requests.post")
def test_client_raises_a_content_free_error_per_status(
    mock_post: Any, status: int, code: str
) -> None:
    mock_post.return_value = _error(status, code)
    with pytest.raises(PurposeKeyGrantFetchError) as caught:
        _client().fetch_content_identity_grant_sync()
    assert caught.value.status_code == status
    assert caught.value.error_code == code
    assert mock_post.call_count == 1  # never retried


@pytest.mark.backend_online
@patch("requests.post")
def test_client_rejects_a_non_object_body(mock_post: Any) -> None:
    mock_post.return_value = _response(200, bad_json=True)
    with pytest.raises(PurposeKeyGrantFetchError):
        _client().fetch_content_identity_grant_sync()
    mock_post.return_value = _response(200, [GRANT])
    with pytest.raises(PurposeKeyGrantFetchError):
        _client().fetch_content_identity_grant_sync()


def test_client_refuses_when_egress_is_disabled() -> None:
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test", no_egress=True
    )
    with patch("requests.post") as mock_post, pytest.raises(CloudEgressBlockedError):
        client.fetch_content_identity_grant_sync()
    mock_post.assert_not_called()


# --------------------------------------------------------------------------
# fetch_content_identity_keys: fail-safe, content-free logging
# --------------------------------------------------------------------------


def _source(result: Any) -> Any:
    fake = MagicMock()
    if isinstance(result, BaseException):
        fake.fetch_content_identity_grant_sync.side_effect = result
    else:
        fake.fetch_content_identity_grant_sync.return_value = result
    return fake


def test_fetch_parses_a_valid_grant(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    keys = fetch_content_identity_keys(_source(dict(GRANT)))
    assert isinstance(keys, ContentIdentityKeys)
    assert keys.kid == GRANT["kid"]
    for secret in SECRETS:
        assert secret not in caplog.text
        assert secret not in repr(keys)


@pytest.mark.parametrize(
    "result",
    [
        PurposeKeyGrantFetchError(401, "unauthorized"),
        PurposeKeyGrantFetchError(403, "tenant_context_required"),
        PurposeKeyGrantFetchError(404, None),
        PurposeKeyGrantFetchError(429, "RATE_LIMIT_EXCEEDED"),
        PurposeKeyGrantFetchError(503, "content_identity_keys_unavailable"),
        ConnectionError("network down"),
        TimeoutError(),
        RuntimeError("unexpected"),
        {**GRANT, "extra": "field"},
        {**GRANT, "encoding": "base64"},
        {k: v for k, v in GRANT.items() if k != "kid"},
        {**GRANT, "example_id_key": GRANT["example_id_key"].upper()},
        {"success": True, "data": GRANT},  # an envelope is not the bare grant
        "not a mapping",
    ],
)
def test_every_failure_yields_no_keys(
    result: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    assert fetch_content_identity_keys(_source(result)) is None
    assert get_content_identity_keys() is None
    for secret in SECRETS:
        assert secret not in caplog.text
        assert secret.upper() not in caplog.text


@pytest.mark.parametrize(
    ("result", "named"),
    [
        (
            PurposeKeyGrantFetchError(503, "content_identity_keys_unavailable"),
            "HTTP 503",
        ),
        (PurposeKeyGrantFetchError(401, None), "HTTP 401"),
        (ConnectionError("secret-ish detail"), "ConnectionError"),
        ({**GRANT, "extra": "x"}, "malformed grant"),
    ],
)
def test_one_warning_names_only_the_failure_type(
    caplog: pytest.LogCaptureFixture, result: Any, named: str
) -> None:
    caplog.set_level(logging.DEBUG)
    fetch_content_identity_keys(_source(result))
    warnings_logged = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings_logged) == 1
    message = warnings_logged[0].getMessage()
    assert named in message
    assert "secret-ish detail" not in message
    assert "content_identity_keys_unavailable" not in message


# --------------------------------------------------------------------------
# End to end: an optimize() run with the switch on
# --------------------------------------------------------------------------


def identity_agent(text: str) -> str:
    return text.upper()


def _dataset() -> Dataset:
    rows = [("a", "A", "row-a"), ("b", "B", "row-b"), ("a", "A", "row-a2")]
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": text},
                expected_output=expected,
                metadata={"example_id": user_id},
            )
            for text, expected, user_id in rows
        ],
        name="identity_dataset",
    )


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    content_identity: bool,
    grant_client: Any,
) -> tuple[dict[str, Any], Any]:
    from traigent.core.backend_session_manager import BackendSessionManager

    captured: dict[str, Any] = {}
    original = BackendSessionManager.create_session

    def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        # The grant is installed before session create, not after.
        captured["_keys_at_session_create"] = get_content_identity_keys()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BackendSessionManager, "create_session", capture)
    if grant_client is not None:
        # The test runs offline (no real Backend session); route the gate to a
        # Backend client whose HTTP transport is mocked.
        real_gate = OptimizationOrchestrator._content_identity_grant_client

        def gate(self: Any) -> Any:
            opened = real_gate(
                SimpleNamespace(
                    traigent_config=self.traigent_config, backend_client=grant_client
                )
            )
            return opened

        monkeypatch.setattr(
            OptimizationOrchestrator, "_content_identity_grant_client", gate
        )
    optimizer = MockOptimizer(config_space={"alpha": [0, 1]}, objectives=["accuracy"])
    optimizer.set_max_suggestions(2)
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=LocalEvaluator(metrics=["accuracy"], detailed=True),
        max_trials=2,
        config=TraigentConfig(
            offline=True, algorithm="grid", content_identity=content_identity
        ),
        agent_key="agent_identity_test",
    )
    result = await orchestrator.optimize(identity_agent, _dataset())
    return captured, result


@pytest.mark.asyncio
async def test_switch_off_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    with patch("requests.post") as mock_post:
        captured, result = await _run(
            monkeypatch,
            content_identity=False,
            grant_client=_client(for_offline_run=True),
        )
    mock_post.assert_not_called()
    assert "content_identity" not in captured
    assert all("content_identity" not in t.metadata for t in result.trials)


@pytest.mark.asyncio
async def test_switch_on_fetches_once_and_sends_the_grant_kid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with patch("requests.post", return_value=_response(200, dict(GRANT))) as post:
        captured, result = await _run(
            monkeypatch,
            content_identity=True,
            grant_client=_client(for_offline_run=True),
        )
    assert post.call_count == 1
    assert captured["_keys_at_session_create"] is not None
    wire = captured["content_identity"]
    assert wire["key_status"] == "available"
    assert wire["key_id"] == GRANT["kid"]
    for trial in result.trials:
        block = trial.metadata["content_identity"]
        assert block["evaluated"]["key_id"] == GRANT["kid"]
    # Keys live for the run only.
    assert get_content_identity_keys() is None
    # No key material anywhere: logs, session payload, trial payloads.
    payload_text = json.dumps(
        {k: v for k, v in captured.items() if not k.startswith("_")}, default=str
    )
    trial_text = json.dumps(
        [
            build_backend_metadata(t, "accuracy", TraigentConfig(), "identity_dataset")
            for t in result.trials
        ]
        + [t.metadata for t in result.trials],
        default=str,
    )
    for secret in SECRETS:
        for text in (caplog.text, payload_text, trial_text):
            assert secret not in text
            assert secret.upper() not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _error(401, "unauthorized"),
        _error(403, "tenant_context_required"),
        _error(404, "not_found"),
        _error(429, "RATE_LIMIT_EXCEEDED"),
        _error(503, "content_identity_keys_unavailable"),
        _response(200, {**GRANT, "extra": "field"}),
        _response(200, bad_json=True),
    ],
)
async def test_failed_fetch_runs_without_content_identity(
    monkeypatch: pytest.MonkeyPatch, response: Any
) -> None:
    with patch("requests.post", return_value=response):
        captured, result = await _run(
            monkeypatch,
            content_identity=True,
            grant_client=_client(for_offline_run=True),
        )
    assert len(result.trials) == 2
    assert "content_identity" not in captured  # == key_status unavailable
    assert all("content_identity" not in t.metadata for t in result.trials)
    assert get_content_identity_keys() is None


def _comparable(captured: dict[str, Any], result: Any) -> Any:
    from tests.unit.identity.no_grant_snapshot import _normalize

    return _normalize(
        {
            "session_create": {
                k: v
                for k, v in captured.items()
                if k not in ("func", "dataset", "function_descriptor", "start_time")
                and not k.startswith("_")
            },
            "trials": [
                {
                    "metadata": t.metadata,
                    "backend": build_backend_metadata(
                        t, "accuracy", TraigentConfig(), "identity_dataset"
                    ),
                }
                for t in result.trials
            ],
        }
    )


@pytest.mark.asyncio
async def test_failed_fetch_payloads_equal_the_switch_off_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with patch("requests.post") as off_post:
        off = await _run(
            monkeypatch,
            content_identity=False,
            grant_client=_client(for_offline_run=True),
        )
    off_post.assert_not_called()
    with patch(
        "requests.post", return_value=_error(503, "content_identity_keys_unavailable")
    ) as on_post:
        failed = await _run(
            monkeypatch,
            content_identity=True,
            grant_client=_client(for_offline_run=True),
        )
    assert on_post.call_count == 1
    assert _comparable(*failed) == _comparable(*off)


@pytest.mark.asyncio
async def test_network_error_runs_without_content_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        captured, result = await _run(
            monkeypatch,
            content_identity=True,
            grant_client=_client(for_offline_run=True),
        )
    assert len(result.trials) == 2
    assert "content_identity" not in captured


@pytest.mark.asyncio
async def test_user_installed_keys_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_keys = ContentIdentityKeys.from_grant(GRANT)
    set_content_identity_keys(user_keys)
    with patch("requests.post") as mock_post:
        captured, _ = await _run(
            monkeypatch,
            content_identity=True,
            grant_client=_client(for_offline_run=True),
        )
    mock_post.assert_not_called()
    assert captured["content_identity"]["key_id"] == GRANT["kid"]
    assert get_content_identity_keys() is user_keys  # left installed


@pytest.mark.asyncio
async def test_privacy_mode_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _source(dict(GRANT))

    real_gate = OptimizationOrchestrator._content_identity_grant_client

    def gate(self: Any) -> Any:
        with pytest.warns(DeprecationWarning):
            self.traigent_config.privacy_enabled = True
        return real_gate(
            SimpleNamespace(traigent_config=self.traigent_config, backend_client=fake)
        )

    monkeypatch.setattr(
        OptimizationOrchestrator, "_content_identity_grant_client", gate
    )
    captured, _ = await _run(monkeypatch, content_identity=True, grant_client=None)
    fake.fetch_content_identity_grant_sync.assert_not_called()
    assert "content_identity" not in captured


# --------------------------------------------------------------------------
# Fetched keys are scoped to the run that fetched them (review finding on
# #2423): two optimize() runs in one process never see each other's grant,
# and one run ending never clears keys another run is still using.
# --------------------------------------------------------------------------

_ROW_B = VECTORS["key_derivation"][2]
GRANT_B: dict[str, Any] = {
    "tenant_id": _ROW_B["tenant_id"],
    "kid": _ROW_B["key_id"],
    "example_id_key": _ROW_B["example_id_key_hex"],
    "example_version_key": _ROW_B["example_version_key_hex"],
    "encoding": "hex",
}
_WAIT = 10.0  # seconds; a deadlock fails the test instead of hanging it


def _expected_example_ids(grant: dict[str, Any]) -> set[str]:
    from traigent.identity.examples import identify_dataset

    identity = identify_dataset(_dataset(), ContentIdentityKeys.from_grant(grant))
    assert identity is not None
    return {ex.example_id for ex in identity.examples}


def _concurrent_capture(monkeypatch: pytest.MonkeyPatch) -> dict[Any, Any]:
    """Session-create kwargs per agent function (each run has its own agent)."""
    from traigent.core.backend_session_manager import BackendSessionManager

    captured: dict[Any, Any] = {}
    original = BackendSessionManager.create_session

    def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured[kwargs.get("func")] = dict(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BackendSessionManager, "create_session", capture)
    return captured


async def _scoped_run(agent: Any, *, content_identity: bool, grant_source: Any) -> Any:
    optimizer = MockOptimizer(config_space={"alpha": [0, 1]}, objectives=["accuracy"])
    optimizer.set_max_suggestions(2)
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=LocalEvaluator(metrics=["accuracy"], detailed=True),
        max_trials=2,
        config=TraigentConfig(
            offline=True, algorithm="grid", content_identity=content_identity
        ),
        agent_key="agent_identity_test",
    )
    real_gate = OptimizationOrchestrator._content_identity_grant_client
    # Per-instance gate: this run's (fake, offline) Backend grant source.
    orchestrator._content_identity_grant_client = lambda: real_gate(  # type: ignore[method-assign]
        SimpleNamespace(
            traigent_config=orchestrator.traigent_config, backend_client=grant_source
        )
    )
    return await orchestrator.optimize(agent, _dataset())


def _assert_all_trials_keyed(result: Any, grant: dict[str, Any]) -> None:
    expected_ids = _expected_example_ids(grant)
    assert len(result.trials) == 2
    for trial in result.trials:
        assert trial.metadata["content_identity"]["evaluated"]["key_id"] == grant["kid"]
        examples = list(trial.metadata.get("example_results") or [])
        assert len(examples) == 3
        assert {e["example_id"] for e in examples} == expected_ids


def _assert_no_content_identity(captured: dict[str, Any], result: Any) -> None:
    assert "content_identity" not in captured
    for trial in result.trials:
        assert "content_identity" not in trial.metadata
        for example in trial.metadata.get("example_results") or []:
            assert not str(example["example_id"]).startswith("ex1:")


@pytest.mark.asyncio
async def test_switch_off_run_never_sees_a_concurrent_runs_fetched_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _concurrent_capture(monkeypatch)
    a_in_trial = asyncio.Event()
    release_a = asyncio.Event()

    async def agent_a(text: str) -> str:
        # Run A holds its fetched grant, mid-trial, for the whole of run B.
        a_in_trial.set()
        await asyncio.wait_for(release_a.wait(), _WAIT)
        return text.upper()

    async def agent_b(text: str) -> str:
        return text.upper()

    source_a = _source(dict(GRANT))
    source_b = _source(dict(GRANT_B))

    async def run_b() -> Any:
        await asyncio.wait_for(a_in_trial.wait(), _WAIT)
        try:
            return await _scoped_run(
                agent_b, content_identity=False, grant_source=source_b
            )
        finally:
            release_a.set()

    result_a, result_b = await asyncio.gather(
        _scoped_run(agent_a, content_identity=True, grant_source=source_a), run_b()
    )
    source_b.fetch_content_identity_grant_sync.assert_not_called()
    _assert_no_content_identity(captured[agent_b], result_b)
    assert captured[agent_a]["content_identity"]["key_id"] == GRANT["kid"]
    _assert_all_trials_keyed(result_a, GRANT)
    assert get_content_identity_keys() is None


@pytest.mark.asyncio
async def test_concurrent_runs_each_use_their_own_tenants_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _concurrent_capture(monkeypatch)
    b_in_trial = asyncio.Event()
    a_done = asyncio.Event()

    async def agent_a(text: str) -> str:
        return text.upper()

    async def agent_b(text: str) -> str:
        # Run B is mid-trial while run A starts, runs and FINISHES; B's later
        # results must still carry B's keys (A's teardown clears nothing of B's).
        b_in_trial.set()
        await asyncio.wait_for(a_done.wait(), _WAIT)
        return text.upper()

    source_a = _source(dict(GRANT))
    source_b = _source(dict(GRANT_B))

    async def run_a() -> Any:
        await asyncio.wait_for(b_in_trial.wait(), _WAIT)
        try:
            return await _scoped_run(
                agent_a, content_identity=True, grant_source=source_a
            )
        finally:
            a_done.set()

    result_b, result_a = await asyncio.gather(
        _scoped_run(agent_b, content_identity=True, grant_source=source_b), run_a()
    )
    source_a.fetch_content_identity_grant_sync.assert_called_once()
    source_b.fetch_content_identity_grant_sync.assert_called_once()
    assert captured[agent_a]["content_identity"]["key_id"] == GRANT["kid"]
    assert captured[agent_b]["content_identity"]["key_id"] == GRANT_B["kid"]
    _assert_all_trials_keyed(result_a, GRANT)
    _assert_all_trials_keyed(result_b, GRANT_B)
    assert GRANT["kid"] != GRANT_B["kid"]
    assert get_content_identity_keys() is None


@pytest.mark.asyncio
async def test_threaded_sync_trials_carry_the_runs_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync agents run in worker threads; per-trial ids still use the run's keys."""
    captured = _concurrent_capture(monkeypatch)
    worker_threads: set[int] = set()
    a_in_trial = threading.Event()
    b_done = threading.Event()

    def sync_agent_a(text: str) -> str:
        # Blocks a worker thread (A's keys live) until run B has finished.
        worker_threads.add(threading.get_ident())
        a_in_trial.set()
        assert b_done.wait(_WAIT)
        return text.upper()

    def sync_agent_b(text: str) -> str:
        worker_threads.add(threading.get_ident())
        return text.upper()

    async def run_b() -> Any:
        # Start B only once A's keys are live and A is evaluating.
        assert await asyncio.to_thread(a_in_trial.wait, _WAIT)
        try:
            return await _scoped_run(
                sync_agent_b,
                content_identity=True,
                grant_source=_source(dict(GRANT_B)),
            )
        finally:
            b_done.set()

    result_a, result_b = await asyncio.gather(
        _scoped_run(
            sync_agent_a, content_identity=True, grant_source=_source(dict(GRANT))
        ),
        run_b(),
    )
    assert threading.get_ident() not in worker_threads  # really off the loop thread
    _assert_all_trials_keyed(result_a, GRANT)
    _assert_all_trials_keyed(result_b, GRANT_B)
    assert captured[sync_agent_a]["content_identity"]["key_id"] == GRANT["kid"]
    assert captured[sync_agent_b]["content_identity"]["key_id"] == GRANT_B["kid"]


def test_fetched_keys_follow_copied_context_into_executor_threads() -> None:
    """Pool threads see the run's keys only under a copy of the run's context."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    from traigent.identity.keys import (
        bind_run_content_identity_keys,
        reset_run_content_identity_keys,
    )

    keys = ContentIdentityKeys.from_grant(GRANT)
    token = bind_run_content_identity_keys(keys)
    try:
        assert get_content_identity_keys() is keys
        with ThreadPoolExecutor(max_workers=1) as pool:
            copied = pool.submit(
                contextvars.copy_context().run, get_content_identity_keys
            ).result()
            bare = pool.submit(get_content_identity_keys).result()
        assert copied is keys
        assert bare is None  # fail closed: no context, no keys
    finally:
        reset_run_content_identity_keys(token)
    assert get_content_identity_keys() is None


def test_user_installed_keys_outrank_run_scoped_keys() -> None:
    from traigent.identity.keys import (
        bind_run_content_identity_keys,
        installed_content_identity_keys,
        reset_run_content_identity_keys,
    )

    fetched = ContentIdentityKeys.from_grant(GRANT_B)
    user = ContentIdentityKeys.from_grant(GRANT)
    token = bind_run_content_identity_keys(fetched)
    try:
        set_content_identity_keys(user)
        assert get_content_identity_keys() is user
        assert installed_content_identity_keys() is user
        clear_content_identity_keys()
        assert get_content_identity_keys() is fetched
        assert installed_content_identity_keys() is None
    finally:
        reset_run_content_identity_keys(token)
