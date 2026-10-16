"""Label-service transport against a fake Backend that validates every body.

The fake serves the vector binding chain, validates PUT/read bodies against the Schema
label contracts (TraigentSchema worktree) and decrypts what it receives with the vector
project key, so the test proves the wire shape AND that the sealed name is what was sent.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from traigent.labels import open_label
from traigent.labels.transport import (
    LabelTransport,
    LabelTransportError,
    load_root_public_key,
    send_session_labels,
)

from .conftest import FIXTURES

SCHEMA_DIR = os.environ.get("TRAIGENT_SCHEMA_DIR")
SCHEMA_WT = Path(SCHEMA_DIR or "")
LABELS_DIR = SCHEMA_WT / "traigent_schema" / "schemas" / "labels"
pytestmark = pytest.mark.skipif(
    not SCHEMA_DIR or not LABELS_DIR.exists(),
    reason="TRAIGENT_SCHEMA_DIR not set or TraigentSchema worktree absent",
)

NAME = "Alice_has_diabetes"  # a customer-chosen name: content, sealed before it leaves
AGENT_REF = "agentref_" + "ab" * 16
DATASET_REF = "datasetref_" + "cd" * 16


def _validator(definition: str, schema_file: str = "label_io_schema.json"):
    from jsonschema import Draft7Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT7

    registry = Registry()
    for path in LABELS_DIR.glob("*.json"):
        doc = json.loads(path.read_text("utf-8"))
        registry = registry.with_resource(
            doc["$id"], Resource.from_contents(doc, default_specification=DRAFT7)
        )
    root = json.loads((LABELS_DIR / schema_file).read_text("utf-8"))
    schema = {"$ref": f"{root['$id']}#/definitions/{definition}"}
    return Draft7Validator(schema, registry=registry)


@pytest.fixture()
def vectors() -> dict:
    return json.loads((FIXTURES / "label_vectors_v0.json").read_text("utf-8"))


class FakeLabelBackend:
    def __init__(self, vectors: dict) -> None:
        self.v = vectors
        self.revisions: dict[str, int] = {}
        self.plaintexts: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.binding_list = vectors["binding_list"]
        self.write_check = _validator("LabelWriteRequest")
        self.sk = X25519PrivateKey.from_private_bytes(
            bytes.fromhex(vectors["keys"]["project_private_hex"])
        )

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/api/v1/labels/binding-list", self.binding)
        app.router.add_put("/api/v1/labels/objects", self.put)
        app.router.add_post("/api/v1/labels/objects/read", self.read)
        return app

    @staticmethod
    def _ok(data: dict) -> web.Response:
        return web.json_response(
            {"success": True, "message": "label service", "data": data}
        )

    @staticmethod
    def _err(code: str, status: int) -> web.Response:
        return web.json_response(
            {"success": False, "error": code, "error_code": code, "message": "x"},
            status=status,
        )

    async def binding(self, request: web.Request) -> web.Response:
        self.calls.append(("GET", "binding-list"))
        assert not request.query_string
        return self._ok(self.binding_list)

    async def read(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.calls.append(("POST", "read"))
        assert set(body) == {"object_refs"} and 1 <= len(body["object_refs"]) <= 200
        items = []
        for ref in body["object_refs"]:
            if ref in self.revisions:
                items.append({"object_ref": ref, "revision": self.revisions[ref]})
        return self._ok({"items": items, "next_cursor": None})

    async def put(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.calls.append(("PUT", "objects"))
        errors = [e.message for e in self.write_check.iter_errors(body)]
        assert not errors, errors
        ref = body["object_ref"]
        if body["base_revision"] != self.revisions.get(ref, 0):
            return self._err("stale_revision", 409)
        assert body["envelope"]["key_epoch"] == body["epoch"] == 1
        revision = body["base_revision"] + 1
        text = open_label(
            self.sk,
            body["envelope"],
            tenant_ref=self.v["refs"]["tenant_ref"],
            project_ref=self.v["refs"]["project_ref"],
            purpose=body["purpose"],
            object_ref=ref,
            revision=revision,
        )
        self.plaintexts[ref] = text
        self.revisions[ref] = revision
        return self._ok(
            {
                "object_ref": ref,
                "purpose": body["purpose"],
                "revision": revision,
                "epoch": body["epoch"],
                "updated_at": "2026-10-16T00:00:02Z",
            }
        )


def _config(tmp_path: Path, vectors: dict, **tweak: str) -> Path:
    root = bytes.fromhex(vectors["keys"]["root_public_hex"])
    section = {
        "root_fingerprint": vectors["keys"]["root_fingerprint_sha256"],
        "tenant_ref": vectors["refs"]["tenant_ref"],
        "project_ref": vectors["refs"]["project_ref"],
        "root_public_key": base64.urlsafe_b64encode(root).decode().rstrip("="),
        **tweak,
    }
    path = tmp_path / "traigent.yml"
    lines = ["labels:", "  trust_anchor:"] + [
        f'    {k}: "{v}"' for k, v in section.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def _headers() -> dict[str, str]:
    return {"X-API-Key": "uk_test"}


async def _with_server(vectors: dict, fn):
    fake = FakeLabelBackend(vectors)
    async with TestServer(fake.app()) as server:
        base = f"http://{server.host}:{server.port}/api/v1"
        return fake, await fn(fake, base)


def test_configured_writer_sends_the_sealed_name_once(
    tmp_path: Path, vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        first = await send_session_labels(
            {"agent": NAME, "dataset": "My dataset", "evaluator": None},
            {"agent": AGENT_REF, "dataset": DATASET_REF},
            base,
            _headers,
            cfg,
        )
        second = await send_session_labels(
            {"agent": "renamed elsewhere"}, {"agent": AGENT_REF}, base, _headers, cfg
        )
        return first, second

    fake, (first, second) = asyncio.run(_with_server(vectors, go))
    assert first == {AGENT_REF: 1, DATASET_REF: 1}
    assert fake.plaintexts == {AGENT_REF: NAME, DATASET_REF: "My dataset"}
    assert second == {}  # an object that already has a label is never overwritten
    assert fake.plaintexts[AGENT_REF] == NAME
    assert [c for c in fake.calls if c[0] == "PUT"] == [("PUT", "objects")] * 2


def test_without_config_nothing_is_sent(
    tmp_path: Path, vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # no traigent.yml anywhere above tmp_path

    async def go(fake, base):
        return await send_session_labels(
            {"agent": NAME}, {"agent": AGENT_REF}, base, _headers, None
        )

    fake, result = asyncio.run(_with_server(vectors, go))
    assert result == {}
    assert fake.calls == []


def test_untrusted_root_sends_nothing(
    tmp_path: Path, vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    cfg = _config(tmp_path, vectors, root_fingerprint="1" * 64)

    async def go(fake, base):
        return await send_session_labels(
            {"agent": NAME}, {"agent": AGENT_REF}, base, _headers, cfg
        )

    fake, result = asyncio.run(_with_server(vectors, go))
    assert result == {}
    assert [c for c in fake.calls if c[0] in ("PUT", "POST")] == []


def test_rolled_back_binding_list_is_refused(
    tmp_path: Path, vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    (tmp_path / "labels_state.json").write_text(
        json.dumps(
            {f"{vectors['refs']['tenant_ref']}/{vectors['refs']['project_ref']}": 9}
        ),
        encoding="utf-8",
    )
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        return await send_session_labels(
            {"agent": NAME}, {"agent": AGENT_REF}, base, _headers, cfg
        )

    fake, result = asyncio.run(_with_server(vectors, go))
    assert result == {}
    assert fake.plaintexts == {}


def test_closed_error_codes_and_no_body_echo(vectors: dict) -> None:
    fake = FakeLabelBackend(vectors)

    async def run():
        async with TestServer(fake.app()) as server:
            base = f"http://{server.host}:{server.port}/api/v1"
            transport = LabelTransport(base, _headers)
            fake.revisions[AGENT_REF] = 3
            sender_env = vectors["label_envelopes"][0]["envelope"]
            with pytest.raises(LabelTransportError) as info:
                await transport.write_label(
                    object_ref=AGENT_REF,
                    purpose="label",
                    base_revision=0,
                    epoch=1,
                    envelope=sender_env,
                )
            return info.value

    err = asyncio.run(run())
    assert err.code == "stale_revision" and err.status == 409
    assert NAME not in str(err)


def test_root_public_key_is_validated(tmp_path: Path, vectors: dict) -> None:
    from traigent.labels.trust_anchor import TrustAnchorError

    cfg = _config(tmp_path, vectors)
    assert load_root_public_key(cfg) == bytes.fromhex(
        vectors["keys"]["root_public_hex"]
    )
    bad = _config(tmp_path, vectors, root_public_key="AAAA")
    with pytest.raises(TrustAnchorError):
        load_root_public_key(bad)


def test_session_create_path_sends_labels_through_the_writer(
    tmp_path: Path, vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.cloud.api_operations import ApiOperations
    from traigent.cloud.models import SessionCreationRequest

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    _config(tmp_path, vectors)
    monkeypatch.chdir(tmp_path)  # traigent.yml is found walking up from here

    request = SessionCreationRequest(
        function_name="pkg.fn", agent_key=NAME, dataset_id="My dataset"
    )

    async def go(fake, base):
        client = MagicMock()
        client.backend_config.api_base_url = base
        client.auth_manager.augment_headers = AsyncMock(return_value={"X-API-Key": "k"})
        ops = ApiOperations(client)
        result = MagicMock()
        result.content_less_refs = {"agent": AGENT_REF, "dataset": DATASET_REF}
        await ops._send_content_less_labels(request, result)

    fake, _ = asyncio.run(_with_server(vectors, go))
    assert fake.plaintexts == {AGENT_REF: NAME, DATASET_REF: "My dataset"}


# ------------------------------------------------- durable, monotonic sequence retention


def _scope(vectors: dict) -> str:
    return f"{vectors['refs']['tenant_ref']}/{vectors['refs']['project_ref']}"


def test_unpersistable_sequence_means_no_writer_and_no_write(
    tmp_path: Path, vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")  # a file where the state dir must be
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(blocker / "sub"))
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        return await send_session_labels(
            {"agent": NAME}, {"agent": AGENT_REF}, base, _headers, cfg
        )

    fake, result = asyncio.run(_with_server(vectors, go))
    assert result == {}
    assert fake.plaintexts == {}
    assert [c for c in fake.calls if c[0] in ("PUT", "POST")] == []


def test_corrupt_state_fails_closed(
    tmp_path: Path, vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    (tmp_path / "labels_state.json").write_text("{not json", encoding="utf-8")
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        return await send_session_labels(
            {"agent": NAME}, {"agent": AGENT_REF}, base, _headers, cfg
        )

    fake, result = asyncio.run(_with_server(vectors, go))
    assert result == {} and fake.plaintexts == {}


def test_sequence_is_persisted_and_an_older_list_is_refused_after_restart(
    tmp_path: Path, vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    cfg = _config(tmp_path, vectors)
    state = tmp_path / "labels_state.json"

    async def first(fake, base):
        return await send_session_labels(
            {"agent": NAME}, {"agent": AGENT_REF}, base, _headers, cfg
        )

    _fake, sent = asyncio.run(_with_server(vectors, first))
    assert sent == {AGENT_REF: 1}
    assert json.loads(state.read_text("utf-8"))[_scope(vectors)] == 1
    assert not list(tmp_path.glob("labels_state.tmp*"))  # temp file renamed away

    # another client has since verified a newer list (sequence 2) and retained it
    state.write_text(json.dumps({_scope(vectors): 2}), encoding="utf-8")

    async def restarted(
        fake, base
    ):  # a new process: new sender, replayed sequence-1 list
        return await send_session_labels(
            {"dataset": "x"}, {"dataset": DATASET_REF}, base, _headers, cfg
        )

    fake, replay = asyncio.run(_with_server(vectors, restarted))
    assert replay == {} and fake.plaintexts == {}
    assert json.loads(state.read_text("utf-8"))[_scope(vectors)] == 2  # never lowered


def test_retention_is_monotonic(tmp_path: Path) -> None:
    from traigent.labels.transport import _retain_sequence

    path = tmp_path / "s.json"
    _retain_sequence(path, "scope", 3)
    _retain_sequence(path, "scope", 3)
    with pytest.raises(LabelTransportError):
        _retain_sequence(path, "scope", 2)
    assert json.loads(path.read_text("utf-8")) == {"scope": 3}


def test_label_routes_pass_the_gate_only_when_sealed(vectors: dict) -> None:
    """The gate now knows the label routes: an envelope passes, a plaintext name is refused."""
    from traigent.labels import transport_gate as gate

    good = {
        "object_ref": AGENT_REF,
        "purpose": "label",
        "base_revision": 0,
        "epoch": 1,
        "envelope": vectors["label_envelopes"][0]["envelope"],
    }
    url = "https://backend.example.test/api/v1/labels/objects"
    gate.set_allowed_origins(["https://backend.example.test"])
    gate.set_mode("refuse")
    try:
        assert gate.inspect_outgoing("PUT", url, json.dumps(good)) is not None
        for bad in (
            {**good, "object_ref": NAME},
            {**good, "purpose": NAME},
            {**good, "name": NAME},
        ):
            with pytest.raises(gate.TransportGateRefusal):
                gate.inspect_outgoing("PUT", url, json.dumps(bad))
        read_url = url + "/read"
        assert gate.inspect_outgoing(
            "POST", read_url, json.dumps({"object_refs": [AGENT_REF]})
        )
        with pytest.raises(gate.TransportGateRefusal):
            gate.inspect_outgoing("POST", read_url, json.dumps({"object_refs": [NAME]}))
    finally:
        gate.set_mode(None)
        gate.set_allowed_origins(None)


# --------------------------------- per-client API base: the real path, no hand-set origins


def _refuse_mode():
    from traigent.labels import transport_gate as gate

    gate.set_mode("refuse")
    gate.set_allowed_origins(
        None
    )  # the configured backends only; the test server is NOT one
    return gate


def test_label_transport_gates_its_own_non_default_base(vectors: dict) -> None:
    """A LabelTransport built for a per-client base is gated at THAT origin."""
    from traigent.labels import seal_label

    envelope = seal_label(
        bytes.fromhex(vectors["keys"]["project_public_hex"]),
        NAME,
        tenant_ref=vectors["refs"]["tenant_ref"],
        project_ref=vectors["refs"]["project_ref"],
        purpose="label",
        object_ref=AGENT_REF,
        key_epoch=1,
        revision=1,
    )

    async def go(fake, base):
        transport = LabelTransport(base, _headers)
        assert base.startswith("http://127.0.0.1")  # not the configured backend
        # a plaintext name where an opaque ref belongs: refused before it is sent
        with pytest.raises(gate.TransportGateRefusal):
            await transport.write_label(
                object_ref=NAME,
                purpose="label",
                base_revision=0,
                epoch=1,
                envelope=envelope,
            )
        with pytest.raises(gate.TransportGateRefusal):
            await transport.read_labels([NAME])
        refused_puts = [c for c in fake.calls if c[0] in ("PUT", "POST")]
        # a sealed, well-formed request to the same origin passes
        await transport.read_labels([AGENT_REF])
        written = await transport.write_label(
            object_ref=AGENT_REF,
            purpose="label",
            base_revision=0,
            epoch=1,
            envelope=envelope,
        )
        return refused_puts, written

    gate = _refuse_mode()
    try:
        fake, (refused_puts, written) = asyncio.run(_with_server(vectors, go))
    finally:
        gate.set_mode(None)
    assert refused_puts == []
    assert written["object_ref"] == AGENT_REF


def test_strict_session_fails_closed_on_an_unregistered_origin(vectors: dict) -> None:
    import aiohttp

    from traigent.labels import transport_gate as gate

    async def go(fake_a, base_a):
        async def inner(fake_b, base_b):
            session = gate.gate_session(
                aiohttp.ClientSession(), origins=[base_a], strict=True
            )
            async with session:
                with pytest.raises(gate.TransportGateRefusal):
                    await session.get(f"{base_b}/labels/binding-list")
                ok = await session.get(f"{base_a}/labels/binding-list")
                assert ok.status == 200
            return fake_b.calls

        _fb, calls_b = await _with_server(vectors, inner)
        return calls_b

    gate.set_mode("report")  # even outside refuse mode a strict session fails closed
    try:
        _fa, calls_b = asyncio.run(_with_server(vectors, go))
    finally:
        gate.set_mode(None)
    assert calls_b == []


def test_api_operations_gates_the_per_client_base(
    vectors: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ApiOperations built with a non-default api base: refuse blocks plaintext there."""
    for name in ("TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.setenv(name, "false")
    from traigent.cloud.api_operations import ApiOperations
    from traigent.labels import transport_gate as gate

    N = NAME

    async def go(fake, base):
        client = MagicMock()
        client.backend_config.api_base_url = base
        client.no_egress = False
        ops = ApiOperations(client)
        bad = {
            "function_name": "f",
            "configuration_space": {N: {"type": "categorical", "choices": [N]}},
            "objectives": [{"name": N, "orientation": "maximize", "weight": 1.0}],
            "dataset_metadata": {"size": 1},
            "max_trials": 1,
        }
        with pytest.raises(gate.TransportGateRefusal):
            await ops._post_session_creation(bad, {}, None)
        return None

    sessions: list[str] = []

    async def sessions_handler(request: web.Request) -> web.Response:
        sessions.append(request.path)
        return web.json_response({"session_id": "s"}, status=500)

    async def main():
        fake = FakeLabelBackend(vectors)
        app = fake.app()
        app.router.add_post("/api/v1/sessions", sessions_handler)
        async with TestServer(app) as server:
            base = f"http://{server.host}:{server.port}/api/v1"
            await go(fake, base)
            # control: report mode lets the same request through, so the refusal above
            # came from the gate at this origin, not from an unreachable server
            gate.set_mode("report")
            client = MagicMock()
            client.backend_config.api_base_url = base
            client.no_egress = False
            from traigent.cloud.api_operations import ApiOperations as Ops

            try:
                await Ops(client)._post_session_creation(
                    {"function_name": "f", "configuration_space": {}, "objectives": []},
                    {},
                    None,
                )
            except Exception:  # noqa: BLE001 - the 500 is expected; only reachability matters
                pass

    gate_ = _refuse_mode()
    try:
        asyncio.run(main())
    finally:
        gate_.set_mode(None)
    assert sessions == ["/api/v1/sessions"]  # only the control reached the server


def test_other_session_factories_register_their_own_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every gated session factory that talks to a per-client base registers it."""
    for name in ("TRAIGENT_OFFLINE", "TRAIGENT_OFFLINE_MODE"):
        monkeypatch.setenv(name, "false")
    import requests

    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig
    from traigent.labels.transport_gate import _instance_origins, _origin

    manager = SyncManager(TraigentConfig(), api_key="uk_" + "a" * 43)
    origin = _origin(manager.base_url)
    assert isinstance(manager.session, requests.Session)
    assert origin in _instance_origins(manager.session)

    async def factories():
        from traigent.cloud.backend_client import (
            BackendClientConfig,
            BackendIntegratedClient,
        )
        from traigent.cloud.client import TraigentCloudClient

        base = "http://127.0.0.1:9"
        cloud = TraigentCloudClient(api_key="uk_" + "a" * 43, base_url=base)
        cloud.auth = MagicMock(get_headers=AsyncMock(return_value={"X-API-Key": "k"}))
        session = await cloud._ensure_session()
        assert _origin(cloud.api_base_url) in _instance_origins(session)
        await cloud.close()

        backend = BackendIntegratedClient(
            api_key="uk_" + "a" * 43,
            backend_config=BackendClientConfig(backend_base_url=base),
        )
        backend.auth_manager = MagicMock()
        backend.auth_manager.auth.get_headers = AsyncMock(
            return_value={"X-API-Key": "k"}
        )
        built = await backend._build_http_session()
        assert _origin(base) in _instance_origins(built)
        await built.close()

    asyncio.run(factories())


# ------------------------------------------------ strict sessions: redirects, own origins


class _Recorder:
    """A tiny aiohttp server that records every request (method, path, headers)."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, dict[str, str]]] = []

    async def handle(self, request: web.Request) -> web.Response:
        self.seen.append((request.method, request.path, dict(request.headers)))
        return web.json_response({"success": True, "data": {"items": []}})

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.handle)
        return app


def _redirecting_app(location_base: str, status: int) -> web.Application:
    async def redirect(request: web.Request) -> web.Response:
        return web.Response(
            status=status, headers={"Location": f"{location_base}{request.path}"}
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", redirect)
    return app


@pytest.mark.parametrize("status", [307, 308])
def test_strict_session_refuses_a_cross_origin_redirect(
    vectors: dict, status: int
) -> None:
    from traigent.labels import transport_gate as gate

    rec = _Recorder()

    async def main():
        async with TestServer(rec.app()) as target:
            tbase = f"http://{target.host}:{target.port}"
            async with TestServer(_redirecting_app(tbase, status)) as first:
                base = f"http://{first.host}:{first.port}/api/v1"

                async def secret_headers() -> dict[str, str]:
                    return {"X-API-Key": "uk_must_not_travel"}

                transport = LabelTransport(base, secret_headers)
                with pytest.raises(gate.TransportGateRefusal) as read_err:
                    await transport.read_labels([AGENT_REF])
                with pytest.raises(gate.TransportGateRefusal) as bin_err:
                    await transport.read_binding_list()
                return read_err.value, bin_err.value

    gate.set_mode("report")
    try:
        errors = asyncio.run(main())
    finally:
        gate.set_mode(None)
    assert rec.seen == []  # the redirect target saw no request and no headers
    for err in errors:
        assert err.route == "redirect_refused"
        assert "uk_must_not_travel" not in str(err)


def test_strict_httpx_and_requests_sessions_refuse_redirects() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import httpx
    import requests

    from traigent.labels import transport_gate as gate

    seen: list[str] = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()

        do_PUT = do_POST = do_GET

        def log_message(self, *a: object) -> None:
            pass

    class Redirector(Target):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/x")
            self.end_headers()

        do_PUT = do_POST = do_GET

    target = HTTPServer(("127.0.0.1", 0), Target)
    first = HTTPServer(("127.0.0.1", 0), Redirector)
    threads = [
        threading.Thread(target=s.serve_forever, daemon=True) for s in (target, first)
    ]
    for t in threads:
        t.start()
    base = f"http://127.0.0.1:{first.server_port}"
    gate.set_mode("report")
    try:
        sync_client = gate.gate_session(httpx.Client(), origins=[base], strict=True)
        with pytest.raises(gate.TransportGateRefusal) as e1:
            sync_client.get(f"{base}/a", headers={"X-API-Key": "k"})
        sess = gate.gate_session(requests.Session(), origins=[base], strict=True)
        with pytest.raises(gate.TransportGateRefusal) as e2:
            sess.get(f"{base}/a", headers={"X-API-Key": "k"})

        async def amain():
            client = gate.gate_session(httpx.AsyncClient(), origins=[base], strict=True)
            async with client:
                with pytest.raises(gate.TransportGateRefusal) as e3:
                    await client.get(f"{base}/a", headers={"X-API-Key": "k"})
            return e3.value

        e3 = asyncio.run(amain())
    finally:
        gate.set_mode(None)
        for s in (target, first):
            s.shutdown()
            s.server_close()
    assert seen == []
    assert {e1.value.route, e2.value.route, e3.route} == {"redirect_refused"}


def test_strict_is_strict_to_the_instance_not_the_process_backend(
    vectors: dict,
) -> None:
    """A second origin that IS a configured backend for the process is still refused."""
    import aiohttp

    from traigent.labels import transport_gate as gate

    rec = _Recorder()

    async def main():
        async with TestServer(rec.app()) as other:
            other_base = f"http://{other.host}:{other.port}"
            async with TestServer(rec.app()) as own:
                own_base = f"http://{own.host}:{own.port}"
                gate.set_allowed_origins([other_base])  # configured for the process
                assert gate._origin(other_base) in gate.backend_origins()
                session = gate.gate_session(
                    aiohttp.ClientSession(), origins=[own_base], strict=True
                )
                async with session:
                    with pytest.raises(gate.TransportGateRefusal) as err:
                        await session.get(f"{other_base}/api/v1/labels/binding-list")
                    ok = await session.get(f"{own_base}/api/v1/labels/binding-list")
                    assert ok.status == 200
                return err.value

    gate.set_mode("report")
    try:
        err = asyncio.run(main())
    finally:
        gate.set_mode(None)
        gate.set_allowed_origins(None)
    assert err.route == "unrecognised_origin"
    assert len(rec.seen) == 1  # only the request to its own origin was ever sent


def test_origin_normalisation() -> None:
    from traigent.labels.transport_gate import _origin

    assert _origin("HTTPS://Example.COM/api/v1/") == ("https", "example.com", 443)
    assert _origin("https://example.com:443") == ("https", "example.com", 443)
    assert _origin("http://example.com") == ("http", "example.com", 80)
    assert _origin("http://example.com:80/x/y?z=1") == ("http", "example.com", 80)
    assert _origin("http://[::1]:8080/a") == ("http", "::1", 8080)
    assert _origin("http://[::1]/") == ("http", "::1", 80)
    assert _origin("http://[0:0:0:0:0:0:0:1]/") == ("http", "::1", 80)
    assert _origin("http://EXAMPLE.com./a") == ("http", "example.com", 80)
    assert _origin("http://127.0.0.1:9") == ("http", "127.0.0.1", 9)
    assert _origin("not a url") is None
    assert _origin("http://example.com:99999") is None
