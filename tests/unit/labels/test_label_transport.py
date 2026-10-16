"""Label-service transport against a fake Backend that validates every body.

The fake serves the vector binding chain, validates PUT/read bodies against the Schema
label contracts (TraigentSchema worktree) and decrypts what it receives with the vector
project key, so the test proves the wire shape AND that the sealed name is what was sent.
"""

from __future__ import annotations

import asyncio
import base64
import json
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

SCHEMA_WT = Path(
    "/home/nimrod/TraigentProjects/core_project/worktrees/TraigentSchema/content-less"
)
LABELS_DIR = SCHEMA_WT / "traigent_schema" / "schemas" / "labels"
pytestmark = pytest.mark.skipif(
    not LABELS_DIR.exists(), reason="TraigentSchema worktree absent"
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
