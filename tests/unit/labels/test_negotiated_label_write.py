"""Label writes on the negotiated route (content-less round 1b-2b, SDK half).

With ``TRAIGENT_TOKEN_STORE_ENABLED`` on, a label write goes to
``POST /api/v1/optimization/labels`` (Schema ``optimization/label_write_schema.json``); off, it
is byte-for-byte the legacy ``PUT /api/v1/labels/objects``. The fake Backend validates every
body against the Schema, decrypts what it receives with the vector project key, and answers
the closed refusal envelope on demand, so the wire shape, the closed error mapping, the
retry pacing, the privacy canary and the journal transition are all tested against it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from traigent.labels import token_store as ts
from traigent.labels import transport as tr
from traigent.labels.token_store import TokenNamespace, TokenStore, ensure_tokens
from traigent.labels.transport import (
    LabelTransport,
    LabelTransportError,
    send_session_labels,
)

from .test_label_transport import (
    AGENT_REF,
    LABELS_DIR,
    FakeLabelBackend,
    _config,
    _headers,
)

SCHEMA_DIR = os.environ.get("TRAIGENT_SCHEMA_DIR")
OPT_DIR = Path(SCHEMA_DIR or "") / "traigent_schema" / "schemas" / "optimization"
pytestmark = pytest.mark.skipif(
    not SCHEMA_DIR or not LABELS_DIR.exists() or not OPT_DIR.exists(),
    reason="TRAIGENT_SCHEMA_DIR not set or TraigentSchema worktree absent",
)

HEBREW = "שלום_עולם_טמפרטורה"
EMAIL = "alice.smith+secret@example.com"
CANARIES = (HEBREW, EMAIL)
KNOB_REF = "knobref_" + "5a" * 16
METRIC_REF = "metricref_" + "6b" * 16
#: the object the first vector envelope is bound to (the legacy fake decrypts it)
VECTOR_REF = "agentref_1fd55a1c631df55176a0db51fd7a99d7"
ORIGIN = "https://api.traigent.test"

UNAVAILABLE = {
    "success": False,
    "error": "service_unavailable",
    "error_code": "service_unavailable",
    "message": "Service is temporarily unavailable.",
}


def _refusal(code: str) -> dict[str, Any]:
    status, message = tr._NEGOTIATED_REFUSALS[code]
    return {"success": False, "error": code, "error_code": code, "message": message}


def _write_validator():
    from jsonschema import Draft7Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT7

    registry = Registry()
    for path in [*LABELS_DIR.glob("*.json"), OPT_DIR / "label_write_schema.json"]:
        doc = json.loads(path.read_text("utf-8"))
        registry = registry.with_resource(
            doc["$id"], Resource.from_contents(doc, default_specification=DRAFT7)
        )
    root = json.loads((OPT_DIR / "label_write_schema.json").read_text("utf-8"))
    return Draft7Validator(
        {"$ref": f"{root['$id']}#/definitions/LabelWriteRequest"}, registry=registry
    )


class NegotiatedBackend(FakeLabelBackend):
    """The legacy fake plus ``POST /api/v1/optimization/labels`` with scripted answers."""

    def __init__(self, vectors: dict) -> None:
        super().__init__(vectors)
        self.neg_check = _write_validator()
        self.neg_raw: list[tuple[str, dict[str, str], bytes]] = []
        self.legacy_raw: list[tuple[str, dict[str, str], bytes]] = []
        self.script: list[Any] = []
        self.errors: list[str] = []

    def app(self) -> web.Application:
        app = super().app()
        app.router.add_post("/api/v1/optimization/labels", self.negotiated)
        return app

    async def put(self, request: web.Request) -> web.Response:
        self.legacy_raw.append(
            (request.path_qs, dict(request.headers), await request.read())
        )
        return await super().put(request)

    async def negotiated(self, request: web.Request) -> web.Response:
        raw = await request.read()
        self.calls.append(("POST", "optimization-labels"))
        self.neg_raw.append((request.path_qs, dict(request.headers), raw))
        body = json.loads(raw)
        errors = [e.message for e in self.neg_check.iter_errors(body)]
        if errors:
            self.errors.extend(errors)
        if self.script:
            kind, *rest = self.script.pop(0)
            if kind == "json":
                status, payload, headers = rest
                return web.json_response(payload, status=status, headers=headers)
            if kind == "text":
                return web.Response(status=rest[0], text=rest[1])
            if kind == "404":
                return web.Response(status=404, text="404: Not Found")
        ref = body["object_ref"]
        if body["base_revision"] != self.revisions.get(ref, 0):
            return web.json_response(_refusal("stale_revision"), status=409)
        revision = body["base_revision"] + 1
        from traigent.labels import open_label
        from traigent.labels._codec import LabelCryptoError

        try:
            self.plaintexts[ref] = open_label(
                self.sk,
                body["envelope"],
                tenant_ref=self.v["refs"]["tenant_ref"],
                project_ref=self.v["refs"]["project_ref"],
                purpose=body["purpose"],
                object_ref=ref,
                revision=revision,
            )
        except LabelCryptoError:
            pass  # a transport-level test sending a vector envelope bound to another ref
        self.revisions[ref] = revision
        return web.json_response(
            {
                "object_ref": ref,
                "purpose": body["purpose"],
                "revision": revision,
                "epoch": body["epoch"],
                "updated_at": "2026-10-19T00:00:02Z",
            }
        )


@pytest.fixture()
def vectors() -> dict:
    from .conftest import FIXTURES

    return json.loads((FIXTURES / "label_vectors_v0.json").read_text("utf-8"))


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "state"))
    (tmp_path / "state").mkdir(mode=0o700, exist_ok=True)
    (tmp_path / "state").chmod(0o700)
    monkeypatch.setenv(ts.ENV_FLAG, "true")


@pytest.fixture()
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(tr, "_sleep", fake_sleep)
    return recorded


def _run(vectors: dict, fn):
    fake = NegotiatedBackend(vectors)

    async def main():
        async with TestServer(fake.app()) as server:
            base = f"http://{server.host}:{server.port}/api/v1"
            return await fn(fake, base)

    return fake, asyncio.run(main())


def _envelope(vectors: dict) -> dict[str, Any]:
    return dict(vectors["label_envelopes"][0]["envelope"])


async def _write(transport: LabelTransport, vectors: dict, **over: Any):
    args: dict[str, Any] = {
        "object_ref": KNOB_REF,
        "purpose": "knob",
        "base_revision": 0,
        "epoch": 1,
        "envelope": _envelope(vectors),
    }
    args.update(over)
    return await transport.write_label(**args)


# ------------------------------------------------------------------ routing (flag on / off)


def test_flag_on_writes_to_the_negotiated_route_with_a_schema_valid_body(
    tmp_path: Path, vectors: dict
) -> None:
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        return await send_session_labels(
            {"agent": HEBREW}, {"agent": AGENT_REF}, base, _headers, cfg
        )

    fake, sent = _run(vectors, go)
    assert sent == {AGENT_REF: 1}
    assert fake.errors == []  # every body validated against label_write_schema.json
    assert [c for c in fake.calls if c[0] in ("PUT", "POST") and c[1] != "read"] == [
        ("POST", "optimization-labels")
    ]
    assert fake.legacy_raw == []  # the legacy route is never called with the flag on
    path, _headers_seen, raw = fake.neg_raw[0]
    assert path == "/api/v1/optimization/labels"
    body = json.loads(raw)
    assert set(body) == {
        "schema_version",
        "object_ref",
        "purpose",
        "base_revision",
        "epoch",
        "envelope",
    }
    assert body["schema_version"] == 1
    assert fake.plaintexts == {AGENT_REF: HEBREW}  # the sealed name is what was sent


@pytest.mark.parametrize("flag", [None, "", "false", "0", "no", "off"])
def test_flag_off_is_the_legacy_put_and_never_touches_the_new_route(
    tmp_path: Path, vectors: dict, monkeypatch, flag
) -> None:
    if flag is None:
        monkeypatch.delenv(ts.ENV_FLAG)
    else:
        monkeypatch.setenv(ts.ENV_FLAG, flag)
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        return await send_session_labels(
            {"agent": HEBREW}, {"agent": AGENT_REF}, base, _headers, cfg
        )

    fake, sent = _run(vectors, go)
    assert sent == {AGENT_REF: 1}
    assert fake.neg_raw == []  # the new route is never called
    assert ("POST", "optimization-labels") not in fake.calls
    path, _h, raw = fake.legacy_raw[0]
    assert path == "/api/v1/labels/objects"
    assert "schema_version" not in json.loads(raw)  # today's body, byte for byte
    assert fake.plaintexts == {AGENT_REF: HEBREW}


def test_flag_off_legacy_bytes_are_unchanged(vectors: dict, monkeypatch) -> None:
    monkeypatch.setenv(ts.ENV_FLAG, "false")

    async def go(fake, base):
        return await _write(
            LabelTransport(base, _headers),
            vectors,
            object_ref=VECTOR_REF,
            purpose="label",
        )

    fake, _ = _run(vectors, go)
    body = json.loads(fake.legacy_raw[0][2])
    assert list(body) == ["object_ref", "purpose", "base_revision", "epoch", "envelope"]


def test_flag_is_read_at_call_time(vectors: dict, monkeypatch) -> None:
    async def go(fake, base):
        transport = LabelTransport(base, _headers)
        monkeypatch.setenv(ts.ENV_FLAG, "false")
        await _write(transport, vectors, object_ref=VECTOR_REF, purpose="label")
        monkeypatch.setenv(ts.ENV_FLAG, "1")
        await _write(transport, vectors, object_ref=METRIC_REF, purpose="metric")

    fake, _ = _run(vectors, go)
    assert len(fake.legacy_raw) == 1 and len(fake.neg_raw) == 1


# ------------------------------------------------------------- closed error mapping


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "invalid_request"),
        (422, "reference_not_issued"),
        (409, "stale_revision"),
        (409, "stale_epoch"),
        (409, "conflict"),
        (500, "internal_error"),
    ],
)
def test_every_declared_refusal_is_a_closed_final_code(
    vectors: dict, status: int, code: str
) -> None:
    async def go(fake, base):
        fake.script = [("json", status, _refusal(code), {})]
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors)
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == code and err.status == status and err.retryable is False
    assert len(fake.neg_raw) == 1  # no retry
    assert fake.legacy_raw == []


def test_reference_not_issued_is_final_even_when_retry_after_is_offered(
    vectors: dict, sleeps: list[float]
) -> None:
    async def go(fake, base):
        fake.script = [
            ("json", 422, _refusal("reference_not_issued"), {"Retry-After": "1"})
        ]
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors)
        return info.value

    fake, err = _run(vectors, go)
    assert (
        err.code == "reference_not_issued" and sleeps == [] and len(fake.neg_raw) == 1
    )


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "unauthenticated"), (403, "forbidden"), (429, "rate_limited")],
)
def test_declared_non_envelope_statuses_have_closed_codes(
    vectors: dict, status: int, code: str
) -> None:
    async def go(fake, base):
        fake.script = [("json", status, {"detail": HEBREW}, {})]
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors)
        return info.value

    _fake, err = _run(vectors, go)
    assert err.code == code and HEBREW not in str(err) and HEBREW not in repr(err)


def test_dark_route_404_is_closed_and_never_falls_back_to_the_legacy_route(
    tmp_path: Path, vectors: dict
) -> None:
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        fake.script = [("404",)]
        sender = tr.sender_from_config(base, _headers, cfg)
        assert sender is not None
        with pytest.raises(LabelTransportError) as info:
            await sender.send_new({AGENT_REF: ("label", HEBREW)})
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "label_route_unavailable" and err.status == 404
    assert len(fake.neg_raw) == 1
    assert (
        fake.legacy_raw == []
    )  # fail closed: the unvalidated legacy write never happens
    assert fake.plaintexts == {}


@pytest.mark.parametrize(
    "answer",
    [
        ("json", 400, {**_refusal("conflict")}, {}),  # a code with the wrong status
        ("json", 409, {**_refusal("conflict"), "message": HEBREW}, {}),  # wrong text
        ("json", 409, {**_refusal("conflict"), "extra": 1}, {}),  # not the closed shape
        ("json", 500, {**_refusal("internal_error"), "error": "conflict"}, {}),
        ("json", 500, {"success": False, "error_code": "never_heard_of_it"}, {}),
        ("json", 400, [HEBREW], {}),
        ("text", 500, HEBREW),
        ("text", 502, "<html>bad gateway</html>"),
        ("json", 503, {"detail": HEBREW}, {}),  # a bare 503 is not the closed envelope
        ("json", 503, {**UNAVAILABLE, "message": HEBREW}, {}),
        ("json", 201, {}, {}),
    ],
)
def test_unknown_code_or_shape_is_a_closed_internal_error_without_retry(
    vectors: dict, sleeps: list[float], answer: tuple[Any, ...]
) -> None:
    async def go(fake, base):
        fake.script = [answer]
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors)
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "internal_error"
    assert sleeps == [] and len(fake.neg_raw) == 1
    rendered = "".join(traceback.format_exception(err)) + repr(err)
    assert HEBREW not in rendered and "bad gateway" not in rendered


@pytest.mark.parametrize(
    "ack",
    [
        {},
        {"object_ref": KNOB_REF, "purpose": "knob", "revision": 1, "epoch": 1},
        {
            "object_ref": KNOB_REF,
            "purpose": "knob",
            "revision": 1,
            "epoch": 1,
            "updated_at": "x",
            "envelope": {},
        },
        {
            "object_ref": METRIC_REF,  # not the object that was written
            "purpose": "knob",
            "revision": 1,
            "epoch": 1,
            "updated_at": "x",
        },
        {
            "object_ref": KNOB_REF,
            "purpose": "knob",
            "revision": 2,  # a first write is revision 1
            "epoch": 1,
            "updated_at": "x",
        },
        {
            "object_ref": KNOB_REF,
            "purpose": "knob",
            "revision": True,
            "epoch": 1,
            "updated_at": "x",
        },
        {
            "object_ref": KNOB_REF,
            "purpose": "knob",
            "revision": 1,
            "epoch": 2,
            "updated_at": "x",
        },
    ],
)
def test_a_success_body_that_is_not_the_contract_is_a_closed_internal_error(
    vectors: dict, ack: dict[str, Any]
) -> None:
    async def go(fake, base):
        fake.script = [("json", 200, ack, {})]
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors)
        return info.value

    _fake, err = _run(vectors, go)
    assert err.code == "internal_error"


def test_the_closed_refusals_match_the_schema() -> None:
    schema = json.loads((OPT_DIR / "label_write_schema.json").read_text("utf-8"))
    refusal = schema["definitions"]["LabelWriteRefusal"]
    codes = refusal["properties"]["error_code"]["enum"]
    messages = refusal["properties"]["message"]["enum"]
    assert list(tr._NEGOTIATED_REFUSALS) == codes
    assert [m for _s, m in tr._NEGOTIATED_REFUSALS.values()] == messages
    endpoints = json.loads((OPT_DIR / "label_write_endpoints.json").read_text("utf-8"))
    responses = endpoints["paths"]["/api/v1/optimization/labels"]["post"]["responses"]
    for code, (status, _message) in tr._NEGOTIATED_REFUSALS.items():
        assert str(status) in responses, code


# ------------------------------------------------------------- 503 service_unavailable


def test_503_is_retried_honoring_retry_after_then_succeeds(
    vectors: dict, sleeps: list[float]
) -> None:
    async def go(fake, base):
        fake.script = [
            ("json", 503, UNAVAILABLE, {"Retry-After": "2"}),
            ("json", 503, UNAVAILABLE, {"Retry-After": "3"}),
        ]
        return await _write(LabelTransport(base, _headers), vectors)

    fake, ack = _run(vectors, go)
    assert ack["revision"] == 1
    assert sleeps == [2.0, 3.0]
    assert len(fake.neg_raw) == 3 and fake.errors == []
    assert len({raw for _p, _h, raw in fake.neg_raw}) == 1  # the same body each time


def test_503_without_retry_after_uses_the_bounded_backoff(
    vectors: dict, sleeps: list[float]
) -> None:
    async def go(fake, base):
        fake.script = [("json", 503, UNAVAILABLE, {})] * 3
        return await _write(LabelTransport(base, _headers), vectors)

    _fake, _ack = _run(vectors, go)
    assert sleeps == [0.5, 1.0, 2.0]


def test_503_forever_is_closed_retryable_after_the_attempt_budget(
    vectors: dict, sleeps: list[float]
) -> None:
    async def go(fake, base):
        fake.script = [("json", 503, UNAVAILABLE, {"Retry-After": "0"})] * 50
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors)
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "service_unavailable" and err.status == 503
    assert err.retryable is True
    assert len(fake.neg_raw) == tr._MAX_WRITE_ATTEMPTS


def test_503_with_a_retry_after_beyond_the_deadline_is_not_waited_for(
    vectors: dict, sleeps: list[float], monkeypatch
) -> None:
    monkeypatch.setattr(tr, "_WRITE_DEADLINE_SECONDS", 5.0)

    async def go(fake, base):
        fake.script = [("json", 503, UNAVAILABLE, {"Retry-After": "120"})]
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors)
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "service_unavailable" and err.retryable
    assert sleeps == [] and len(fake.neg_raw) == 1


@pytest.mark.parametrize(
    "header", ["-3", "nan", "inf", "soon", "Wed, 21 Oct 2026 07:28:00 GMT"]
)
def test_unreadable_retry_after_falls_back_to_the_backoff(
    vectors: dict, sleeps: list[float], header: str
) -> None:
    async def go(fake, base):
        fake.script = [("json", 503, UNAVAILABLE, {"Retry-After": header})]
        return await _write(LabelTransport(base, _headers), vectors)

    _run(vectors, go)
    assert sleeps == [0.5]


# ---------------------------------------------------- local validation before any send


@pytest.mark.parametrize(
    "over",
    [
        {"object_ref": HEBREW},
        {"object_ref": EMAIL},
        {"object_ref": "knobref_" + "A" * 32},
        {"object_ref": "widgetref_" + "a" * 32},
        {"purpose": HEBREW},
        {"purpose": "other"},
        {"base_revision": True},
        {"base_revision": -1},
        {"base_revision": 1.0},
        {"base_revision": 2**53},
        {"epoch": 0},
        {"epoch": True},
        {"epoch": 2},  # envelope.key_epoch (1) must equal epoch
    ],
)
def test_a_body_off_the_closed_contract_is_refused_locally_and_nothing_is_sent(
    vectors: dict, over: dict[str, Any]
) -> None:
    async def go(fake, base):
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors, **over)
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "invalid_request" and err.status is None
    assert fake.calls == [] and fake.neg_raw == [] and fake.legacy_raw == []
    rendered = "".join(traceback.format_exception(err)) + repr(err)
    assert not any(c in rendered for c in CANARIES)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update(name=HEBREW),
        lambda e: e.pop("enc"),
        lambda e: e.update(version=2),
        lambda e: e.update(version=True),
        lambda e: e.update(enc="short"),
        lambda e: e.update(ciphertext=HEBREW),
        lambda e: e.update(ciphertext="A" * 65),
        lambda e: e.update(ciphertext=EMAIL),
        lambda e: e.update(key_epoch="1"),
    ],
)
def test_an_envelope_off_the_closed_contract_is_refused_locally(
    vectors: dict, mutate
) -> None:
    envelope = _envelope(vectors)
    mutate(envelope)

    async def go(fake, base):
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, _headers), vectors, envelope=envelope)
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "invalid_request" and fake.calls == []


def test_a_non_dict_envelope_is_refused_locally(vectors: dict) -> None:
    async def go(fake, base):
        with pytest.raises(LabelTransportError):
            await _write(LabelTransport(base, _headers), vectors, envelope=[HEBREW])
        return None

    fake, _ = _run(vectors, go)
    assert fake.calls == []


def test_the_sent_body_is_a_private_copy_of_what_was_validated(vectors: dict) -> None:
    envelope = _envelope(vectors)
    body = tr._negotiated_body(KNOB_REF, "knob", 0, 1, envelope)
    envelope["ciphertext"] = HEBREW
    envelope["name"] = EMAIL
    assert HEBREW not in json.dumps(body) and EMAIL not in json.dumps(body)
    assert body["envelope"]["ciphertext"] == _envelope(vectors)["ciphertext"]


# ------------------------------------------------------------------- privacy canary


def test_names_never_leave_in_the_clear_on_either_route_or_in_logs(
    tmp_path: Path, vectors: dict, monkeypatch, caplog
) -> None:
    caplog.set_level(logging.DEBUG)
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        dataset_ref = "datasetref_" + "cd" * 16
        on = await send_session_labels(
            {"agent": HEBREW, "dataset": EMAIL},
            {"agent": AGENT_REF, "dataset": dataset_ref},
            base,
            _headers,
            cfg,
        )
        monkeypatch.setenv(ts.ENV_FLAG, "false")
        off_ref = "evaluatorref_" + "ef" * 16
        off = await send_session_labels(
            {"evaluator": f"{HEBREW} {EMAIL}"},
            {"evaluator": off_ref},
            base,
            _headers,
            cfg,
        )
        return on, off

    fake, (on, off) = _run(vectors, go)
    assert len(on) == 2 and len(off) == 1
    seen = [*fake.neg_raw, *fake.legacy_raw]
    assert len(seen) == 3
    for path, headers, raw in seen:
        text = raw.decode("utf-8") + path + json.dumps(headers, ensure_ascii=False)
        assert not any(c in text for c in CANARIES)
        assert not any(c in text for c in (HEBREW.encode().hex(), EMAIL.encode().hex()))
    assert not any(c in caplog.text for c in CANARIES)
    assert sorted(fake.plaintexts.values()) == sorted(
        [HEBREW, EMAIL, f"{HEBREW} {EMAIL}"]
    )


def test_server_text_and_names_are_absent_from_every_error(
    tmp_path: Path, vectors: dict, caplog
) -> None:
    caplog.set_level(logging.DEBUG)
    cfg = _config(tmp_path, vectors)
    hostile = {**_refusal("invalid_request"), "message": f"{HEBREW} {EMAIL}"}

    async def go(fake, base):
        errors = []
        for script in (
            [("json", 400, hostile, {})],
            [("json", 422, _refusal("reference_not_issued"), {})],
            [("404",)],
            [("text", 500, f"{HEBREW}{EMAIL}")],
            [("json", 503, UNAVAILABLE, {"Retry-After": "0"})] * 20,
        ):
            fake.script = list(script)
            sender = tr.sender_from_config(base, _headers, cfg)
            assert sender is not None
            try:
                await sender.send_new({AGENT_REF: ("label", HEBREW)})
            except LabelTransportError as exc:
                errors.append(exc)
        return errors

    async def no_sleep(_delay: float) -> None:
        return None

    import unittest.mock as mock

    with mock.patch.object(tr, "_sleep", no_sleep):
        fake, errors = _run(vectors, go)
    assert len(errors) == 5
    for err in errors:
        rendered = "".join(traceback.format_exception(err)) + repr(err) + str(err)
        assert not any(c in rendered for c in CANARIES)
        assert err.__cause__ is None and err.__context__ is None
    assert not any(c in caplog.text for c in CANARIES)


def test_the_gate_passes_a_sealed_write_and_refuses_a_plaintext_name(
    vectors: dict,
) -> None:
    """The content-class table knows the new route: sealed passes, a name is refused."""
    from traigent.labels import transport_gate as gate

    good = tr._negotiated_body(KNOB_REF, "knob", 0, 1, _envelope(vectors))
    url = "https://backend.example.test/api/v1/optimization/labels"
    gate.set_allowed_origins(["https://backend.example.test"])
    gate.set_mode("refuse")
    try:
        assert gate.inspect_outgoing("POST", url, json.dumps(good)) is not None
        for bad in (
            {**good, "object_ref": HEBREW},
            {**good, "purpose": HEBREW},
            {**good, "name": EMAIL},
            {**good, "schema_version": HEBREW},
        ):
            with pytest.raises(gate.TransportGateRefusal):
                gate.inspect_outgoing("POST", url, json.dumps(bad))
    finally:
        gate.set_mode(None)
        gate.set_allowed_origins(None)


# ------------------------------------------------------ S1 -> S2 and the conflict state


def _issue_one_token(kind: str = "knob", name: str = HEBREW) -> tuple[TokenStore, str]:
    namespace = TokenNamespace.for_backend(ORIGIN, "agentref_" + "a1" * 16)
    ref = f"{kind}ref_" + "5a" * 16

    def issue(body: dict[str, Any]) -> tuple[int, Any]:
        return 200, {"refs": [ref]}

    ensure_tokens(namespace, [(kind, None, name)], transport=issue)
    return TokenStore(namespace), ref


def _sender(base: str, tmp_path: Path, vectors: dict, store: TokenStore | None):
    sender = tr.sender_from_config(
        base, _headers, _config(tmp_path, vectors), token_store_=store
    )
    assert sender is not None
    return sender


def test_an_acknowledged_negotiated_write_moves_the_token_from_s1_to_s2(
    tmp_path: Path, vectors: dict
) -> None:
    store, ref = _issue_one_token()
    assert store.state_of(ref) == ts.S1_PENDING_LABEL

    async def go(fake, base):
        return await _sender(base, tmp_path, vectors, store).send_new(
            {ref: ("knob", HEBREW)}
        )

    fake, done = _run(vectors, go)
    assert done == {ref: 1}
    assert fake.plaintexts == {ref: HEBREW}
    assert store.state_of(ref) == ts.S2_LABEL_ACKED
    assert store.revision_of(ref) == 1
    # no S3 / spec commit here (1b-4)
    assert store.spec_commits(ref) == []


def test_flag_off_the_token_journal_is_not_touched(
    tmp_path: Path, vectors: dict, monkeypatch
) -> None:
    store, ref = _issue_one_token()
    monkeypatch.setenv(ts.ENV_FLAG, "false")

    async def go(fake, base):
        return await _sender(base, tmp_path, vectors, store).send_new(
            {ref: ("knob", HEBREW)}
        )

    fake, done = _run(vectors, go)
    assert done == {ref: 1} and fake.neg_raw == [] and len(fake.legacy_raw) == 1
    monkeypatch.setenv(ts.ENV_FLAG, "true")
    assert store.state_of(ref) == ts.S1_PENDING_LABEL


def test_stale_revision_without_an_approved_reader_is_label_conflict_unverified(
    tmp_path: Path, vectors: dict
) -> None:
    store, ref = _issue_one_token()

    async def go(fake, base):
        fake.script = [("json", 409, _refusal("stale_revision"), {})]
        with pytest.raises(LabelTransportError) as info:
            await _sender(base, tmp_path, vectors, store).send_new(
                {ref: ("knob", HEBREW)}
            )
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "label_conflict_unverified"
    assert store.state_of(ref) == ts.S_LABEL_CONFLICT
    assert store.conflicted_refs() == [ref]
    with pytest.raises(ts.TokenStoreError) as refusal:
        store.require_no_label_conflict()
    assert refusal.value.code == "label_conflict_unverified"
    assert fake.plaintexts == {}  # nothing was overwritten
    rendered = "".join(traceback.format_exception(err)) + repr(err)
    assert not any(c in rendered for c in CANARIES)
    assert err.__cause__ is None


def test_stale_revision_on_a_ref_that_is_not_a_token_stays_a_plain_stale_revision(
    tmp_path: Path, vectors: dict
) -> None:
    store, ref = _issue_one_token()

    async def go(fake, base):
        fake.script = [("json", 409, _refusal("stale_revision"), {})]
        with pytest.raises(LabelTransportError) as info:
            await _sender(base, tmp_path, vectors, store).send_new(
                {AGENT_REF: ("label", HEBREW)}
            )
        return info.value

    _fake, err = _run(vectors, go)
    assert err.code == "stale_revision"
    assert store.state_of(ref) == ts.S1_PENDING_LABEL  # untouched


def test_a_refusal_other_than_stale_revision_leaves_the_token_at_s1(
    tmp_path: Path, vectors: dict
) -> None:
    store, ref = _issue_one_token()

    async def go(fake, base):
        fake.script = [("json", 422, _refusal("reference_not_issued"), {})]
        with pytest.raises(LabelTransportError) as info:
            await _sender(base, tmp_path, vectors, store).send_new(
                {ref: ("knob", HEBREW)}
            )
        return info.value

    _fake, err = _run(vectors, go)
    assert err.code == "reference_not_issued"
    assert store.state_of(ref) == ts.S1_PENDING_LABEL


def test_a_session_label_ack_does_not_open_the_token_store(
    tmp_path: Path, vectors: dict
) -> None:
    store, _ref = _issue_one_token()
    before = sorted(p.name for p in (tmp_path / "state").rglob("*"))

    async def go(fake, base):
        return await _sender(base, tmp_path, vectors, store).send_new(
            {AGENT_REF: ("label", HEBREW)}
        )

    _fake, done = _run(vectors, go)
    assert done == {AGENT_REF: 1}
    assert before and store.revision_of(_ref) == 0  # S1 token unaffected


def test_a_redirect_is_refused_by_the_strict_gate_and_never_followed(
    vectors: dict,
) -> None:
    """The gate's own closed refusal (counts only) stands; nothing follows the redirect."""
    from traigent.labels.transport_gate import TransportGateRefusal

    async def go(fake, base):
        fake.script = [("json", 302, {}, {"Location": "https://elsewhere.invalid/x"})]
        with pytest.raises(TransportGateRefusal) as info:
            await _write(LabelTransport(base, _headers), vectors)
        return info.value

    fake, err = _run(vectors, go)
    assert len(fake.neg_raw) == 1 and fake.legacy_raw == []
    assert HEBREW not in str(err) and "elsewhere" not in str(err)


# ------------------------------------------------------------- round 1b-2b r2 (sol findings)


def test_an_existing_revision_on_an_s1_token_is_label_conflict_unverified_not_a_skip(
    tmp_path: Path, vectors: dict
) -> None:
    store, ref = _issue_one_token()

    async def go(fake, base):
        fake.revisions[ref] = (
            1  # a lost-ack write of ours, or another writer's: same picture
        )
        with pytest.raises(LabelTransportError) as info:
            await _sender(base, tmp_path, vectors, store).send_new(
                {ref: ("knob", HEBREW)}
            )
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "label_conflict_unverified"
    assert store.state_of(ref) == ts.S_LABEL_CONFLICT
    assert fake.neg_raw == [] and fake.plaintexts == {}  # nothing written
    rendered = "".join(traceback.format_exception(err)) + repr(err)
    assert not any(c in rendered for c in CANARIES)
    assert err.__cause__ is None and err.__context__ is None


def test_an_existing_revision_on_a_non_token_object_is_still_left_alone(
    tmp_path: Path, vectors: dict
) -> None:
    store, ref = _issue_one_token()

    async def go(fake, base):
        fake.revisions[AGENT_REF] = 3
        return await _sender(base, tmp_path, vectors, store).send_new(
            {AGENT_REF: ("label", HEBREW)}
        )

    fake, done = _run(vectors, go)
    assert done == {} and fake.neg_raw == []
    assert store.state_of(ref) == ts.S1_PENDING_LABEL


def _budget(monkeypatch, seconds: float) -> None:
    monkeypatch.setattr(tr, "_WRITE_DEADLINE_SECONDS", seconds)


def test_a_stalled_headers_provider_is_bounded_by_the_operation_budget(
    vectors: dict, monkeypatch
) -> None:
    _budget(monkeypatch, 0.3)

    async def stalled() -> dict[str, str]:
        await asyncio.sleep(60)
        return {}

    async def go(fake, base):
        started = time.monotonic()
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, stalled), vectors)
        return info.value, time.monotonic() - started

    fake, (err, took) = _run(vectors, go)
    assert err.code == "service_unavailable" and err.status == 503
    assert took < 5 and fake.neg_raw == []  # nothing was sent


def test_a_slow_headers_provider_leaves_the_http_call_only_what_is_left(
    vectors: dict, monkeypatch
) -> None:
    import aiohttp

    _budget(monkeypatch, 2.0)
    totals: list[float] = []
    real = aiohttp.ClientTimeout

    def spy(*args: Any, **kwargs: Any):
        totals.append(kwargs["total"])
        return real(*args, **kwargs)

    monkeypatch.setattr(aiohttp, "ClientTimeout", spy)

    async def slow() -> dict[str, str]:
        await asyncio.sleep(0.8)
        return await _headers()

    async def go(fake, base):
        return await _write(LabelTransport(base, slow), vectors)

    _fake, ack = _run(vectors, go)
    assert ack["revision"] == 1
    assert (
        len(totals) == 1 and totals[0] <= 2.0 - 0.8 + 0.05
    )  # remaining, not the original


def test_a_provider_that_spends_the_whole_budget_sends_nothing(
    vectors: dict, monkeypatch
) -> None:
    _budget(monkeypatch, 0.3)

    async def slow() -> dict[str, str]:
        await asyncio.sleep(0.2)
        time.sleep(0.2)  # blocks past the budget without being cancellable
        return await _headers()

    async def go(fake, base):
        with pytest.raises(LabelTransportError) as info:
            await _write(LabelTransport(base, slow), vectors)
        return info.value

    fake, err = _run(vectors, go)
    assert err.code == "service_unavailable" and fake.neg_raw == []


def _chain_text(err: BaseException) -> str:
    return (
        "".join(traceback.format_exception(err))
        + repr(err)
        + str(err)
        + repr(err.__cause__)
        + repr(err.__context__)
    )


@pytest.mark.parametrize("where", ["local_validation", "headers_provider", "client"])
def test_negotiated_errors_clear_cause_and_context_of_the_callers_handled_exception(
    vectors: dict, monkeypatch, where: str
) -> None:
    import aiohttp

    async def leaky_headers() -> dict[str, str]:
        raise RuntimeError(f"auth failed for {EMAIL}")

    class Refusing(aiohttp.ClientConnectionError):
        pass

    async def go(fake, base):
        headers = leaky_headers if where == "headers_provider" else _headers
        over: dict[str, Any] = (
            {"object_ref": f"not-a-ref {EMAIL}"} if where == "local_validation" else {}
        )
        if where == "client":
            monkeypatch.setattr(
                aiohttp.ClientSession,
                "request",
                lambda *a, **k: (_ for _ in ()).throw(Refusing(EMAIL)),
            )
        try:
            raise ValueError(EMAIL)  # the caller is handling an email-shaped error
        except ValueError:
            try:
                await _write(LabelTransport(base, headers), vectors, **over)
            except LabelTransportError as exc:
                return exc
        raise AssertionError("no error")

    _fake, err = _run(vectors, go)
    assert err.__cause__ is None and err.__context__ is None
    assert err.__suppress_context__ is True
    assert EMAIL not in _chain_text(err)


@pytest.mark.parametrize(
    "ack_over",
    [
        {"epoch": True},  # bool is not an int, even though True == 1
        {"updated_at": "x"},
        {"updated_at": "2026-10-19 00:00:02"},
        {"updated_at": "2026-13-45T99:00:02Z"},
        {"updated_at": "2026-10-19T00:00:02+02:00"},
        {"updated_at": 1760832002},
    ],
)
def test_ack_epoch_must_be_a_true_int_and_updated_at_a_utc_timestamp(
    tmp_path: Path, vectors: dict, ack_over: dict[str, Any]
) -> None:
    store, ref = _issue_one_token()

    async def go(fake, base):
        fake.script = [
            (
                "json",
                200,
                {
                    "object_ref": ref,
                    "purpose": "knob",
                    "revision": 1,
                    "epoch": 1,
                    "updated_at": "2026-10-19T00:00:02Z",
                    **ack_over,
                },
                {},
            )
        ]
        with pytest.raises(LabelTransportError) as info:
            await _sender(base, tmp_path, vectors, store).send_new(
                {ref: ("knob", HEBREW)}
            )
        return info.value

    _fake, err = _run(vectors, go)
    assert err.code == "internal_error"
    assert store.state_of(ref) == ts.S1_PENDING_LABEL  # no S2 on a bad ack


# ------------------------------------------- legacy (flag off) timeout / error contract


def test_flag_off_a_slow_provider_is_awaited_unbounded_and_the_http_budget_is_full(
    vectors: dict, monkeypatch
) -> None:
    """Pins the legacy ``_call`` contract of 980e72515: the headers provider is NOT bounded
    and the HTTP call still gets the whole ``_TIMEOUT_SECONDS`` (not what is left)."""
    import aiohttp

    monkeypatch.setenv(ts.ENV_FLAG, "false")
    monkeypatch.setattr(tr, "_TIMEOUT_SECONDS", 0.3)
    totals: list[float] = []
    real = aiohttp.ClientTimeout

    def spy(*args: Any, **kwargs: Any):
        totals.append(kwargs["total"])
        return real(*args, **kwargs)

    monkeypatch.setattr(aiohttp, "ClientTimeout", spy)

    async def slow() -> dict[str, str]:
        await asyncio.sleep(0.6)  # longer than the whole HTTP budget
        return await _headers()

    async def go(fake, base):
        return await _write(
            LabelTransport(base, slow), vectors, object_ref=VECTOR_REF, purpose="label"
        )

    fake, ack = _run(vectors, go)
    assert ack["revision"] == 1 and len(fake.legacy_raw) == 1 and fake.neg_raw == []
    assert totals == [0.3]  # full budget, not reduced by the provider's time


def test_flag_off_a_provider_failure_escapes_unchanged_and_nothing_is_sent(
    vectors: dict, monkeypatch
) -> None:
    monkeypatch.setenv(ts.ENV_FLAG, "false")

    async def failing() -> dict[str, str]:
        raise TimeoutError("provider")

    async def go(fake, base):
        with pytest.raises(TimeoutError) as info:  # not wrapped, as at 980e72515
            await _write(LabelTransport(base, failing), vectors)
        return info.value

    fake, err = _run(vectors, go)
    assert not isinstance(err, LabelTransportError)
    assert fake.legacy_raw == [] and fake.neg_raw == []


# ------------------------------- production entry wires the journal (S1 -> S2), flag on/off


def _issue_for(base: str, agent_ref: str, name: str = HEBREW) -> tuple[TokenStore, str]:
    namespace = TokenNamespace.for_backend(base, agent_ref)
    ref = "knobref_" + "5a" * 16
    ensure_tokens(
        namespace, [("knob", None, name)], transport=lambda body: (200, {"refs": [ref]})
    )
    return TokenStore(namespace), ref


def test_the_production_entry_moves_an_issued_token_from_s1_to_s2(
    tmp_path: Path, vectors: dict
) -> None:
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        store, ref = _issue_for(base, AGENT_REF)
        assert store.state_of(ref) == ts.S1_PENDING_LABEL
        done = await send_session_labels(
            {"knob": HEBREW}, {"knob": ref, "agent": AGENT_REF}, base, _headers, cfg
        )
        return store, ref, done

    fake, (store, ref, done) = _run(vectors, go)
    assert done == {ref: 1}
    assert fake.plaintexts == {ref: HEBREW}
    assert store.state_of(ref) == ts.S2_LABEL_ACKED
    assert store.revision_of(ref) == 1


def test_the_production_entry_with_the_flag_off_never_touches_the_store(
    tmp_path: Path, vectors: dict, monkeypatch
) -> None:
    cfg = _config(tmp_path, vectors)
    created: list[Any] = []
    real = ts.TokenStore.__init__

    def spy(self, namespace):
        created.append(namespace)
        real(self, namespace)

    async def go(fake, base):
        store, ref = _issue_for(base, AGENT_REF)
        monkeypatch.setenv(ts.ENV_FLAG, "false")
        before = sorted(
            (p.name, p.stat().st_mtime_ns)
            for p in (tmp_path / "state" / "token_store").rglob("*")
        )
        monkeypatch.setattr(ts.TokenStore, "__init__", spy)
        done = await send_session_labels(
            {"knob": HEBREW}, {"knob": ref, "agent": AGENT_REF}, base, _headers, cfg
        )
        monkeypatch.setattr(ts.TokenStore, "__init__", real)
        after = sorted(
            (p.name, p.stat().st_mtime_ns)
            for p in (tmp_path / "state" / "token_store").rglob("*")
        )
        monkeypatch.setenv(ts.ENV_FLAG, "true")
        return store, ref, done, before, after

    fake, (store, ref, done, before, after) = _run(vectors, go)
    assert done == {ref: 1} and fake.neg_raw == [] and len(fake.legacy_raw) == 1
    assert created == []  # no store object was even built
    assert before == after  # no file created or changed
    assert store.state_of(ref) == ts.S1_PENDING_LABEL


def test_the_production_entry_without_an_agent_reference_opens_no_store(
    tmp_path: Path, vectors: dict
) -> None:
    assert tr._token_store_for("https://x.test/api/v1", {"knob": KNOB_REF}) is None
    assert tr._token_store_for("not a url", {"agent": AGENT_REF}) is None


# ------------- explicit issuance store; unknown token ref fails closed (round 1b-2b r4)

OTHER_AGENT_REF = "agentref_" + "b2" * 16


def test_a_token_issued_under_another_agent_is_token_ref_unknown_and_nothing_is_sent(
    tmp_path: Path, vectors: dict, caplog
) -> None:
    caplog.set_level(logging.DEBUG)
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        store, ref = _issue_for(base, AGENT_REF)
        done = await send_session_labels(
            {"knob": HEBREW},
            {"knob": ref, "agent": OTHER_AGENT_REF},
            base,
            _headers,
            cfg,
        )
        sender = tr.sender_from_config(
            base,
            _headers,
            cfg,
            token_store_=TokenStore(TokenNamespace.for_backend(base, OTHER_AGENT_REF)),
        )
        with pytest.raises(LabelTransportError) as info:
            await sender.send_new({ref: ("knob", HEBREW)})
        return store, ref, done, info.value

    fake, (store, ref, done, err) = _run(vectors, go)
    assert done == {}
    assert err.code == "token_ref_unknown"
    assert fake.neg_raw == [] and fake.legacy_raw == [] and fake.plaintexts == {}
    assert store.state_of(ref) == ts.S1_PENDING_LABEL
    rendered = "".join(traceback.format_exception(err)) + repr(err) + caplog.text
    for secret in (*CANARIES, ref, OTHER_AGENT_REF, AGENT_REF):
        assert secret not in rendered


def test_an_explicit_store_wins_over_the_session_agent_and_advances_s1_to_s2(
    tmp_path: Path, vectors: dict
) -> None:
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        store, ref = _issue_for(base, AGENT_REF)
        done = await send_session_labels(
            {"knob": HEBREW},
            {"knob": ref, "agent": OTHER_AGENT_REF},
            base,
            _headers,
            cfg,
            token_store_=store,
        )
        return store, ref, done

    fake, (store, ref, done) = _run(vectors, go)
    assert done == {ref: 1}
    assert store.state_of(ref) == ts.S2_LABEL_ACKED


def test_flag_off_a_token_ref_unknown_to_the_session_store_is_the_legacy_write(
    tmp_path: Path, vectors: dict, monkeypatch
) -> None:
    cfg = _config(tmp_path, vectors)

    async def go(fake, base):
        store, ref = _issue_for(base, AGENT_REF)
        monkeypatch.setenv(ts.ENV_FLAG, "false")
        done = await send_session_labels(
            {"knob": HEBREW},
            {"knob": ref, "agent": OTHER_AGENT_REF},
            base,
            _headers,
            cfg,
        )
        monkeypatch.setenv(ts.ENV_FLAG, "true")
        return store, ref, done

    fake, (store, ref, done) = _run(vectors, go)
    assert done == {ref: 1} and fake.neg_raw == [] and len(fake.legacy_raw) == 1
    assert store.state_of(ref) == ts.S1_PENDING_LABEL
