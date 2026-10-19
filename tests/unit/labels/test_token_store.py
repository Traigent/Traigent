"""Durable typed token store + issuance journal (content-less round 1b-2a, SDK half).

Contract: PHASE1b-DESIGN-v2 sections 3.2, 3.3, 10.1', 10.2, 10.5-10.7 and the Schema
``optimization/token_refs_schema.json``. The fake server below implements the Backend's
10.1' replay rule (same key + same agent + same ordered items -> same refs; anything else ->
``operation_key_conflict``), so a resume is tested against the real semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import multiprocessing
import os
import secrets
import stat
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

from traigent.labels import token_store as ts
from traigent.labels.token_store import (
    TokenNamespace,
    TokenStore,
    TokenStoreError,
    ensure_tokens,
)

AGENT_A = "agentref_" + "a1" * 16
AGENT_B = "agentref_" + "b2" * 16
ORIGIN = "https://api.traigent.test"

HEBREW = "שלום_עולם_טמפרטורה"
EMAIL = "alice.smith+secret@example.com"
CANARIES = (HEBREW, EMAIL)


class SimulatedCrash(BaseException):  # noqa: N818 - a process death, not an Exception
    """Stands in for the process dying at a boundary (not an ``Exception`` on purpose)."""


# ------------------------------------------------------------------ fake server (10.1')


class FakeServer:
    """Backend issuance semantics, in memory. ``calls`` records every request body."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.receipts: dict[
            str, tuple[str, str]
        ] = {}  # key -> (agent, canonical items)
        self.refs: dict[str, list[str]] = {}
        self.knobs: dict[str, str] = {}  # knob ref -> agent
        self.issued = 0
        self.status_override: tuple[int, Any] | None = None

    @staticmethod
    def _closed(code: str, message: str, status: int) -> tuple[int, dict[str, Any]]:
        return status, {
            "success": False,
            "error": code,
            "error_code": code,
            "message": message,
        }

    def __call__(self, body: dict[str, Any]) -> tuple[int, Any]:
        self.calls.append(copy.deepcopy(body))
        if self.status_override is not None:
            return self.status_override
        items = body["items"]
        canonical = json.dumps(items, sort_keys=True)
        key, agent = body["operation_key"], body["agent_ref"]
        if key in self.receipts:
            if self.receipts[key] != (agent, canonical):
                return self._closed(
                    "operation_key_conflict",
                    "Operation key conflicts with an earlier request.",
                    409,
                )
            return 200, {"refs": list(self.refs[key])}
        for item in items:
            if item["kind"] == "choice" and self.knobs.get(item["parent_ref"]) != agent:
                return self._closed("reference_not_found", "Reference not found.", 404)
        refs = [f"{item['kind']}ref_{secrets.token_hex(16)}" for item in items]
        self.receipts[key] = (agent, canonical)
        self.refs[key] = refs
        self.issued += len(refs)
        for item, ref in zip(items, refs, strict=True):
            if item["kind"] == "knob":
                self.knobs[ref] = agent
        return 200, {"refs": refs}


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "state"))
    monkeypatch.setenv(ts.ENV_FLAG, "true")


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


def ns(agent: str = AGENT_A, origin: str = ORIGIN) -> TokenNamespace:
    return TokenNamespace.for_backend(origin, agent)


def knobs(*names: str) -> list[tuple[str, None, str]]:
    return [("knob", None, n) for n in names]


def store_file(namespace: TokenNamespace) -> Path:
    return ts.store_path(namespace)


# --------------------------------------------------------------- 10.2 namespace scoping


def test_same_name_under_two_agents_is_two_tokens(server):
    a = ensure_tokens(ns(AGENT_A), knobs("temperature"), transport=server)
    b = ensure_tokens(ns(AGENT_B), knobs("temperature"), transport=server)
    assert a[("knob", None, "temperature")] != b[("knob", None, "temperature")]
    assert store_file(ns(AGENT_A)) != store_file(ns(AGENT_B))


def test_same_choice_name_under_two_knobs_is_two_tokens(server):
    ensure_tokens(ns(), knobs("model", "tone"), transport=server)
    got = ensure_tokens(
        ns(),
        [("choice", "model", "low"), ("choice", "tone", "low")],
        transport=server,
    )
    assert got[("choice", "model", "low")] != got[("choice", "tone", "low")]
    # and each is bound to its own parent knob on the wire
    parents = [i["parent_ref"] for i in server.calls[-1]["items"]]
    assert len(set(parents)) == 2


def test_same_name_same_agent_two_specs_is_one_token_and_no_second_call(server):
    first = ensure_tokens(ns(), knobs("temperature"), transport=server)
    n_calls = len(server.calls)
    second = ensure_tokens(ns(), knobs("temperature"), transport=server)  # a later spec
    assert second == first
    assert len(server.calls) == n_calls
    assert server.issued == 1


def test_two_backend_origins_are_two_stores(server):
    a = ensure_tokens(ns(origin="https://api.one.test"), knobs("k"), transport=server)
    b = ensure_tokens(ns(origin="https://api.two.test"), knobs("k"), transport=server)
    assert a != b
    assert store_file(ns(origin="https://api.one.test")) != store_file(
        ns(origin="https://api.two.test")
    )
    assert store_file(ns(origin="https://api.one.test")).exists()
    assert store_file(ns(origin="https://api.two.test")).exists()


@pytest.mark.parametrize(
    "url",
    [
        "HTTPS://API.Traigent.TEST:443/api/v1",
        "https://user:pw@api.traigent.test/x?y=1#z",
        "https://api.traigent.test/",
    ],
)
def test_origin_normalization_is_one_namespace(url):
    assert ts.normalize_origin(url) == "https://api.traigent.test"
    assert ns(origin=url).digest == ns().digest


def test_origin_normalization_keeps_nondefault_port_and_distinguishes_scheme():
    assert ts.normalize_origin("http://Host:8080/p") == "http://host:8080"
    assert ts.normalize_origin("http://host:80") == "http://host"
    assert ts.normalize_origin("https://host:80") == "https://host:80"
    assert ts.normalize_origin("https://host") != ts.normalize_origin("http://host")


@pytest.mark.parametrize("bad", ["", "ftp://x", "host-only", "https://", None, 5])
def test_bad_origin_is_refused_without_echo(bad):
    with pytest.raises(TokenStoreError) as err:
        ts.normalize_origin(bad)  # type: ignore[arg-type]
    assert err.value.code == "token_namespace_invalid"


def test_store_filename_is_a_hash_not_the_agent_or_origin(server):
    ensure_tokens(ns(), knobs("k"), transport=server)
    name = store_file(ns()).name
    assert (
        name
        == hashlib.sha256(
            json.dumps(
                ["traigent-token-store-ns-v1", ORIGIN, AGENT_A], separators=(",", ":")
            ).encode()
        ).hexdigest()
        + ".json"
    )
    assert AGENT_A not in str(store_file(ns())) and "traigent.test" not in name


# ----------------------------------------------------- kinds are distinct identities


def test_same_name_across_kinds_is_distinct(server):
    got = ensure_tokens(
        ns(),
        [("knob", None, "low"), ("metric", None, "low"), ("template", None, "low")],
        transport=server,
    )
    refs = list(got.values())
    assert len(set(refs)) == 3
    assert got[("knob", None, "low")].startswith("knobref_")
    assert got[("metric", None, "low")].startswith("metricref_")
    assert got[("template", None, "low")].startswith("templateref_")


def test_choice_requires_a_parent_knob_already_in_the_store(server):
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), [("choice", "model", "gpt")], transport=server)
    assert err.value.code == "token_parent_missing"
    assert server.calls == []  # refused locally, nothing journalled or sent
    # a knob issued in the same call does not count: the server needs an EARLIER request
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(
            ns(),
            [("knob", None, "model"), ("choice", "model", "gpt")],
            transport=server,
        )
    assert err.value.code == "token_parent_missing"
    ensure_tokens(ns(), knobs("model"), transport=server)
    got = ensure_tokens(ns(), [("choice", "model", "gpt")], transport=server)
    assert got[("choice", "model", "gpt")].startswith("choiceref_")


@pytest.mark.parametrize(
    "bad",
    [
        [("knob", None, "")],
        [("knob", "parent", "x")],
        [("choice", None, "x")],
        [("bogus", None, "x")],
        [("knob", None, 5)],
        [("knob", None, "x" * 5000)],
        [("knob", None, "lone\ud800surrogate")],
        [("knob", None)],
        "knob",
    ],
)
def test_invalid_requests_are_refused_before_anything_happens(server, bad):
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), bad, transport=server)  # type: ignore[arg-type]
    assert err.value.code == "token_request_invalid"
    assert server.calls == [] and not store_file(ns()).exists()


# ------------------------------------------------- file mode, version, corruption


def test_store_is_created_0600_in_a_0700_directory(server):
    ensure_tokens(ns(), knobs("k"), transport=server)
    path = store_file(ns())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.with_suffix(".lock").stat().st_mode) == 0o600
    assert [p.name for p in path.parent.iterdir() if ".tmp" in p.name] == []


def test_store_is_0600_even_under_a_permissive_umask(server):
    old = os.umask(0)
    try:
        ensure_tokens(ns(), knobs("k"), transport=server)
    finally:
        os.umask(old)
    assert stat.S_IMODE(store_file(ns()).stat().st_mode) == 0o600
    assert stat.S_IMODE(store_file(ns()).parent.stat().st_mode) == 0o700


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o666, 0o604, 0o400, 0o700])
def test_wrong_file_mode_on_open_is_refused_and_not_repaired(server, mode):
    ensure_tokens(ns(), knobs("k"), transport=server)
    path = store_file(ns())
    os.chmod(path, mode)
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("k"), transport=server)
    assert err.value.code == "token_store_insecure_permissions"
    assert stat.S_IMODE(path.stat().st_mode) == mode  # no chmod repair


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o770, 0o707])
def test_wrong_directory_mode_is_refused_and_not_repaired(server, mode):
    ensure_tokens(ns(), knobs("k"), transport=server)
    directory = store_file(ns()).parent
    os.chmod(directory, mode)
    try:
        with pytest.raises(TokenStoreError) as err:
            ensure_tokens(ns(), knobs("k"), transport=server)
        assert err.value.code == "token_store_insecure_permissions"
        assert stat.S_IMODE(directory.stat().st_mode) == mode
    finally:
        os.chmod(directory, 0o700)


def test_symlinked_store_file_is_refused(server, tmp_path):
    ensure_tokens(ns(), knobs("k"), transport=server)
    path = store_file(ns())
    target = tmp_path / "elsewhere"
    target.write_bytes(path.read_bytes())
    os.chmod(target, 0o600)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("k"), transport=server)
    assert err.value.code == "token_map_not_persisted"


def _good_state(server) -> tuple[bytes, dict[str, Any]]:
    ensure_tokens(ns(), knobs("alpha") + [("metric", None, "acc")], transport=server)
    raw = store_file(ns()).read_bytes()
    return raw, json.loads(raw[1:].decode())


def _put(raw: bytes) -> None:
    path = store_file(ns())
    path.write_bytes(raw)
    os.chmod(path, 0o600)


def _enc(state: Any) -> bytes:
    return bytes([1]) + json.dumps(state, ensure_ascii=False).encode()


def _mutations(state: dict[str, Any]) -> dict[str, bytes]:
    def edit(fn):
        s = copy.deepcopy(state)
        fn(s)
        return _enc(s)

    first_key = sorted(state["tokens"])[0]
    first_op = next(iter(state["operations"]))
    return {
        "bad_json": bytes([1]) + b"{not json",
        "empty_object": _enc({}),
        "not_an_object": _enc([1, 2]),
        "empty_file": b"",
        "version_only": bytes([1]),
        "no_version_byte": json.dumps(state).encode(),
        "invalid_utf8": bytes([1]) + b"\xff\xfe\x00",
        "extra_top_key": edit(lambda s: s.update(extra=1)),
        "missing_top_key": edit(lambda s: s.pop("revisions")),
        "wrong_namespace": edit(lambda s: s.update(namespace="0" * 64)),
        "token_missing_field": edit(lambda s: s["tokens"][first_key].pop("name")),
        "token_extra_field": edit(lambda s: s["tokens"][first_key].update(x=1)),
        "token_bad_kind": edit(lambda s: s["tokens"][first_key].update(kind="bogus")),
        "token_bad_ref": edit(
            lambda s: s["tokens"][first_key].update(ref="knobref_zz")
        ),
        "token_wrong_ref_for_kind": edit(
            lambda s: s["tokens"][first_key].update(ref="metricref_" + "0" * 32)
        ),
        "token_key_mismatch": edit(
            lambda s: s["tokens"].update(
                {'["knob",null,"zzz"]': s["tokens"].pop(first_key)}
            )
        ),
        "token_bad_state": edit(lambda s: s["tokens"][first_key].update(state="weird")),
        "token_dangling_op": edit(lambda s: s["tokens"][first_key].update(op="f" * 32)),
        "token_bad_ordinal": edit(lambda s: s["tokens"][first_key].update(ordinal=99)),
        "token_ordinal_bool": edit(
            lambda s: s["tokens"][first_key].update(ordinal=True)
        ),
        "token_name_not_string": edit(lambda s: s["tokens"][first_key].update(name=5)),
        "partial_batch_missing_token": edit(lambda s: s["tokens"].pop(first_key)),
        "op_refs_wrong_length": edit(
            lambda s: s["operations"][first_op].update(refs=[])
        ),
        "op_digest_mismatch": edit(
            lambda s: s["operations"][first_op].update(digest="0" * 64)
        ),
        "op_items_tampered": edit(
            lambda s: s["operations"][first_op]["items"][0].__setitem__(2, "other")
        ),
        "op_bad_key": edit(
            lambda s: s["operations"].update({"nothex": s["operations"].pop(first_op)})
        ),
        "revision_for_unknown_ref": edit(
            lambda s: s["revisions"].update({"knobref_" + "9" * 32: 1})
        ),
        "revision_zero": edit(
            lambda s: s["revisions"].update({s["tokens"][first_key]["ref"]: 0})
        ),
        "acked_without_revision": edit(
            lambda s: s["tokens"][first_key].update(state="label_acked")
        ),
        "intent_with_ref_but_op_allocated": edit(
            lambda s: s["tokens"][first_key].update(state="intent")
        ),
        "spec_without_state": edit(
            lambda s: s["tokens"][first_key].update(spec="searchspecref_" + "1" * 32)
        ),
    }


def test_every_corrupt_shape_is_refused_whole_and_never_returns_a_partial_map(server):
    _, state = _good_state(server)
    mutations = _mutations(state)
    for label, blob in mutations.items():
        _put(blob)
        calls = len(server.calls)
        with pytest.raises(TokenStoreError) as err:
            ensure_tokens(ns(), knobs("alpha"), transport=server)
        assert err.value.code == "token_map_not_persisted", label
        assert len(server.calls) == calls, label  # no request on a corrupt store
        for call in (
            lambda: TokenStore(ns()).lookup("knob", None, "alpha"),
            lambda: TokenStore(ns()).conflicted_refs(),
        ):
            with pytest.raises(TokenStoreError) as err2:
                call()
            assert err2.value.code == "token_map_not_persisted", label


def test_truncation_at_every_length_is_refused_or_complete(server):
    raw, _ = _good_state(server)
    for cut in range(0, len(raw)):
        _put(raw[:cut])
        with pytest.raises(TokenStoreError) as err:
            ensure_tokens(ns(), knobs("alpha"), transport=server)
        assert err.value.code in {
            "token_map_not_persisted",
            "token_store_version_unsupported",
        }, cut
    _put(raw)  # the intact file works again
    assert ensure_tokens(ns(), knobs("alpha"), transport=server)


@pytest.mark.parametrize("version", [2, 3, 0x1F])
def test_unknown_version_is_refused_distinctly(server, version):
    raw, _ = _good_state(server)
    _put(bytes([version]) + raw[1:])
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("alpha"), transport=server)
    assert err.value.code == "token_store_version_unsupported"


@pytest.mark.parametrize("byte", [0x00, 0x20, ord("{"), 0xFF])
def test_non_version_leading_byte_is_corruption_not_a_version(server, byte):
    raw, _ = _good_state(server)
    _put(bytes([byte]) + raw[1:])
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("alpha"), transport=server)
    assert err.value.code == "token_map_not_persisted"


def test_stray_temp_file_from_a_torn_write_is_harmless(server):
    ensure_tokens(ns(), knobs("a"), transport=server)
    path = store_file(ns())
    stray = path.with_name(path.name + ".tmp99999.deadbeef")
    stray.write_bytes(b"torn")
    assert ensure_tokens(ns(), knobs("a", "b"), transport=server)
    assert stray.exists()


# ------------------------------------------------------------ the flag (off = inert)


@pytest.mark.parametrize(
    "value", ["", "false", "0", "no", "yes", "on", "TRUE-ish", "2"]
)
def test_flag_off_means_no_file_no_dir_no_request(monkeypatch, tmp_path, server, value):
    monkeypatch.setenv(ts.ENV_FLAG, value)
    assert ts.is_enabled() is False
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("k"), transport=server)
    assert err.value.code == "token_store_disabled"
    assert server.calls == []
    assert not (tmp_path / "state").exists()
    with pytest.raises(TokenStoreError):
        TokenStore(ns()).conflicted_refs()
    assert not (tmp_path / "state").exists()


def test_flag_unset_means_off(monkeypatch, tmp_path, server):
    monkeypatch.delenv(ts.ENV_FLAG)
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("k"), transport=server)
    assert err.value.code == "token_store_disabled"
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("value", ["true", "1", " True ", "TRUE"])
def test_flag_on_values(monkeypatch, value, server):
    monkeypatch.setenv(ts.ENV_FLAG, value)
    assert ts.is_enabled() is True
    assert ensure_tokens(ns(), knobs("k"), transport=server)


def test_flag_is_read_at_call_time(monkeypatch, server):
    ensure_tokens(ns(), knobs("k"), transport=server)
    monkeypatch.setenv(ts.ENV_FLAG, "false")
    with pytest.raises(TokenStoreError):
        ensure_tokens(ns(), knobs("k"), transport=server)
    monkeypatch.setenv(ts.ENV_FLAG, "true")
    assert ensure_tokens(ns(), knobs("k"), transport=server)


def test_flag_off_never_reaches_the_http_layer(monkeypatch, tmp_path):
    import requests

    def boom(*a, **k):
        raise AssertionError("HTTP layer reached with the token store disabled")

    monkeypatch.setattr(requests.Session, "request", boom)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", boom)
    monkeypatch.setenv(ts.ENV_FLAG, "false")

    def transport(body):
        boom()

    with pytest.raises(TokenStoreError):
        ensure_tokens(ns(), knobs("k"), transport=transport)
    assert not (tmp_path / "state").exists()


def test_importing_the_module_has_no_side_effects(tmp_path):
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != ts.ENV_FLAG}
    env["TRAIGENT_RESULTS_FOLDER"] = str(tmp_path / "fresh")
    subprocess.run(
        [sys.executable, "-I", "-c", "import traigent.labels.token_store"],
        check=True,
        env=env,
        timeout=120,
    )
    assert not (tmp_path / "fresh").exists()


# -------------------------------------------------- crash injection at every boundary


class WriteCrasher:
    """Raise ``SimulatedCrash`` at the Nth durable write (before or after it lands)."""

    def __init__(self, monkeypatch, nth: int, *, after: bool = False) -> None:
        self.real = ts._write_atomic
        self.n = 0
        self.nth, self.after = nth, after
        self.monkeypatch = monkeypatch
        monkeypatch.setattr(ts, "_write_atomic", self)

    def disarm(self) -> None:
        self.monkeypatch.setattr(ts, "_write_atomic", self.real)

    def __call__(self, path, payload):
        self.n += 1
        if self.n == self.nth and not self.after:
            raise SimulatedCrash
        self.real(path, payload)
        if self.n == self.nth and self.after:
            raise SimulatedCrash


def crash_transport(server: FakeServer, *, before: bool = False, after: bool = False):
    """A transport that dies before sending, or after the server answered (reply lost)."""

    def transport(body):
        if before:
            raise SimulatedCrash
        result = server(body)
        if after:
            raise SimulatedCrash
        return result

    return transport


ITEMS = knobs("alpha", "beta") + [("metric", None, "accuracy")]


def _only_one_identity_each(server: FakeServer, n_items: int) -> None:
    assert server.issued == n_items  # no orphan, no duplicate identity on the server
    assert len(server.receipts) == 1


def test_crash_before_s0_is_durable_leaves_nothing_and_a_clean_retry_issues_once(
    server, monkeypatch
):
    crasher = WriteCrasher(monkeypatch, 1)
    with pytest.raises(SimulatedCrash):
        ensure_tokens(ns(), ITEMS, transport=server)
    assert server.calls == []  # S0 is durable BEFORE any request: no request without it
    crasher.disarm()
    got = ensure_tokens(ns(), ITEMS, transport=server)
    assert len(got) == 3
    _only_one_identity_each(server, 3)


def test_crash_after_s0_before_the_request_resends_the_same_key_and_batch(server):
    with pytest.raises(SimulatedCrash):
        ensure_tokens(ns(), ITEMS, transport=crash_transport(server, before=True))
    assert server.calls == []
    state = json.loads(store_file(ns()).read_bytes()[1:])
    ((op_key, op),) = state["operations"].items()
    assert op["refs"] is None and len(op["items"]) == 3
    assert {e["state"] for e in state["tokens"].values()} == {"intent"}
    got = ensure_tokens(ns(), ITEMS, transport=server)
    assert [c["operation_key"] for c in server.calls] == [op_key]  # the SAME key
    assert [(i["kind"]) for i in server.calls[0]["items"]] == ["knob", "knob", "metric"]
    assert len(got) == 3
    _only_one_identity_each(server, 3)


def test_crash_after_response_before_s1_resumes_to_the_same_refs(server, monkeypatch):
    # the server answered and issued, the reply was lost with the process
    with pytest.raises(SimulatedCrash):
        ensure_tokens(ns(), ITEMS, transport=crash_transport(server, after=True))
    issued = list(server.refs.values())[0]
    state = json.loads(store_file(ns()).read_bytes()[1:])
    assert {e["state"] for e in state["tokens"].values()} == {"intent"}
    got = ensure_tokens(ns(), ITEMS, transport=server)
    assert len(server.calls) == 2
    assert (
        server.calls[0] == server.calls[1]
    )  # same key, same ordered batch, byte for byte
    assert sorted(got.values()) == sorted(issued)
    assert (
        got[("knob", None, "alpha")] == issued[0]
    )  # ordinal -> ref, in the original order
    _only_one_identity_each(server, 3)


def test_crash_while_persisting_s1_resumes_to_the_same_refs(server, monkeypatch):
    crasher = WriteCrasher(monkeypatch, 2)  # write #1 = S0, write #2 = S1
    with pytest.raises(SimulatedCrash):
        ensure_tokens(ns(), ITEMS, transport=server)
    issued = list(server.refs.values())[0]
    crasher.disarm()
    got = ensure_tokens(ns(), ITEMS, transport=server)
    assert sorted(got.values()) == sorted(issued)
    _only_one_identity_each(server, 3)


def test_crash_right_after_s1_lands_needs_no_second_request(server, monkeypatch):
    crasher = WriteCrasher(monkeypatch, 2, after=True)
    with pytest.raises(SimulatedCrash):
        ensure_tokens(ns(), ITEMS, transport=server)
    crasher.disarm()
    got = ensure_tokens(ns(), ITEMS, transport=server)
    assert len(server.calls) == 1  # S1 was durable: nothing to resume
    assert {TokenStore(ns()).state_of(r) for r in got.values()} == {"pending_label"}
    _only_one_identity_each(server, 3)


def test_crash_at_s1_to_s2_keeps_s1_and_the_ack_can_be_redone(server, monkeypatch):
    got = ensure_tokens(ns(), knobs("alpha"), transport=server)
    ref = got[("knob", None, "alpha")]
    crasher = WriteCrasher(monkeypatch, 1)
    with pytest.raises(SimulatedCrash):
        TokenStore(ns()).mark_label_acked(ref, 1)
    crasher.disarm()
    assert TokenStore(ns()).state_of(ref) == "pending_label"
    assert TokenStore(ns()).revision_of(ref) == 0
    TokenStore(ns()).mark_label_acked(ref, 1)
    assert TokenStore(ns()).state_of(ref) == "label_acked"
    assert TokenStore(ns()).revision_of(ref) == 1
    assert len(server.calls) == 1


def test_crash_at_s2_to_s3_keeps_s2_and_the_commit_can_be_redone(server, monkeypatch):
    ref = ensure_tokens(ns(), knobs("alpha"), transport=server)[("knob", None, "alpha")]
    TokenStore(ns()).mark_label_acked(ref, 3)
    crasher = WriteCrasher(monkeypatch, 1)
    with pytest.raises(SimulatedCrash):
        TokenStore(ns()).mark_spec_committed(ref, "searchspecref_" + "5" * 32)
    crasher.disarm()
    assert TokenStore(ns()).state_of(ref) == "label_acked"
    assert TokenStore(ns()).revision_of(ref) == 3
    TokenStore(ns()).mark_spec_committed(ref, "searchspecref_" + "5" * 32)
    assert TokenStore(ns()).state_of(ref) == "spec_committed"


def test_torn_replace_leaves_the_previous_file_intact(server, monkeypatch):
    ensure_tokens(ns(), knobs("alpha"), transport=server)
    before = store_file(ns()).read_bytes()
    root = os.environ["TRAIGENT_RESULTS_FOLDER"]

    def torn(src, dst):
        raise SimulatedCrash

    monkeypatch.setattr(ts.os, "replace", torn)
    with pytest.raises(SimulatedCrash):
        ensure_tokens(ns(), knobs("beta"), transport=server)
    monkeypatch.undo()
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", root)
    monkeypatch.setenv(ts.ENV_FLAG, "true")
    assert store_file(ns()).read_bytes() == before
    assert len(server.calls) == 1  # never sent: S0 for beta was not durable
    assert ensure_tokens(ns(), knobs("alpha", "beta"), transport=server)


def test_operation_key_conflict_on_resume_is_refused_and_never_remints(server):
    with pytest.raises(SimulatedCrash):
        ensure_tokens(ns(), ITEMS, transport=crash_transport(server, before=True))
    state = json.loads(store_file(ns()).read_bytes()[1:])
    (op_key,) = state["operations"]
    server.receipts[op_key] = (
        AGENT_A,
        "something else entirely",
    )  # the key was used before
    before = store_file(ns()).read_bytes()
    for _ in range(2):
        with pytest.raises(TokenStoreError) as err:
            ensure_tokens(ns(), ITEMS, transport=server)
        assert err.value.code == "operation_key_conflict"
    assert {c["operation_key"] for c in server.calls} == {op_key}  # never a new key
    assert store_file(ns()).read_bytes() == before  # journal untouched


def test_resume_batches_every_pending_operation_before_new_work(server):
    with pytest.raises(SimulatedCrash):
        ensure_tokens(
            ns(), knobs("one"), transport=crash_transport(server, before=True)
        )
    got = ensure_tokens(ns(), knobs("one", "two"), transport=server)
    assert len(got) == 2
    assert server.issued == 2 and len(server.receipts) == 2


def test_definitive_server_refusal_abandons_the_intent_so_it_does_not_fail_forever(
    server,
):
    ensure_tokens(ns(), knobs("model"), transport=server)
    server.knobs.clear()  # the server no longer knows the parent: 404 reference_not_found
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), [("choice", "model", "gpt")], transport=server)
    assert err.value.code == "token_refs_reference_not_found"
    state = json.loads(store_file(ns()).read_bytes()[1:])
    assert all(e["state"] != "intent" for e in state["tokens"].values())
    assert len(state["operations"]) == 1  # only the knob's


def test_ambiguous_failures_keep_the_intent_for_resume(server):
    def down(body):
        raise ConnectionError(f"cannot reach {HEBREW}")

    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("k"), transport=down)
    assert err.value.code == "token_refs_transport_failed"
    state = json.loads(store_file(ns()).read_bytes()[1:])
    assert {e["state"] for e in state["tokens"].values()} == {"intent"}
    server.status_override = (500, None)
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("k"), transport=server)
    assert err.value.code == "token_refs_server_error"
    server.status_override = None
    assert ensure_tokens(ns(), knobs("k"), transport=server)
    assert len(server.receipts) == 1


# ----------------------------------------------------- journal transitions S2 / S3


def test_full_journal_walk_and_illegal_transitions(server):
    store = TokenStore(ns())
    ref = ensure_tokens(ns(), knobs("alpha"), transport=server)[("knob", None, "alpha")]
    assert store.state_of(ref) == "pending_label"
    with pytest.raises(TokenStoreError) as err:  # S1 -> S3 skips the label ack
        store.mark_spec_committed(ref, "searchspecref_" + "7" * 32)
    assert err.value.code == "token_state_invalid"
    for bad in (0, -1, True, 2**53, 1.5, "1", None):
        with pytest.raises(TokenStoreError) as err:
            store.mark_label_acked(ref, bad)  # type: ignore[arg-type]
        assert err.value.code == "token_state_invalid"
    store.mark_label_acked(ref, 1)
    store.mark_label_acked(ref, 1)  # idempotent
    assert store.revision_of(ref) == 1
    with pytest.raises(TokenStoreError) as err:
        store.mark_spec_committed(ref, "not-a-spec-ref")
    assert err.value.code == "token_state_invalid"
    store.mark_spec_committed(ref, "searchspecref_" + "7" * 32)
    store.mark_spec_committed(ref, "searchspecref_" + "7" * 32)  # idempotent
    assert store.state_of(ref) == "spec_committed"
    with pytest.raises(TokenStoreError) as err:  # nothing leaves S3
        store.mark_label_acked(ref, 2)
    assert err.value.code == "token_state_invalid"
    with pytest.raises(TokenStoreError) as err:
        store.state_of("knobref_" + "0" * 32)
    assert err.value.code == "token_state_invalid"
    # the walk survives a full reload and a re-validation of the whole file
    assert TokenStore(ns()).state_of(ref) == "spec_committed"


def test_stale_revision_without_a_reader_becomes_label_conflict_unverified(server):
    store = TokenStore(ns())
    got = ensure_tokens(ns(), knobs("alpha", "beta"), transport=server)
    alpha, beta = got[("knob", None, "alpha")], got[("knob", None, "beta")]
    store.require_no_label_conflict()
    store.mark_label_stale(alpha)
    assert store.state_of(alpha) == "label_conflict_unverified"
    assert store.state_of(beta) == "pending_label"
    assert store.conflicted_refs() == [alpha]
    with pytest.raises(TokenStoreError) as err:  # the run refuses to start
        store.require_no_label_conflict()
    assert err.value.code == "label_conflict_unverified"
    with pytest.raises(TokenStoreError):  # no silent way forward
        store.mark_label_acked(alpha, 1)
    with pytest.raises(TokenStoreError):
        store.mark_spec_committed(alpha, "searchspecref_" + "7" * 32)
    # survives a restart
    assert TokenStore(ns()).conflicted_refs() == [alpha]
    # an acked token cannot be flipped to conflict by a stale signal
    store.mark_label_acked(beta, 1)
    with pytest.raises(TokenStoreError) as err:
        store.mark_label_stale(beta)
    assert err.value.code == "token_state_invalid"


# ------------------------------------------------------ durable revision maximum


def test_revision_maximum_is_durable_monotonic_and_refuses_lower(server):
    ref = ensure_tokens(ns(), knobs("alpha"), transport=server)[("knob", None, "alpha")]
    TokenStore(ns()).mark_label_acked(ref, 5)
    # a brand-new store object (= a process restart) still knows the maximum
    store = TokenStore(ns())
    assert store.revision_of(ref) == 5
    for lower in (4, 1):
        with pytest.raises(TokenStoreError) as err:
            store.record_revision(ref, lower)
        assert err.value.code == "token_revision_stale"
        with pytest.raises(TokenStoreError) as err:
            store.mark_label_acked(ref, lower)
        assert err.value.code == "token_revision_stale"
    assert store.revision_of(ref) == 5  # a refusal never lowers it
    store.record_revision(ref, 5)  # equal is fine
    store.record_revision(ref, 9)
    assert TokenStore(ns()).revision_of(ref) == 9
    with pytest.raises(TokenStoreError) as err:
        store.record_revision(ref, 0)
    assert err.value.code == "token_state_invalid"


def test_revision_maximum_survives_a_real_process_restart(server, tmp_path):
    import subprocess
    import sys

    ref = ensure_tokens(ns(), knobs("alpha"), transport=server)[("knob", None, "alpha")]
    TokenStore(ns()).mark_label_acked(ref, 7)
    code = (
        "import sys\n"
        "from traigent.labels.token_store import TokenNamespace, TokenStore, TokenStoreError\n"
        f"s = TokenStore(TokenNamespace.for_backend({ORIGIN!r}, {AGENT_A!r}))\n"
        f"assert s.revision_of({ref!r}) == 7\n"
        "try:\n"
        f"    s.record_revision({ref!r}, 6)\n"
        "except TokenStoreError as e:\n"
        "    sys.exit(0 if e.code == 'token_revision_stale' else 3)\n"
        "sys.exit(2)\n"
    )
    env = dict(os.environ)
    subprocess.run([sys.executable, "-I", "-c", code], check=True, env=env, timeout=120)


# ------------------------------------------------------------------- concurrency


class FileServer:
    """A fake server shared by several processes (state in a flock-guarded file)."""

    def __init__(self, path: str, delay: float = 0.0) -> None:
        self.path, self.delay = path, delay

    def __call__(self, body):
        import fcntl

        with open(self.path + ".lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = json.loads(Path(self.path).read_text())
            except FileNotFoundError:
                data = {"calls": 0, "receipts": {}}
            data["calls"] += 1
            key = body["operation_key"]
            if key not in data["receipts"]:
                data["receipts"][key] = [
                    f"{i['kind']}ref_{secrets.token_hex(16)}" for i in body["items"]
                ]
            Path(self.path).write_text(json.dumps(data))
            refs = data["receipts"][key]
        time.sleep(self.delay)
        return 200, {"refs": refs}


def _worker(server_path, barrier, queue):
    barrier.wait(30)
    got = ensure_tokens(
        ns(), [("knob", None, "shared")], transport=FileServer(server_path, 0.4)
    )
    queue.put(got[("knob", None, "shared")])


def test_two_processes_ensuring_the_same_key_make_one_operation_and_one_token(tmp_path):
    ctx = multiprocessing.get_context("fork")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    server_path = str(tmp_path / "server.json")
    procs = [
        ctx.Process(target=_worker, args=(server_path, barrier, queue))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    refs = {queue.get(timeout=10), queue.get(timeout=10)}
    data = json.loads(Path(server_path).read_text())
    assert data["calls"] == 1 and len(data["receipts"]) == 1  # one operation
    assert len(refs) == 1  # one token
    state = json.loads(store_file(ns()).read_bytes()[1:])
    assert len(state["operations"]) == 1 and len(state["tokens"]) == 1


# ---------------------------------------------------------- batching (> 200 items)


def test_more_than_200_items_split_into_journalled_operations(server):
    names = [f"knob_{i:03d}" for i in range(450)]
    got = ensure_tokens(ns(), knobs(*names), transport=server)
    assert len(got) == 450 and len(set(got.values())) == 450
    sizes = [len(c["items"]) for c in server.calls]
    assert sizes == [200, 200, 50]
    keys = [c["operation_key"] for c in server.calls]
    assert len(set(keys)) == 3
    state = json.loads(store_file(ns()).read_bytes()[1:])
    assert set(state["operations"]) == set(keys)  # each has its own S0 record
    assert all(
        len(op["items"]) == s
        for op, s in zip((state["operations"][k] for k in keys), sizes, strict=True)
    )
    # order is preserved: ordinal i of the first call is names[i]
    first = server.refs[keys[0]]
    assert [got[("knob", None, n)] for n in names[:200]] == first


def test_crash_between_chunks_resumes_without_duplicates(server):
    names = [f"k{i}" for i in range(250)]
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] == 2:
            raise SimulatedCrash  # second chunk: S0 durable, request never completes
        return server(body)

    with pytest.raises(SimulatedCrash):
        ensure_tokens(ns(), knobs(*names), transport=flaky)
    got = ensure_tokens(ns(), knobs(*names), transport=server)
    assert len(got) == 250 and len(set(got.values())) == 250
    assert server.issued == 250  # no duplicate identities


def test_duplicate_requests_in_one_call_collapse(server):
    got = ensure_tokens(ns(), knobs("a", "a", "b"), transport=server)
    assert len(got) == 2 and len(server.calls[0]["items"]) == 2


# ------------------------------------------------------------- the wire body


def test_wire_body_is_exactly_the_schema_shape_and_carries_no_name(server):
    ensure_tokens(ns(), knobs(HEBREW, EMAIL), transport=server)
    ensure_tokens(
        ns(), [("choice", HEBREW, "גבוה"), ("metric", None, "דיוק")], transport=server
    )
    for body in server.calls:
        assert set(body) == {"schema_version", "agent_ref", "operation_key", "items"}
        assert body["schema_version"] == 1 and body["agent_ref"] == AGENT_A
        assert len(body["operation_key"]) == 32 and int(body["operation_key"], 16) >= 0
        for item in body["items"]:
            assert set(item) <= {"kind", "parent_ref"}
            assert item["kind"] in ts.KINDS
            assert ("parent_ref" in item) == (item["kind"] == "choice")
    wire = json.dumps(server.calls, ensure_ascii=False)
    assert not any(c in wire for c in ("גבוה", "דיוק", *CANARIES))


# ------------------------------------------------------ response / error handling


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        ((404, {"detail": "no such route"}), "token_refs_unavailable"),
        ((404, None), "token_refs_unavailable"),
        ((401, {"x": 1}), "token_refs_unauthenticated"),
        ((403, None), "token_refs_forbidden"),
        ((429, None), "token_refs_quota_exceeded"),
        ((503, None), "token_refs_server_error"),
        ((302, None), "token_refs_response_invalid"),
        ((200, None), "token_refs_response_invalid"),
        ((200, {"refs": []}), "token_refs_response_invalid"),
        (
            (200, {"refs": ["knobref_" + "1" * 32], "extra": 1}),
            "token_refs_response_invalid",
        ),
        ((200, {"refs": ["metricref_" + "1" * 32]}), "token_refs_response_invalid"),
        ((200, {"refs": ["knobref_xyz"]}), "token_refs_response_invalid"),
        ((200, {"refs": [5]}), "token_refs_response_invalid"),
        (
            (
                400,
                {
                    "success": False,
                    "error": "invalid_request",
                    "error_code": "invalid_request",
                    "message": "Request is invalid.",
                },
            ),
            "token_refs_invalid_request",
        ),
        (
            (
                409,
                {
                    "success": False,
                    "error": "operation_key_conflict",
                    "error_code": "operation_key_conflict",
                    "message": "Operation key conflicts with an earlier request.",
                },
            ),
            "operation_key_conflict",
        ),
        (  # a closed code with the wrong status or wrong text is not trusted
            (
                400,
                {
                    "success": False,
                    "error": "operation_key_conflict",
                    "error_code": "operation_key_conflict",
                    "message": "Operation key conflicts with an earlier request.",
                },
            ),
            "token_refs_response_invalid",
        ),
        (
            (
                404,
                {
                    "success": False,
                    "error": "reference_not_found",
                    "error_code": "reference_not_found",
                    "message": HEBREW,
                },
            ),
            "token_refs_unavailable",
        ),
    ],
)
def test_server_answers_map_to_closed_codes(server, answer, code):
    server.status_override = answer
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("k"), transport=server)
    assert err.value.code == code


def test_duplicate_refs_in_a_response_or_across_operations_are_refused(server):
    dup = "knobref_" + "1" * 32
    server.status_override = (200, {"refs": [dup, dup]})
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("a", "b"), transport=server)
    assert err.value.code == "token_refs_response_invalid"
    server.status_override = None
    got = ensure_tokens(ns(), knobs("a", "b"), transport=server)  # resumes the same op
    taken = got[("knob", None, "a")]
    # a later operation that answers with an already-issued ref is refused, not stored
    server.status_override = (200, {"refs": [taken]})
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs("c"), transport=server)
    assert err.value.code == "token_refs_response_invalid"
    state = json.loads(store_file(ns()).read_bytes()[1:])
    assert sorted(e["ref"] or "" for e in state["tokens"].values()).count(taken) == 1


def test_unexpected_exceptions_are_replaced_by_a_closed_one_without_cause(
    server, monkeypatch
):
    def explode(*a, **k):
        raise RuntimeError(f"boom {HEBREW} {EMAIL}")

    monkeypatch.setattr(ts, "_write_atomic", explode)
    with pytest.raises(TokenStoreError) as err:
        ensure_tokens(ns(), knobs(HEBREW), transport=server)
    assert err.value.code == "token_store_internal"
    assert err.value.__cause__ is None and err.value.__context__ is None


# ---------------------------------------------------------- privacy canary


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def every_log():
    handler = _Capture()
    loggers = [logging.getLogger()] + [
        lg
        for lg in logging.Logger.manager.loggerDict.values()
        if isinstance(lg, logging.Logger)
    ]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    yield handler
    for lg, level in saved:
        lg.removeHandler(handler)
        lg.setLevel(level)


def _deep_text(exc: BaseException) -> str:
    parts = [str(exc), repr(exc), repr(exc.args)]
    parts += ["".join(traceback.format_exception(exc))]
    for link in (exc.__cause__, exc.__context__):
        if link is not None:
            parts.append(_deep_text(link))
    return "\n".join(parts)


def test_canary_names_live_only_in_the_local_store_file(server, every_log, tmp_path):
    names = [HEBREW, EMAIL]
    ensure_tokens(ns(), knobs(*names), transport=server)
    ensure_tokens(
        ns(),
        [
            ("choice", HEBREW, EMAIL),
            ("choice", EMAIL, HEBREW),
            ("metric", None, HEBREW),
        ],
        transport=server,
    )

    # failure paths: transport error quoting names, server refusal, corruption, wrong mode
    def leaky(body):
        raise ConnectionError(f"{HEBREW} {EMAIL} " + json.dumps(body))

    errors: list[BaseException] = []

    def attempt(fn):
        try:
            fn()
        except TokenStoreError as exc:
            errors.append(exc)

    attempt(lambda: ensure_tokens(ns(), knobs("fresh-" + EMAIL), transport=leaky))
    server.status_override = (404, {"success": False, "error": HEBREW})
    attempt(lambda: ensure_tokens(ns(), knobs("fresh2-" + HEBREW), transport=server))
    server.status_override = None
    path = store_file(ns())
    good = path.read_bytes()
    path.write_bytes(bytes([1]) + good[1:40])
    os.chmod(path, 0o600)
    attempt(lambda: ensure_tokens(ns(), knobs(HEBREW), transport=server))
    path.write_bytes(good)
    os.chmod(path, 0o644)
    attempt(lambda: ensure_tokens(ns(), knobs(HEBREW), transport=server))
    os.chmod(path, 0o600)
    attempt(
        lambda: ensure_tokens(
            ns(), [("choice", "missing-" + EMAIL, HEBREW)], transport=server
        )
    )
    assert len(errors) == 5

    # 1. the names ARE in the local store file (that is its job) ...
    stored = path.read_bytes().decode()
    assert HEBREW in stored and EMAIL in stored
    # 2. ... and nowhere else
    on_wire = json.dumps(server.calls, ensure_ascii=False)
    for canary in CANARIES:
        assert canary not in on_wire
    for record in every_log.records:
        text = (
            record.getMessage()
            + repr(record.args)
            + str(record.exc_info)
            + str(record.__dict__)
        )
        for canary in CANARIES:
            assert canary not in text, record.name
    for exc in errors:
        text = _deep_text(exc)
        for canary in CANARIES:
            assert canary not in text
        assert exc.__cause__ is None and exc.__context__ is None
        assert exc.args == (exc.code,)
    # no other file under the state root holds a name (lock/tmp files included)
    for other in (tmp_path / "state").rglob("*"):
        if other.is_file() and other != path:
            data = other.read_bytes()
            assert not any(c.encode() in data for c in CANARIES), other.name
    # the filename and directory carry no name either
    assert not any(c in str(path) for c in CANARIES)


# ------------------------------------------------- digest cross-check (frozen preimage)

BE_AGENT = "3f2b8c1e-5d4a-4b7e-9c60-1a2b3c4d5e6f"
BE_KNOB = "knobref_" + "1" * 32
#: copied from TraigentBackend tests/unit/services/test_optimization_token_refs_digest.py
#: (84e0f8c0e): (agent id, items, canonical JSON, sha256 hex)
BE_VECTORS = [
    (
        BE_AGENT,
        [("knob", None)],
        '{"agent_id":"3f2b8c1e-5d4a-4b7e-9c60-1a2b3c4d5e6f","items":[[0,"knob",null]]}',
        "92d5e15a6e8c80c4d225d17dd66aecdb4505bfd0a1c51c33d5c1fdebf5311379",
    ),
    (
        BE_AGENT,
        [("knob", None), ("metric", None), ("template", None)],
        '{"agent_id":"3f2b8c1e-5d4a-4b7e-9c60-1a2b3c4d5e6f",'
        '"items":[[0,"knob",null],[1,"metric",null],[2,"template",null]]}',
        "9569f38604cfd6d366a0ce3908925f2ccea72bc34a105fa2e5c93531395f6ebe",
    ),
    (
        BE_AGENT,
        [("choice", BE_KNOB)],
        '{"agent_id":"3f2b8c1e-5d4a-4b7e-9c60-1a2b3c4d5e6f",'
        '"items":[[0,"choice","knobref_11111111111111111111111111111111"]]}',
        "492e0426695c41b75374e55f739c85851ba9c7ca6f187d276bdb8be44b8cdb95",
    ),
    (
        'a"b\\c\x01é€',
        [("knob", None)],
        '{"agent_id":"a\\"b\\\\c\\u0001é€","items":[[0,"knob",null]]}',
        "0c641bed462ed8884dd605ab9668bfede3990190d768fb0a82a97c7f665309f0",
    ),
]


@pytest.mark.parametrize(("agent", "items", "canonical", "digest"), BE_VECTORS)
def test_request_digest_matches_the_backends_pinned_vectors(
    agent, items, canonical, digest
):
    assert ts.request_digest(agent, items).hex() == digest
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest


def test_request_digest_is_sensitive_to_order_and_kind():
    base = ts.request_digest(BE_AGENT, [("knob", None), ("metric", None)])
    assert ts.request_digest(BE_AGENT, [("metric", None), ("knob", None)]) != base
    assert ts.request_digest(BE_AGENT, [("knob", None)]) != base


def test_journalled_digest_is_over_the_wire_order(server):
    ensure_tokens(ns(), ITEMS, transport=server)
    state = json.loads(store_file(ns()).read_bytes()[1:])
    (op,) = state["operations"].values()
    wire = server.calls[0]["items"]
    assert (
        op["digest"]
        == ts.request_digest(
            AGENT_A, [(i["kind"], i.get("parent_ref")) for i in wire]
        ).hex()
    )
