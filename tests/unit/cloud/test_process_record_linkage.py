"""C1 SDK producer: opt-in commitment-linkage finalize client (mocked backend)."""

from __future__ import annotations

import base64
import json
import logging
import os
import stat
import traceback
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("traigent_schema.certification.commitment_linkage")

from traigent.cloud.process_record_linkage import (  # noqa: E402
    BUNDLE_SCHEMA,
    CommitmentBundle,
    CommitmentLinkageClientError,
    ProcessRecordRequestError,
    finalize_process_record,
    generate_openings,
)

SESSION = "bsess_" + "0" * 32
FP = "sha256:" + "ab" * 32
RECORD = {"unsigned_manifest": {"project_ref": "p"}}
LINKAGE = {"schema_version": "traigent.commitment_linkage.v1"}
PAYLOAD = {"project_ref": "p", "base_certificate_v0": {}}


class FakeClient:
    """Scripted transport; records every payload it was sent (deep-copied)."""

    def __init__(self, *outcomes: Any, get_outcome: Any = None) -> None:
        self.outcomes = list(outcomes)
        self.get_outcome = get_outcome
        self.sent: list[dict[str, Any]] = []
        self.bundle_path: Path | None = None
        self.bundle_existed_at_send: list[bool] = []

    def finalize_process_record_sync(
        self, sid: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.sent.append(json.loads(json.dumps(payload)))
        if self.bundle_path is not None:
            self.bundle_existed_at_send.append(self.bundle_path.exists())
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def get_process_record_sync(self, sid: str) -> dict[str, Any]:
        if isinstance(self.get_outcome, BaseException):
            raise self.get_outcome
        return self.get_outcome


def _ok() -> dict[str, Any]:
    return {"success": True, "data": RECORD, "commitment_linkage": LINKAGE}


def _all_openings(path: Path) -> list[str]:
    return list(json.loads(path.read_text())["openings"].values())


def test_default_off_sends_no_openings_and_stores_none(tmp_path: Path) -> None:
    client = FakeClient({"success": True, "data": RECORD})
    out = finalize_process_record(
        client, SESSION, PAYLOAD, bundle_path=tmp_path / "b.json"
    )
    assert "openings" not in client.sent[0]
    assert out == {"record": RECORD, "commitment_linkage": None}
    assert "openings" not in json.loads((tmp_path / "b.json").read_text())


def test_link_requires_a_bundle_path() -> None:
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(FakeClient(), SESSION, PAYLOAD, link_commitments=True)
    assert err.value.code == "BUNDLE_REQUIRED"


def test_sends_only_dataset_and_config_space_32_byte_openings(tmp_path: Path) -> None:
    client = FakeClient(_ok())
    finalize_process_record(
        client, SESSION, PAYLOAD, link_commitments=True, bundle_path=tmp_path / "b.json"
    )
    openings = client.sent[0]["openings"]
    assert set(openings) == {"dataset", "config_space"}
    assert all(len(base64.b64decode(v, validate=True)) == 32 for v in openings.values())
    assert openings["dataset"] != openings["config_space"]
    # what was sent equals what the bundle holds
    assert sorted(openings.values()) == sorted(_all_openings(tmp_path / "b.json"))


def test_bundle_is_written_before_send_and_retry_reuses_openings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "b.json"
    client = FakeClient(ProcessRecordRequestError(503, "STORAGE_UNAVAILABLE"), _ok())
    client.bundle_path = path
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            client, SESSION, PAYLOAD, link_commitments=True, bundle_path=path
        )
    assert err.value.code == "BACKEND_REFUSED"
    assert client.bundle_existed_at_send == [True]  # on disk before the send
    first = _all_openings(path)

    out = finalize_process_record(
        client, SESSION, PAYLOAD, link_commitments=True, bundle_path=path
    )
    assert (
        client.sent[1]["openings"] == client.sent[0]["openings"]
    )  # REUSED, not regenerated
    assert _all_openings(path) == first
    assert out["commitment_linkage"] == LINKAGE
    data = json.loads(path.read_text())
    assert data["schema"] == BUNDLE_SCHEMA
    assert data["record"] == RECORD and data["commitment_linkage"] == LINKAGE


def test_transport_exception_carrying_openings_is_not_leaked(tmp_path: Path) -> None:
    path = tmp_path / "b.json"

    class Leaky(FakeClient):
        def finalize_process_record_sync(
            self, sid: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            raise RuntimeError(f"boom {payload!r}")

    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            Leaky(), SESSION, PAYLOAD, link_commitments=True, bundle_path=path
        )
    assert err.value.code == "TRANSPORT_ERROR"
    assert err.value.__cause__ is None and err.value.__context__ is None
    text = (
        "".join(traceback.format_exception(err.value))
        + str(err.value)
        + repr(err.value)
    )
    for opening in _all_openings(path):
        assert opening not in text


def test_bundle_mode_is_0600_and_loosened_mode_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    finalize_process_record(
        FakeClient(_ok()), SESSION, PAYLOAD, link_commitments=True, bundle_path=path
    )
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    os.chmod(path, 0o644)
    with pytest.raises(CommitmentLinkageClientError) as err:
        CommitmentBundle.load(path)
    assert err.value.code == "BUNDLE_PERMISSIONS"


def test_openings_absent_from_logs_stdout_and_reprs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "b.json"
    caplog.set_level(logging.DEBUG)
    finalize_process_record(
        FakeClient(_ok()), SESSION, PAYLOAD, link_commitments=True, bundle_path=path
    )
    bundle = CommitmentBundle.load(path)
    surfaces = [
        capsys.readouterr().out,
        capsys.readouterr().err,
        caplog.text,
        repr(bundle),
        str(bundle),
    ]
    surfaces += [repr(v) + str(v) for v in bundle.openings.values()]
    surfaces += [repr(generate_openings()), repr(bundle.openings)]
    blob = "\n".join(surfaces)
    for opening in _all_openings(path):
        assert opening not in blob
    for raw in bundle.openings.values():
        assert bytes(raw).hex() not in blob
    assert "redacted" in repr(bundle)


def test_existing_bundle_for_another_session_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    CommitmentBundle.create(path, "bsess_" + "1" * 32, with_openings=True)
    client = FakeClient()
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            client, SESSION, PAYLOAD, link_commitments=True, bundle_path=path
        )
    assert err.value.code == "BUNDLE_SESSION_MISMATCH"
    assert client.sent == []


def test_request_payload_must_not_carry_openings(tmp_path: Path) -> None:
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            FakeClient(),
            SESSION,
            {"openings": {}},
            link_commitments=True,
            bundle_path=tmp_path / "b.json",
        )
    assert err.value.code == "REQUEST_HAS_OPENINGS"


@pytest.mark.parametrize(
    "code",
    [
        "COMMITMENT_LINKAGE_DISABLED",
        "COMMITMENT_LINKAGE_OPENING_SHAPE",
        "COMMITMENT_LINKAGE_UNAVAILABLE",
        "COMMITMENT_LINKAGE_CORRUPT",
        "ISSUANCE_IN_PROGRESS",
    ],
)
def test_backend_closed_codes_map_with_no_echo(tmp_path: Path, code: str) -> None:
    client = FakeClient(ProcessRecordRequestError(400, code))
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            client,
            SESSION,
            PAYLOAD,
            link_commitments=True,
            bundle_path=tmp_path / "b.json",
        )
    assert err.value.code == code
    assert str(err.value).startswith(code)


def test_unknown_backend_code_collapses_to_backend_refused(tmp_path: Path) -> None:
    client = FakeClient(ProcessRecordRequestError(400, "SECRET_ECHO_xyz"))
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            client,
            SESSION,
            PAYLOAD,
            link_commitments=True,
            bundle_path=tmp_path / "b.json",
        )
    assert err.value.code == "BACKEND_REFUSED"
    assert "xyz" not in str(err.value)


def test_duplicate_issuance_after_ambiguous_commit_fetches_instead_of_reminting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "b.json"
    client = FakeClient(
        ProcessRecordRequestError(409, "DUPLICATE_ISSUANCE"), get_outcome=_ok()
    )
    out = finalize_process_record(
        client, SESSION, PAYLOAD, link_commitments=True, bundle_path=path
    )
    assert len(client.sent) == 1
    assert out["commitment_linkage"] == LINKAGE
    assert json.loads(path.read_text())["record"] == RECORD


def test_completed_bundle_short_circuits_without_sending(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    finalize_process_record(
        FakeClient(_ok()), SESSION, PAYLOAD, link_commitments=True, bundle_path=path
    )
    client = FakeClient()
    out = finalize_process_record(
        client, SESSION, PAYLOAD, link_commitments=True, bundle_path=path
    )
    assert client.sent == [] and out["record"] == RECORD


def test_missing_linkage_in_response_is_reported_not_faked(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            FakeClient({"success": True, "data": RECORD}),
            SESSION,
            PAYLOAD,
            link_commitments=True,
            bundle_path=path,
        )
    assert err.value.code == "LINKAGE_ABSENT"
    assert "commitment_linkage" not in json.loads(path.read_text())


def test_bad_fp_digest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            FakeClient(),
            SESSION,
            PAYLOAD,
            link_commitments=True,
            bundle_path=tmp_path / "b.json",
            fp_digests={"agent": FP},
        )
    assert err.value.code == "BAD_FP_DIGEST"


# --- BackendIntegratedClient transport -------------------------------------------------


def test_backend_client_transport_posts_finalize_and_keeps_errors_content_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud import backend_client as bc

    class Resp:
        def __init__(self, status: int, body: Any) -> None:
            self.status_code, self._body = status, body

        def json(self) -> Any:
            return self._body

    calls: list[dict[str, Any]] = []
    answers = [
        Resp(201, {"data": RECORD, "commitment_linkage": LINKAGE}),
        Resp(409, {"error_code": "DUPLICATE_ISSUANCE", "message": "SECRET-BODY"}),
    ]

    def fake_request(method: str, url: str, **kw: Any) -> Resp:
        calls.append({"method": method, "url": url, **kw})
        return answers.pop(0)

    import requests

    monkeypatch.setattr(requests, "request", fake_request)
    client = bc.BackendIntegratedClient.__new__(bc.BackendIntegratedClient)
    client.api_base_url = "https://api.example.test/api/v1"
    client.timeout = 30.0
    monkeypatch.setattr(
        client,
        "_raise_if_backend_egress_disabled",
        lambda *_a, **_k: None,
        raising=False,
    )
    monkeypatch.setattr(
        client,
        "_get_sync_auth_headers",
        lambda target="backend": {"X-API-Key": "k"},
        raising=False,
    )

    body = client.finalize_process_record_sync(SESSION, {"openings": {"dataset": "AA"}})
    assert body["commitment_linkage"] == LINKAGE
    assert calls[0]["method"] == "POST"
    assert (
        calls[0]["url"]
        == f"https://api.example.test/api/v1/certification/process-records/{SESSION}/finalize"
    )
    assert calls[0]["allow_redirects"] is False

    with pytest.raises(ProcessRecordRequestError) as err:
        client.finalize_process_record_sync(SESSION, {})
    assert err.value.status_code == 409 and err.value.error_code == "DUPLICATE_ISSUANCE"
    assert "SECRET-BODY" not in str(err.value)


# --- r2 review fixes ---------------------------------------------------------------------


def _chain_text(exc: BaseException) -> str:
    """repr + traceback text + every linked exception's repr/args (cause AND context)."""
    seen: list[str] = []
    stack: list[BaseException | None] = [exc]
    visited: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in visited:
            continue
        visited.add(id(cur))
        seen.append(repr(cur) + str(cur) + repr(cur.args))
        stack.extend([cur.__cause__, cur.__context__])
    return "".join(seen) + "".join(traceback.format_exception(exc)) + repr(exc)


@pytest.mark.parametrize("loser_sends", [True, False])
def test_concurrent_first_init_sends_one_set_of_openings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loser_sends: bool
) -> None:
    """Schedule-independent: both outcomes (loser sends / loser returns cached) are forced."""
    import threading

    from traigent.cloud import process_record_linkage as prl

    path = tmp_path / "b.json"
    gen_barrier = threading.Barrier(2, timeout=10)
    send_barrier = threading.Barrier(2, timeout=10)
    winner_done = threading.Event()
    real = prl.generate_openings

    def racing() -> dict[str, bytes]:
        openings = real()
        gen_barrier.wait()  # both callers hold distinct openings before either publishes
        if threading.current_thread().name == "t1" and not loser_sends:
            # t1 loses on purpose: t0 publishes AND stores its result first.
            assert winner_done.wait(timeout=10)
        return openings

    monkeypatch.setattr(prl, "generate_openings", racing)

    class SyncClient(FakeClient):
        def finalize_process_record_sync(
            self, sid: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            if loser_sends:
                send_barrier.wait()  # both callers are inside the transport together
            return super().finalize_process_record_sync(sid, payload)

    clients = [SyncClient(_ok()), SyncClient(_ok())]
    errors: list[BaseException] = []

    def run(client: FakeClient) -> None:
        try:
            finalize_process_record(
                client, SESSION, PAYLOAD, link_commitments=True, bundle_path=path
            )
            if threading.current_thread().name == "t0":
                winner_done.set()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(c,), name=f"t{i}")
        for i, c in enumerate(clients)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors
    sent = [p for c in clients for p in c.sent]
    assert len(sent) == (2 if loser_sends else 1)
    distinct = {json.dumps(p["openings"], sort_keys=True) for p in sent}
    assert len(distinct) == 1  # only ONE set of openings was ever sent
    assert json.loads(path.read_text())["openings"] == sent[0]["openings"]
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_loser_fsyncs_directory_before_its_send_while_winner_is_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from traigent.cloud import process_record_linkage as prl

    path = tmp_path / "b.json"
    events: list[tuple[str, str]] = []
    linked = threading.Event()
    release_winner = threading.Event()
    real_link, real_fsync = os.link, os.fsync

    def pausing_link(src: Any, dst: Any) -> None:
        real_link(src, dst)
        if threading.current_thread().name == "winner":
            linked.set()
            assert release_winner.wait(timeout=10)  # paused right after os.link

    def recording_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            events.append((threading.current_thread().name, "fsync_dir"))
        real_fsync(fd)

    monkeypatch.setattr(prl.os, "link", pausing_link)
    monkeypatch.setattr(prl.os, "fsync", recording_fsync)

    class RecClient(FakeClient):
        def finalize_process_record_sync(
            self, sid: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            events.append((threading.current_thread().name, "send"))
            return super().finalize_process_record_sync(sid, payload)

    errors: list[BaseException] = []

    def run() -> None:
        try:
            finalize_process_record(
                RecClient(_ok()),
                SESSION,
                PAYLOAD,
                link_commitments=True,
                bundle_path=path,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    winner = threading.Thread(target=run, name="winner")
    winner.start()
    assert linked.wait(timeout=10)
    loser = threading.Thread(target=run, name="loser")
    loser.start()
    loser.join(timeout=20)  # loser finishes while the winner is still paused
    assert not loser.is_alive()
    release_winner.set()
    winner.join(timeout=20)
    assert not errors
    loser_events = [e for who, e in events if who == "loser"]
    assert "fsync_dir" in loser_events and "send" in loser_events
    assert loser_events.index("fsync_dir") < loser_events.index("send")
    assert ("winner", "fsync_dir") not in events[: events.index(("loser", "send"))]


def test_link_commitments_refused_off_posix_before_any_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.cloud import process_record_linkage as prl

    monkeypatch.setattr(prl, "_is_posix", lambda: False)  # == os.name != "posix"
    monkeypatch.setattr(
        prl, "generate_openings", lambda: pytest.fail("secret generated")
    )
    client = FakeClient()
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            client,
            SESSION,
            PAYLOAD,
            link_commitments=True,
            bundle_path=tmp_path / "b.json",
        )
    assert err.value.code == "LINKAGE_UNSUPPORTED_PLATFORM"
    assert client.sent == [] and not (tmp_path / "b.json").exists()


def test_bundle_only_refused_off_posix_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.cloud import process_record_linkage as prl

    monkeypatch.setattr(prl, "_is_posix", lambda: False)
    client = FakeClient()
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            client, SESSION, PAYLOAD, bundle_path=tmp_path / "b.json"
        )
    assert err.value.code == "LINKAGE_UNSUPPORTED_PLATFORM"
    assert client.sent == [] and not (tmp_path / "b.json").exists()


def test_bundle_owned_by_another_uid_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "b.json"
    CommitmentBundle.create(path, SESSION, with_openings=True)
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(path).st_uid + 1)
    with pytest.raises(CommitmentLinkageClientError) as err:
        CommitmentBundle.load(path)
    assert err.value.code == "BUNDLE_PERMISSIONS"


@pytest.mark.parametrize("mode", [0o400, 0o700, 0o640, 0o644])
def test_bundle_mode_must_be_exactly_0600(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "b.json"
    CommitmentBundle.create(path, SESSION, with_openings=True)
    os.chmod(path, mode)
    with pytest.raises(CommitmentLinkageClientError) as err:
        CommitmentBundle.load(path)
    assert err.value.code == "BUNDLE_PERMISSIONS"


def test_temp_file_removed_when_link_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traigent.cloud import process_record_linkage as prl

    def boom(src: Any, dst: Any) -> None:
        raise OSError("no hard links here")

    monkeypatch.setattr(prl.os, "link", boom)
    path = tmp_path / "b.json"
    client = FakeClient()
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            client, SESSION, PAYLOAD, link_commitments=True, bundle_path=path
        )
    assert err.value.code == "BUNDLE_WRITE_FAILED"
    assert client.sent == []  # fail closed: nothing sent
    assert list(tmp_path.iterdir()) == []  # no bundle, no leftover temp


def test_leftover_temp_file_is_never_picked_up_as_a_bundle(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    leftover = tmp_path / ".b.json.deadbeef0000.tmp"
    leftover.write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "build_session_row_id": SESSION,
                "openings": {"dataset": "LEFTOVER", "config_space": "LEFTOVER"},
            }
        )
    )
    os.chmod(leftover, 0o600)
    client = FakeClient(_ok())
    finalize_process_record(
        client, SESSION, PAYLOAD, link_commitments=True, bundle_path=path
    )
    assert "LEFTOVER" not in json.dumps(client.sent)
    assert leftover.exists()  # untouched
    assert json.loads(path.read_text())["openings"] == client.sent[0]["openings"]


def test_exclusive_create_never_replaces_an_existing_bundle(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    CommitmentBundle.create(path, SESSION, with_openings=True)
    first = path.read_text()
    with pytest.raises(CommitmentLinkageClientError) as err:
        CommitmentBundle.create(path, SESSION, with_openings=True)
    assert err.value.code == "BUNDLE_EXISTS"
    assert path.read_text() == first


@pytest.mark.parametrize(
    "body",
    [
        '{"schema": "OPENING-SECRET-AAAA", "openings": {"dataset": "OPENING-SECRET-BBBB"',
        '{"openings": {"dataset": "OPENING-SECRET-CCCC"}} trailing',
    ],
)
def test_malformed_bundle_error_chain_holds_no_opening(
    tmp_path: Path, body: str
) -> None:
    path = tmp_path / "b.json"
    path.write_text(body)
    os.chmod(path, 0o600)
    with pytest.raises(CommitmentLinkageClientError) as err:
        CommitmentBundle.load(path)
    assert err.value.code == "BUNDLE_CORRUPT"
    assert err.value.__cause__ is None and err.value.__context__ is None
    assert "OPENING-SECRET" not in _chain_text(err.value)


def test_bundle_with_bad_opening_value_has_clean_chain(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "build_session_row_id": SESSION,
                "openings": {"dataset": "OPENING-SECRET-DDDD!!"},
            }
        )
    )
    os.chmod(path, 0o600)
    with pytest.raises(CommitmentLinkageClientError) as err:
        CommitmentBundle.load(path)
    assert err.value.__cause__ is None and err.value.__context__ is None
    assert "OPENING-SECRET" not in _chain_text(err.value)


def _transport_client(monkeypatch: pytest.MonkeyPatch, adapter_exc: Exception) -> Any:
    from traigent.cloud import backend_client as bc

    import requests

    def failing_request(method: str, url: str, **kw: Any) -> Any:
        raise adapter_exc

    monkeypatch.setattr(requests, "request", failing_request)
    client = bc.BackendIntegratedClient.__new__(bc.BackendIntegratedClient)
    client.api_base_url = "https://api.example.test/api/v1"
    client.timeout = 30.0
    monkeypatch.setattr(
        client,
        "_raise_if_backend_egress_disabled",
        lambda *_a, **_k: None,
        raising=False,
    )
    monkeypatch.setattr(
        client,
        "_get_sync_auth_headers",
        lambda target="backend": {"X-API-Key": "KEY-SECRET-ZZZZ"},
        raising=False,
    )
    return client


def test_transport_failing_adapter_leaks_no_openings_or_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    # Built from parts so the literal is not in any traceback source line.
    secret = "OPEN" + "ING-SECRET-EEEE"
    api_key = "KEY-" + "SECRET-ZZZZ"
    body = {"openings": {"dataset": secret}}
    prepared = requests.Request(
        "POST",
        "https://api.example.test/x",
        headers={"X-API-Key": api_key},
        json=body,
    ).prepare()
    client = _transport_client(
        monkeypatch, requests.ConnectionError("boom " + secret, request=prepared)
    )
    for call in (
        lambda: client.finalize_process_record_sync(SESSION, body),
        lambda: client.get_process_record_sync(SESSION),
    ):
        with pytest.raises(ProcessRecordRequestError) as err:
            call()
        assert err.value.error_code == "TRANSPORT_ERROR"
        assert err.value.__cause__ is None and err.value.__context__ is None
        text = _chain_text(err.value)
        assert "OPENING-SECRET" not in text and "KEY-SECRET" not in text


def test_transport_malformed_response_has_clean_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud import backend_client as bc

    import requests

    class Resp:
        status_code = 201

        def json(self) -> Any:
            return json.loads('{"x": "OPENING-SECRET-FFFF"')

    monkeypatch.setattr(requests, "request", lambda *a, **k: Resp())
    client = bc.BackendIntegratedClient.__new__(bc.BackendIntegratedClient)
    client.api_base_url = "https://api.example.test/api/v1"
    client.timeout = 30.0
    monkeypatch.setattr(
        client,
        "_raise_if_backend_egress_disabled",
        lambda *_a, **_k: None,
        raising=False,
    )
    monkeypatch.setattr(
        client, "_get_sync_auth_headers", lambda target="backend": {}, raising=False
    )
    with pytest.raises(ProcessRecordRequestError) as err:
        client.get_process_record_sync(SESSION)
    assert err.value.__cause__ is None and err.value.__context__ is None
    assert "OPENING-SECRET" not in _chain_text(err.value)


def test_transport_gate_refusal_passes_through_and_nothing_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud import backend_client as bc
    from traigent.labels.transport_gate import TransportGateRefusal

    client = _transport_client(monkeypatch, AssertionError("must not be sent"))
    sent: list[str] = []

    import requests

    monkeypatch.setattr(
        requests, "request", lambda *a, **k: sent.append("sent")  # pragma: no cover
    )

    def refusing_preflight(*_a: Any, **_k: Any) -> None:
        raise TransportGateRefusal("process-records", {"blocked": 1})

    monkeypatch.setattr(bc, "preflight_requests", refusing_preflight)

    with pytest.raises(TransportGateRefusal) as err:
        client.finalize_process_record_sync(SESSION, {})
    assert not isinstance(err.value, ProcessRecordRequestError)
    assert sent == []
