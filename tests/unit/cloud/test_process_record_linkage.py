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
