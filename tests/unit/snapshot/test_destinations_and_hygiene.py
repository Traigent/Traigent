"""Destinations (HTTPS, probe, locators), error hygiene, no Traigent traffic."""

from __future__ import annotations

import logging
import socket
import traceback
from pathlib import Path
from typing import Any

import pytest

from tests.unit.snapshot.fakes import FakeS3Client, client_error
from traigent.snapshot import (
    SnapshotErrorCode,
    SnapshotStoreError,
    build_execution_snapshot,
    publish_snapshot,
    verify_snapshot,
)
from traigent.snapshot.backends import LocalDirStore, S3Store
from traigent.snapshot.store import require_secure_endpoint

CANARY = "CONTENT-CANARY-91c2"
SENTINEL_BACKEND = "https://sentinel-traigent-backend.invalid"


# ---------------------------------------------------------------- HTTPS


@pytest.mark.parametrize(
    "url",
    [
        "http://s3.example.com",
        "http://127.0.0.1:9000",  # loopback still needs the explicit flag
        "ftp://x",
        "https://",
        "s3.example.com",
    ],
)
def test_insecure_or_malformed_endpoint_refused(url: str) -> None:
    with pytest.raises(SnapshotStoreError) as caught:
        require_secure_endpoint(url, allow_insecure_local=False, backend="t")
    assert caught.value.code is SnapshotErrorCode.UNSAFE_DESTINATION


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_insecure_loopback_allowed_only_with_flag(host: str) -> None:
    require_secure_endpoint(
        f"http://{host}:9000", allow_insecure_local=True, backend="t"
    )


@pytest.mark.parametrize(
    "url", ["http://s3.example.com", "http://10.0.0.5:9000", "http://evil.test"]
)
def test_flag_does_not_open_non_loopback_http(url: str) -> None:
    with pytest.raises(SnapshotStoreError):
        require_secure_endpoint(url, allow_insecure_local=True, backend="t")


def test_https_endpoint_accepted() -> None:
    require_secure_endpoint(
        "https://s3.example.com", allow_insecure_local=False, backend="t"
    )


def test_s3store_refuses_http_endpoint_client() -> None:
    with pytest.raises(SnapshotStoreError) as caught:
        S3Store("my-bucket", client=FakeS3Client("http://s3.example.com"))
    assert caught.value.code is SnapshotErrorCode.UNSAFE_DESTINATION
    S3Store(
        "my-bucket",
        client=FakeS3Client("http://127.0.0.1:9000"),
        allow_insecure_local=True,
    )


# ---------------------------------------------------------------- S3 probe


def test_custom_endpoint_ignoring_if_none_match_is_refused() -> None:
    client = FakeS3Client("https://minio.example.com", honour_if_none_match=False)
    store = S3Store("my-bucket", "p", client=client)
    with pytest.raises(SnapshotStoreError) as caught:
        store.create_if_absent("blobs/x", b"data")
    assert caught.value.code is SnapshotErrorCode.UNSUPPORTED_BACKEND
    assert all("/.probe/" in key for _, key in client.objects) or not client.objects
    assert ("my-bucket", "p/blobs/x") not in client.objects  # no real write happened


def test_custom_endpoint_honouring_if_none_match_passes_once() -> None:
    client = FakeS3Client("https://minio.example.com")
    store = S3Store("my-bucket", "p", client=client)
    store.create_if_absent("blobs/x", b"1")
    store.create_if_absent("blobs/y", b"2")
    probes = [c for c in client.put_calls if "/.probe/" in c["Key"]]
    assert len(probes) == 2  # one probe pair, run once
    assert all(c["Key"].startswith("p/.probe/") for c in probes)
    assert not [k for _, k in client.objects if "/.probe/" in k]  # best-effort cleanup


def test_aws_endpoint_is_trusted_without_probe() -> None:
    client = FakeS3Client(
        "https://s3.eu-west-1.amazonaws.com", honour_if_none_match=False
    )
    S3Store("my-bucket", client=client).create_if_absent("blobs/x", b"1")
    assert not [c for c in client.put_calls if "/.probe/" in c["Key"]]


def test_probe_transport_failure_raises_not_unsupported() -> None:
    client = FakeS3Client("https://minio.example.com")
    client.fail_with = client_error("SlowDown", 503)
    with pytest.raises(SnapshotStoreError) as caught:
        S3Store("my-bucket", client=client).create_if_absent("blobs/x", b"1")
    assert caught.value.code is SnapshotErrorCode.BACKEND_UNAVAILABLE


# ---------------------------------------------------------------- locators


def test_locator_never_carries_credentials_or_query() -> None:
    userinfo = "AKIAUSER" + ":" + "SECRETPASS"
    client = FakeS3Client(f"https://{userinfo}@minio.example.com/x?sig=ABC")
    store = S3Store("my-bucket", "pre/fix", client=client)
    for text in (store.locator(), repr(store), str(store)):
        assert text == "s3://my-bucket/pre/fix"


def test_credentials_not_reachable_from_repr_of_store() -> None:
    class CredentialedClient(FakeS3Client):
        marker = "AKIA-" + "MARKER-IN-CLIENT"

    store = S3Store("my-bucket", client=CredentialedClient())
    assert "AKIA-MARKER" not in repr(store) and "AKIA-MARKER" not in store.locator()


@pytest.mark.parametrize("bad", ["UPPER", "a", "bad bucket", "../x"])
def test_bad_bucket_names_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        S3Store(bad, client=FakeS3Client())


@pytest.mark.parametrize("bad", ["a/../b", "a//b", "a b", "a?x=1"])
def test_bad_prefixes_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        S3Store("my-bucket", bad, client=FakeS3Client())


# ---------------------------------------------------------------- hygiene


def test_no_content_or_locator_canary_in_logs_or_tracebacks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    snapshot = build_execution_snapshot(
        dataset=[{"input": CANARY, "expected": CANARY}],
        config={"k": CANARY},
        agent=lambda q: q,
        declared_assets={"prompts": {"p": CANARY}, "tool_definitions": {}},
    )
    client = FakeS3Client()
    store = S3Store("canary-bucket-name", "canary/prefix", client=client)
    texts: list[str] = []
    with caplog.at_level(logging.DEBUG):
        ref = publish_snapshot(store, snapshot)
        verify_snapshot(store, ref.manifest_digest)
        client.fail_with = client_error(
            "InternalError", 500, CANARY + " canary-bucket-name"
        )
        for operation in (
            lambda: verify_snapshot(store, ref.manifest_digest),
            lambda: publish_snapshot(store, snapshot),
        ):
            with pytest.raises(SnapshotStoreError) as caught:
                operation()
            error = caught.value
            assert error.__cause__ is None and error.__suppress_context__ is True
            texts.append(
                "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
            )
    for text in [caplog.text, *texts]:
        assert CANARY not in text
        assert "canary-bucket-name" not in text
        assert "canary/prefix" not in text


def test_error_exposes_only_code_part_backend() -> None:
    error = SnapshotStoreError(
        SnapshotErrorCode.AUTH, part="config/config.json", backend="s3"
    )
    assert set(vars(error)) == {"code", "part", "backend"}
    assert "auth" in str(error) and "s3" in str(error)


def test_local_store_errors_hide_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "very-secret-dir-name"
    store = LocalDirStore(root)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError(f"cannot open {root}/x")

    monkeypatch.setattr(Path, "read_bytes", boom)
    with pytest.raises(SnapshotStoreError) as caught:
        store.read("blobs/x")
    text = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    assert "very-secret-dir-name" not in text


# ---------------------------------------------------------------- transport


def test_no_request_reaches_a_traigent_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", SENTINEL_BACKEND)
    monkeypatch.setenv("TRAIGENT_API_URL", SENTINEL_BACKEND)
    network: list[Any] = []
    real_getaddrinfo = socket.getaddrinfo

    def watch_dns(host: Any, *args: Any, **kwargs: Any) -> Any:
        network.append(("dns", host))
        return real_getaddrinfo(host, *args, **kwargs)

    def watch_connect(self: socket.socket, address: Any) -> Any:
        network.append(("connect", address))
        raise OSError("network blocked in test")

    monkeypatch.setattr(socket, "getaddrinfo", watch_dns)
    monkeypatch.setattr(socket.socket, "connect", watch_connect)
    snapshot = build_execution_snapshot(
        dataset=[{"input": 1, "expected": 1}],
        config={"a": 1},
        agent_id="a",
        agent=lambda q: q,
    )
    for store in (
        LocalDirStore(tmp_path / "s"),
        S3Store("my-bucket", client=FakeS3Client()),
    ):
        ref = publish_snapshot(store, snapshot)
        verify_snapshot(store, ref.manifest_digest)
    assert network == []
