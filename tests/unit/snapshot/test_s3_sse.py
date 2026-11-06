"""Optional server-side encryption on S3Store puts (SSE-KMS buckets deny plain PutObject)."""

from __future__ import annotations

import pytest

from tests.unit.snapshot.fakes import FakeS3Client
from traigent.snapshot.backends import S3Store

_ALIAS = "alias/traigent-dev-snapshots"
_ARN = "arn:aws:kms:eu-west-1:123456789012:key/12345678-1234-1234-1234-123456789012"


@pytest.fixture(autouse=True)
def _no_sse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRAIGENT_SNAPSHOT_S3_SSE", raising=False)
    monkeypatch.delenv("TRAIGENT_SNAPSHOT_S3_KMS_KEY_ID", raising=False)


def test_default_put_sends_no_encryption() -> None:
    client = FakeS3Client()
    S3Store("bucket", "p", client=client).create_if_absent("k", b"x")
    assert "ServerSideEncryption" not in client.put_calls[0]
    assert "SSEKMSKeyId" not in client.put_calls[0]


def test_kms_encryption_and_key_are_sent() -> None:
    client = FakeS3Client()
    store = S3Store(
        "bucket",
        "p",
        client=client,
        server_side_encryption="aws:kms",
        sse_kms_key_id=_ALIAS,
    )
    store.create_if_absent("k", b"x")
    call = client.put_calls[0]
    assert call["ServerSideEncryption"] == "aws:kms"
    assert call["SSEKMSKeyId"] == _ALIAS
    assert call["IfNoneMatch"] == "*"
    assert _ALIAS not in repr(store)


def test_kms_without_key_id_omits_sse_kms_key_id_header() -> None:
    client = FakeS3Client()
    S3Store("bucket", client=client, server_side_encryption="aws:kms").create_if_absent(
        "k", b"x"
    )
    assert client.put_calls[0]["ServerSideEncryption"] == "aws:kms"
    assert "SSEKMSKeyId" not in client.put_calls[0]


def test_environment_configures_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_SNAPSHOT_S3_SSE", "aws:kms")
    monkeypatch.setenv("TRAIGENT_SNAPSHOT_S3_KMS_KEY_ID", _ARN)
    client = FakeS3Client()
    S3Store("bucket", client=client).create_if_absent("k", b"x")
    assert client.put_calls[0]["SSEKMSKeyId"] == _ARN


def test_custom_endpoint_probe_puts_are_encrypted_too() -> None:
    client = FakeS3Client("https://minio.example.com")
    S3Store("bucket", client=client, server_side_encryption="AES256").create_if_absent(
        "k", b"x"
    )
    assert len(client.put_calls) >= 3
    assert all(c["ServerSideEncryption"] == "AES256" for c in client.put_calls)


@pytest.mark.parametrize(
    "sse,key",
    [
        ("rot13", None),
        ("AES256", _ALIAS),
        (None, _ALIAS),
        ("aws:kms", "not a key id"),
        ("aws:kms", "alias/" + "a" * 300),
    ],
)
def test_invalid_settings_are_refused_without_echo(
    sse: str | None, key: str | None
) -> None:
    with pytest.raises(ValueError) as caught:
        S3Store(
            "bucket",
            client=FakeS3Client(),
            server_side_encryption=sse,
            sse_kms_key_id=key,
        )
    text = str(caught.value)
    assert (key or "<none>") not in text
    assert (sse or "<none>") not in text
