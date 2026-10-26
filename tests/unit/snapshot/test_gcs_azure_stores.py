"""GCS and Azure Blob backends: conformance plus backend-specific behaviour."""

from __future__ import annotations

import builtins
import traceback
from typing import Any

import pytest

from tests.unit.snapshot.conformance import CANARY, StoreConformance
from tests.unit.snapshot.fakes import FakeAzureContainer, FakeGCSClient
from traigent.snapshot import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.backends import AzureBlobStore, GCSStore

gcs_exceptions = pytest.importorskip("google.api_core.exceptions")
gauth_exceptions = pytest.importorskip("google.auth.exceptions")
azure_exceptions = pytest.importorskip("azure.core.exceptions")


class TestGCSStoreConformance(StoreConformance):
    def make_store(self, tmp_path: Any) -> GCSStore:
        return GCSStore("my-bucket", "snap/prefix", client=FakeGCSClient())

    def break_backend(self, store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        store._client.fail_with = gcs_exceptions.InternalServerError(CANARY)

    def test_locator_format(self) -> None:
        client = FakeGCSClient()
        assert GCSStore("my-bucket", "a/b/", client=client).locator() == (
            "gs://my-bucket/a/b"
        )
        assert GCSStore("my-bucket", client=client).locator() == "gs://my-bucket"

    def test_generation_match_zero_sent(self) -> None:
        client = FakeGCSClient()
        GCSStore("my-bucket", "p", client=client).create_if_absent("blobs/x", b"1")
        call = client.upload_calls[0]
        assert call["if_generation_match"] == 0
        assert call["name"] == "p/blobs/x"
        assert call["timeout"] == 30

    def test_412_reads_and_compares_never_overwrites(self) -> None:
        client = FakeGCSClient()
        store = GCSStore("my-bucket", client=client)
        store.create_if_absent("blobs/x", b"one")
        with pytest.raises(SnapshotStoreError) as caught:
            store.create_if_absent("blobs/x", b"two")
        assert caught.value.code is SnapshotErrorCode.CONFLICT
        assert client.objects["blobs/x"] == b"one"

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (gcs_exceptions.Forbidden("x"), SnapshotErrorCode.AUTH),
            (gcs_exceptions.Unauthorized("x"), SnapshotErrorCode.AUTH),
            (gauth_exceptions.DefaultCredentialsError("x"), SnapshotErrorCode.AUTH),
            (
                gcs_exceptions.ServiceUnavailable("x"),
                SnapshotErrorCode.BACKEND_UNAVAILABLE,
            ),
            (OSError("x"), SnapshotErrorCode.BACKEND_UNAVAILABLE),
        ],
    )
    def test_error_mapping_raises_never_none(
        self, error: Exception, expected: SnapshotErrorCode
    ) -> None:
        client = FakeGCSClient()
        store = GCSStore("my-bucket", client=client)
        client.fail_with = error
        for operation in (
            lambda: store.read("blobs/x"),
            lambda: store.create_if_absent("blobs/x", b"d"),
        ):
            with pytest.raises(SnapshotStoreError) as caught:
                operation()
            assert caught.value.code is expected

    def test_missing_bucket_on_write_is_not_found(self) -> None:
        client = FakeGCSClient()
        store = GCSStore("my-bucket", client=client)
        client.fail_with = gcs_exceptions.NotFound("no bucket")
        with pytest.raises(SnapshotStoreError) as caught:
            store.create_if_absent("blobs/x", b"d")
        assert caught.value.code is SnapshotErrorCode.NOT_FOUND

    @pytest.mark.parametrize(
        "base", ["http://storage.example.com", "ftp://x", "storage.example.com"]
    )
    def test_insecure_emulator_endpoint_refused(self, base: str) -> None:
        with pytest.raises(SnapshotStoreError) as caught:
            GCSStore("my-bucket", client=FakeGCSClient(base))
        assert caught.value.code is SnapshotErrorCode.UNSAFE_DESTINATION

    def test_loopback_emulator_needs_explicit_flag(self) -> None:
        with pytest.raises(SnapshotStoreError):
            GCSStore("my-bucket", client=FakeGCSClient("http://localhost:4443"))
        store = GCSStore(
            "my-bucket",
            client=FakeGCSClient("http://localhost:4443"),
            allow_insecure_local=True,
        )
        assert store.locator() == "gs://my-bucket"

    def test_unknown_endpoint_refused(self) -> None:
        client = FakeGCSClient()
        del client._connection
        with pytest.raises(SnapshotStoreError) as caught:
            GCSStore("my-bucket", client=client)
        assert caught.value.code is SnapshotErrorCode.UNSAFE_DESTINATION

    def test_bad_bucket_refused(self) -> None:
        with pytest.raises(ValueError):
            GCSStore("A", client=FakeGCSClient())

    def test_missing_sdk_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("google.cloud"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ImportError, match=r"traigent\[snapshot-gcs\]"):
            GCSStore("my-bucket")


SAS_MARKER = "sig-" + "SASSECRET" + "-9d1"
SAS_URL = "https://acct.blob.core.windows.net/cont?sv=2024-01-01&sig=" + SAS_MARKER


class TestAzureBlobStoreConformance(StoreConformance):
    def make_store(self, tmp_path: Any) -> AzureBlobStore:
        return AzureBlobStore(FakeAzureContainer(), "snap/prefix")

    def break_backend(self, store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        store._container.fail_with = azure_exceptions.ServiceResponseError(CANARY)

    def test_locator_format(self) -> None:
        assert AzureBlobStore(FakeAzureContainer(), "a/b/").locator() == (
            "azblob://acct.blob.core.windows.net/cont/a/b"
        )
        assert AzureBlobStore(FakeAzureContainer()).locator() == (
            "azblob://acct.blob.core.windows.net/cont"
        )

    def test_overwrite_false_sent(self) -> None:
        container = FakeAzureContainer()
        AzureBlobStore(container, "p").create_if_absent("blobs/x", b"1")
        call = container.upload_calls[0]
        assert call["overwrite"] is False and call["name"] == "p/blobs/x"
        assert call["timeout"] == 30

    def test_sas_never_in_locator_repr_attrs_or_errors(self) -> None:
        container = FakeAzureContainer(SAS_URL)
        store = AzureBlobStore(container, "p")
        assert SAS_MARKER not in store.locator()
        assert SAS_MARKER not in repr(store)
        assert "?" not in store.locator() and "sig=" not in store.locator()
        owned = {
            name: value for name, value in vars(store).items() if name != "_container"
        }
        assert SAS_MARKER not in repr(owned)
        container.fail_with = azure_exceptions.HttpResponseError(SAS_URL)
        with pytest.raises(SnapshotStoreError) as caught:
            store.read("blobs/x")
        text = "".join(
            traceback.format_exception(
                type(caught.value), caught.value, caught.value.__traceback__
            )
        )
        assert SAS_MARKER not in text and SAS_MARKER not in repr(vars(caught.value))

    def test_userinfo_and_port_handling(self) -> None:
        container = FakeAzureContainer("http://user:pw@127.0.0.1:10000/acct/cont")
        store = AzureBlobStore(container, allow_insecure_local=True)
        assert store.locator() == "azblob://127.0.0.1:10000/cont"

    def test_existing_blob_reads_and_compares(self) -> None:
        container = FakeAzureContainer()
        store = AzureBlobStore(container)
        store.create_if_absent("blobs/x", b"one")
        with pytest.raises(SnapshotStoreError) as caught:
            store.create_if_absent("blobs/x", b"two")
        assert caught.value.code is SnapshotErrorCode.CONFLICT
        assert container.objects["blobs/x"] == b"one"

    def test_client_authentication_error_is_auth(self) -> None:
        container = FakeAzureContainer()
        store = AzureBlobStore(container)
        container.fail_with = azure_exceptions.ClientAuthenticationError("x")
        with pytest.raises(SnapshotStoreError) as caught:
            store.read("blobs/x")
        assert caught.value.code is SnapshotErrorCode.AUTH

    def test_401_http_error_is_auth_and_500_is_unavailable(self) -> None:
        container = FakeAzureContainer()
        store = AzureBlobStore(container)
        for status, expected in (
            (401, SnapshotErrorCode.AUTH),
            (403, SnapshotErrorCode.AUTH),
            (500, SnapshotErrorCode.BACKEND_UNAVAILABLE),
        ):
            error = azure_exceptions.HttpResponseError("x")
            error.status_code = status
            container.fail_with = error
            with pytest.raises(SnapshotStoreError) as caught:
                store.create_if_absent("blobs/x", b"d")
            assert caught.value.code is expected

    def test_missing_container_on_write_is_not_found(self) -> None:
        container = FakeAzureContainer()
        store = AzureBlobStore(container)
        container.fail_with = azure_exceptions.ResourceNotFoundError("x")
        with pytest.raises(SnapshotStoreError) as caught:
            store.create_if_absent("blobs/x", b"d")
        assert caught.value.code is SnapshotErrorCode.NOT_FOUND

    @pytest.mark.parametrize(
        "url",
        ["http://acct.blob.core.windows.net/cont", "ftp://x/cont", "acct/cont"],
    )
    def test_insecure_account_url_refused(self, url: str) -> None:
        with pytest.raises(SnapshotStoreError) as caught:
            AzureBlobStore(FakeAzureContainer(url))
        assert caught.value.code is SnapshotErrorCode.UNSAFE_DESTINATION
        with pytest.raises(SnapshotStoreError):
            AzureBlobStore.from_account(url.rsplit("/", 1)[0], "cont", None)

    def test_from_account_builds_a_real_client(self) -> None:
        store = AzureBlobStore.from_account(
            "https://acct.blob.core.windows.net", "cont", "not-a-real-key", "pre"
        )
        assert store.locator() == "azblob://acct.blob.core.windows.net/cont/pre"
        assert "not-a-real-key" not in repr(store)

    def test_from_account_bad_container(self) -> None:
        with pytest.raises(ValueError):
            AzureBlobStore.from_account("https://a.blob.core.windows.net", "Bad_Name")

    def test_not_a_container_client(self) -> None:
        with pytest.raises(ValueError):
            AzureBlobStore(object())

    def test_missing_sdk_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("azure.storage"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ImportError, match=r"traigent\[snapshot-azure\]"):
            AzureBlobStore.from_account("https://a.blob.core.windows.net", "cont")
