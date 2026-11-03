"""Hand-written S3 client fake (botocore's real ClientError, no moto)."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

from botocore.exceptions import ClientError


def client_error(code: str, status: int = 400, message: str = "") -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "Op",
    )


class FakeS3Client:
    """put_object(IfNoneMatch) / get_object / delete_object over a dict."""

    def __init__(
        self,
        endpoint_url: str = "https://s3.us-east-1.amazonaws.com",
        *,
        honour_if_none_match: bool = True,
    ) -> None:
        self.meta = SimpleNamespace(endpoint_url=endpoint_url)
        self.objects: dict[tuple[str, str], bytes] = {}
        self.honour = honour_if_none_match
        self.put_calls: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        self.put_calls.append(kwargs)
        slot = (kwargs["Bucket"], kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and self.honour and slot in self.objects:
            raise client_error("PreconditionFailed", 412)
        self.objects[slot] = bytes(kwargs["Body"])
        return {}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        slot = (kwargs["Bucket"], kwargs["Key"])
        if slot not in self.objects:
            raise client_error("NoSuchKey", 404)
        return {"Body": io.BytesIO(self.objects[slot])}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.objects.pop((kwargs["Bucket"], kwargs["Key"]), None)
        return {}


# --------------------------------------------------------------------- GCS
# Real google.api_core exceptions, hand-written client (no emulator).


class FakeGCSBlob:
    def __init__(self, client: FakeGCSClient, name: str) -> None:
        self._client = client
        self._name = name

    def upload_from_string(
        self, data: bytes, *, if_generation_match: int | None = None, **kwargs: Any
    ) -> None:
        from google.api_core import exceptions

        client = self._client
        if client.fail_with is not None:
            raise client.fail_with
        client.upload_calls.append(
            {"name": self._name, "if_generation_match": if_generation_match, **kwargs}
        )
        if if_generation_match == 0 and self._name in client.objects and client.honour:
            raise exceptions.PreconditionFailed("object exists")
        client.objects[self._name] = bytes(data)

    def download_as_bytes(self, **_kwargs: Any) -> bytes:
        from google.api_core import exceptions

        if self._client.fail_with is not None:
            raise self._client.fail_with
        if self._name not in self._client.objects:
            raise exceptions.NotFound("no such object")
        return self._client.objects[self._name]


class FakeGCSBucket:
    def __init__(self, client: FakeGCSClient) -> None:
        self._client = client

    def blob(self, name: str) -> FakeGCSBlob:
        return FakeGCSBlob(self._client, name)


class FakeGCSClient:
    def __init__(
        self,
        api_base: str = "https://storage.googleapis.com",
        *,
        honour: bool = True,
    ) -> None:
        self._connection = SimpleNamespace(API_BASE_URL=api_base)
        self.objects: dict[str, bytes] = {}
        self.upload_calls: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None
        self.honour = honour

    def bucket(self, _name: str) -> FakeGCSBucket:
        return FakeGCSBucket(self)


# ------------------------------------------------------------------- Azure


class FakeAzureDownloader:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class FakeAzureContainer:
    """Mimics ContainerClient: ``url`` may carry a SAS query, ``upload_blob`` refuses overwrite."""

    def __init__(
        self,
        url: str = "https://acct.blob.core.windows.net/cont",
        container_name: str = "cont",
    ) -> None:
        self.url = url
        self.container_name = container_name
        self.objects: dict[str, bytes] = {}
        self.upload_calls: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None

    def upload_blob(
        self, name: str, data: bytes, *, overwrite: bool = False, **kwargs: Any
    ) -> None:
        from azure.core.exceptions import ResourceExistsError

        if self.fail_with is not None:
            raise self.fail_with
        self.upload_calls.append({"name": name, "overwrite": overwrite, **kwargs})
        if name in self.objects and not overwrite:
            raise ResourceExistsError("blob exists")
        self.objects[name] = bytes(data)

    def download_blob(self, name: str, **_kwargs: Any) -> FakeAzureDownloader:
        from azure.core.exceptions import ResourceNotFoundError

        if self.fail_with is not None:
            raise self.fail_with
        if name not in self.objects:
            raise ResourceNotFoundError("no such blob")
        return FakeAzureDownloader(self.objects[name])
