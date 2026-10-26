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
