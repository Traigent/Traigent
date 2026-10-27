"""S3 (and S3-compatible) snapshot store using conditional writes."""

from __future__ import annotations

import re
import uuid
from typing import Any
from urllib.parse import urlsplit

from traigent.snapshot.errors import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.store import CreateResult, require_secure_endpoint, validate_key

__all__ = ["S3Store"]

_BACKEND = "s3"
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_MAX_PUT_BYTES = 5 * 1024**3  # single PutObject limit
_AWS_SUFFIXES = (".amazonaws.com", ".amazonaws.com.cn")
# Existing object (create_if_absent): read it and compare. 409 is a transient
# conflict, but the answer is still "look at what is there".
_PRECONDITION_CODES = frozenset(
    {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}
)
# Capability probe: only a failed precondition proves IfNoneMatch is enforced.
_PROBE_PROOF_CODES = frozenset({"PreconditionFailed", "412"})
_NOT_FOUND_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
_AUTH_CODES = frozenset(
    {
        "AccessDenied",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "ExpiredToken",
        "InvalidToken",
        "403",
    }
)
_UNSUPPORTED_CODES = frozenset({"NotImplemented", "501", "InvalidArgument"})
_IMPORT_HINT = (
    'S3Store needs boto3: pip install "traigent[snapshot-s3]" '
    "(or pass your own S3 client)"
)


def _error(code: SnapshotErrorCode) -> SnapshotStoreError:
    return SnapshotStoreError(code, backend=_BACKEND)


def _error_code(error: BaseException) -> str:
    """The provider's error code, read structurally (botocore is not imported)."""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        details = response.get("Error")
        if isinstance(details, dict) and isinstance(details.get("Code"), str):
            return str(details["Code"])
        meta = response.get("ResponseMetadata")
        if isinstance(meta, dict) and meta.get("HTTPStatusCode") is not None:
            return str(meta["HTTPStatusCode"])
    return ""


def _translate(error: BaseException) -> SnapshotStoreError:
    code = _error_code(error)
    if code in _AUTH_CODES:
        return _error(SnapshotErrorCode.AUTH)
    if code in _UNSUPPORTED_CODES:
        return _error(SnapshotErrorCode.UNSUPPORTED_BACKEND)
    if code == "NoSuchBucket":
        return _error(SnapshotErrorCode.NOT_FOUND)
    return _error(SnapshotErrorCode.BACKEND_UNAVAILABLE)


class S3Store:
    """Snapshot objects under ``s3://<bucket>/<prefix>``.

    Atomicity comes from ``PutObject(IfNoneMatch="*")``. AWS endpoints are
    trusted to honour it; a custom ``endpoint_url`` (set only on an injected
    ``client``) is probed once before the first write and refused with
    ``UNSUPPORTED_BACKEND`` if it accepts a second unconditional create.

    Credentials stay inside the boto3 client; ``repr`` is the locator.
    """

    max_object_bytes: int | None = _MAX_PUT_BYTES

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client: Any = None,
        session: Any = None,
        allow_insecure_local: bool = False,
    ) -> None:
        if not isinstance(bucket, str) or _BUCKET_RE.fullmatch(bucket) is None:
            raise ValueError("invalid S3 bucket name")
        clean = prefix.strip("/") if isinstance(prefix, str) else ""
        if clean:
            validate_key(clean)  # same rules as keys: no "..", no empty segments
        self._bucket = bucket
        self._prefix = clean
        self._client = client if client is not None else self._make_client(session)
        self._custom_endpoint = False
        self._probed = False
        self._check_endpoint(allow_insecure_local)

    @staticmethod
    def _make_client(session: Any) -> Any:
        if session is not None:
            try:
                return session.client("s3")
            except Exception as error:
                raise _translate(error) from None
        try:
            import boto3
        except ImportError:
            raise ImportError(_IMPORT_HINT) from None
        try:
            return boto3.client("s3")
        except Exception as error:
            raise _translate(error) from None

    def _check_endpoint(self, allow_insecure_local: bool) -> None:
        meta = getattr(self._client, "meta", None)
        endpoint = getattr(meta, "endpoint_url", None)
        if not isinstance(endpoint, str) or not endpoint:
            # Unknown destination: refuse rather than guess it is HTTPS.
            raise _error(SnapshotErrorCode.UNSAFE_DESTINATION) from None
        require_secure_endpoint(
            endpoint, allow_insecure_local=allow_insecure_local, backend=_BACKEND
        )
        host = (urlsplit(endpoint).hostname or "").lower()
        self._custom_endpoint = not host.endswith(_AWS_SUFFIXES)

    def __repr__(self) -> str:
        return self.locator()

    def locator(self) -> str:
        return (
            f"s3://{self._bucket}/{self._prefix}"
            if self._prefix
            else f"s3://{self._bucket}"
        )

    def _full(self, key: str) -> str:
        validate_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    def _put(self, full_key: str, data: bytes) -> None:
        self._client.put_object(
            Bucket=self._bucket, Key=full_key, Body=data, IfNoneMatch="*"
        )

    def _probe(self) -> None:
        """Refuse a custom endpoint that does not enforce ``IfNoneMatch``."""
        if self._probed or not self._custom_endpoint:
            return
        key = f"{self._prefix}/.probe/{uuid.uuid4().hex}".lstrip("/")
        try:
            self._put(key, b"probe")
        except Exception as error:
            raise _translate(error) from None
        try:
            self._put(key, b"probe")
        except Exception as error:
            if _error_code(error) not in _PROBE_PROOF_CODES:
                self._best_effort_delete(key)  # our first PUT created this key
                raise _translate(error) from None
            self._probed = True
            self._best_effort_delete(key)
            return
        self._best_effort_delete(key)
        raise _error(SnapshotErrorCode.UNSUPPORTED_BACKEND)

    def _best_effort_delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except Exception:
            return

    def read(self, key: str) -> bytes | None:
        full = self._full(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=full)
            body = response["Body"]
            try:
                return bytes(body.read())
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
        except Exception as error:
            if _error_code(error) in _NOT_FOUND_CODES:
                return None
            raise _translate(error) from None

    def create_if_absent(self, key: str, data: bytes) -> CreateResult:
        full = self._full(key)
        data = bytes(data)
        if len(data) > _MAX_PUT_BYTES:
            raise _error(SnapshotErrorCode.TOO_LARGE)
        self._probe()
        try:
            self._put(full, data)
            return CreateResult.CREATED
        except Exception as error:
            if _error_code(error) not in _PRECONDITION_CODES:
                raise _translate(error) from None
        existing = self.read(key)
        if existing is None:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE)
        if existing == data:
            return CreateResult.EXISTED_IDENTICAL
        raise _error(SnapshotErrorCode.CONFLICT)
