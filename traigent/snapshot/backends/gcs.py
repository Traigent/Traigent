"""Google Cloud Storage snapshot store using generation-match conditional writes."""

from __future__ import annotations

import re
from typing import Any

from traigent.snapshot.errors import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.store import CreateResult, require_secure_endpoint, validate_key

__all__ = ["GCSStore"]

_BACKEND = "gcs"
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]")
_TIMEOUT_SECONDS = 30
_PRECONDITION_STATUS = frozenset({412})
_NOT_FOUND_STATUS = frozenset({404})
_AUTH_STATUS = frozenset({401, 403})
_IMPORT_HINT = (
    'GCSStore needs google-cloud-storage: pip install "traigent[snapshot-gcs]" '
    "(or pass your own storage client)"
)


def _error(code: SnapshotErrorCode) -> SnapshotStoreError:
    return SnapshotStoreError(code, backend=_BACKEND)


def _status(error: BaseException) -> int | None:
    """The HTTP status of a ``google.api_core`` error, read structurally."""
    code = getattr(error, "code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _is_auth_failure(error: BaseException) -> bool:
    if _status(error) in _AUTH_STATUS:
        return True
    return any(cls.__module__.startswith("google.auth") for cls in type(error).__mro__)


def _translate(error: BaseException) -> SnapshotStoreError:
    if _is_auth_failure(error):
        return _error(SnapshotErrorCode.AUTH)
    if _status(error) in _NOT_FOUND_STATUS:
        return _error(SnapshotErrorCode.NOT_FOUND)
    return _error(SnapshotErrorCode.BACKEND_UNAVAILABLE)


class GCSStore:
    """Snapshot objects under ``gs://<bucket>/<prefix>``.

    Atomicity comes from ``upload_from_string(if_generation_match=0)``: GCS
    answers 412 when the object already exists, and the store then reads and
    compares the bytes. A custom API endpoint (an emulator, set on an injected
    ``client``) must be HTTPS unless ``allow_insecure_local`` and loopback.

    Credentials stay inside the storage client; ``repr`` is the locator.
    """

    max_object_bytes: int | None = None

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client: Any = None,
        allow_insecure_local: bool = False,
    ) -> None:
        if not isinstance(bucket, str) or _BUCKET_RE.fullmatch(bucket) is None:
            raise ValueError("invalid GCS bucket name")
        clean = prefix.strip("/") if isinstance(prefix, str) else ""
        if clean:
            validate_key(clean)
        self._bucket_name = bucket
        self._prefix = clean
        self._client = client if client is not None else self._make_client()
        self._check_endpoint(allow_insecure_local)
        self._bucket = self._client.bucket(bucket)

    @staticmethod
    def _make_client() -> Any:
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(_IMPORT_HINT) from None
        try:
            return storage.Client()
        except Exception as error:
            raise _translate(error) from None

    def _check_endpoint(self, allow_insecure_local: bool) -> None:
        connection = getattr(self._client, "_connection", None)
        endpoint = getattr(connection, "API_BASE_URL", None)
        if not isinstance(endpoint, str) or not endpoint:
            # Unknown destination: refuse rather than guess it is HTTPS.
            raise _error(SnapshotErrorCode.UNSAFE_DESTINATION) from None
        require_secure_endpoint(
            endpoint, allow_insecure_local=allow_insecure_local, backend=_BACKEND
        )

    def __repr__(self) -> str:
        return self.locator()

    def locator(self) -> str:
        return (
            f"gs://{self._bucket_name}/{self._prefix}"
            if self._prefix
            else f"gs://{self._bucket_name}"
        )

    def _full(self, key: str) -> str:
        validate_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    def read(self, key: str) -> bytes | None:
        full = self._full(key)
        try:
            return bytes(
                self._bucket.blob(full).download_as_bytes(timeout=_TIMEOUT_SECONDS)
            )
        except Exception as error:
            if _status(error) in _NOT_FOUND_STATUS and not _is_auth_failure(error):
                return None
            raise _translate(error) from None

    def create_if_absent(self, key: str, data: bytes) -> CreateResult:
        full = self._full(key)
        data = bytes(data)
        try:
            self._bucket.blob(full).upload_from_string(
                data, if_generation_match=0, timeout=_TIMEOUT_SECONDS
            )
            return CreateResult.CREATED
        except Exception as error:
            if _status(error) not in _PRECONDITION_STATUS:
                raise _translate(error) from None
        existing = self.read(key)
        if existing is None:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        if existing == data:
            return CreateResult.EXISTED_IDENTICAL
        raise _error(SnapshotErrorCode.CONFLICT) from None
