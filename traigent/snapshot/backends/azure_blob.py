"""Azure Blob Storage snapshot store using ``overwrite=False`` creates."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from traigent.snapshot.errors import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.store import CreateResult, require_secure_endpoint, validate_key

__all__ = ["AzureBlobStore"]

_BACKEND = "azure_blob"
_CONTAINER_RE = re.compile(r"[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9])){1,61}[a-z0-9]")
_TIMEOUT_SECONDS = 30
_AUTH_STATUS = frozenset({401, 403})
_NOT_FOUND_STATUS = frozenset({404})
_EXISTS_STATUS = frozenset({409})
_IMPORT_HINT = (
    "AzureBlobStore.from_account needs azure-storage-blob: pip install "
    '"traigent[snapshot-azure]" (or pass your own ContainerClient)'
)


def _error(code: SnapshotErrorCode) -> SnapshotStoreError:
    return SnapshotStoreError(code, backend=_BACKEND)


def _class_names(error: BaseException) -> frozenset[str]:
    return frozenset(cls.__name__ for cls in type(error).__mro__)


def _status(error: BaseException) -> int | None:
    code = getattr(error, "status_code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _is_auth(error: BaseException) -> bool:
    return (
        "ClientAuthenticationError" in _class_names(error)
        or _status(error) in _AUTH_STATUS
    )


def _is_not_found(error: BaseException) -> bool:
    return not _is_auth(error) and (
        "ResourceNotFoundError" in _class_names(error)
        or _status(error) in _NOT_FOUND_STATUS
    )


def _is_exists(error: BaseException) -> bool:
    return not _is_auth(error) and (
        "ResourceExistsError" in _class_names(error) or _status(error) in _EXISTS_STATUS
    )


def _translate(error: BaseException) -> SnapshotStoreError:
    if _is_auth(error):
        return _error(SnapshotErrorCode.AUTH)
    if _is_not_found(error):
        return _error(SnapshotErrorCode.NOT_FOUND)
    return _error(SnapshotErrorCode.BACKEND_UNAVAILABLE)


class AzureBlobStore:
    """Snapshot objects under ``azblob://<account host>/<container>/<prefix>``.

    Atomicity comes from ``upload_blob(name, data, overwrite=False)``: an
    existing blob raises ``ResourceExistsError`` and the store then reads and
    compares the bytes. The locator is built from the account host, container
    and prefix only; a SAS token or any other query string, and any userinfo,
    is dropped and never kept on the store. Credentials stay inside the
    ``ContainerClient``; ``repr`` is the locator.
    """

    max_object_bytes: int | None = None

    def __init__(
        self,
        container_client: Any,
        prefix: str = "",
        *,
        allow_insecure_local: bool = False,
    ) -> None:
        clean = prefix.strip("/") if isinstance(prefix, str) else ""
        if clean:
            validate_key(clean)
        url = getattr(container_client, "url", None)
        container = getattr(container_client, "container_name", None)
        if not isinstance(url, str) or not isinstance(container, str):
            raise ValueError("container_client must be an azure ContainerClient")
        require_secure_endpoint(
            url, allow_insecure_local=allow_insecure_local, backend=_BACKEND
        )
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if ":" in host:
            host = f"[{host}]"
        netloc = f"{host}:{parts.port}" if parts.port else host
        self._container = container_client
        self._prefix = clean
        self._locator = f"azblob://{netloc}/{container}" + (
            f"/{clean}" if clean else ""
        )

    @classmethod
    def from_account(
        cls,
        account_url: str,
        container: str,
        credential: Any = None,
        prefix: str = "",
        *,
        allow_insecure_local: bool = False,
    ) -> AzureBlobStore:
        if not isinstance(container, str) or _CONTAINER_RE.fullmatch(container) is None:
            raise ValueError("invalid Azure container name")
        require_secure_endpoint(
            account_url, allow_insecure_local=allow_insecure_local, backend=_BACKEND
        )
        try:
            from azure.storage.blob import ContainerClient
        except ImportError:
            raise ImportError(_IMPORT_HINT) from None
        client = ContainerClient(account_url, container, credential=credential)
        return cls(client, prefix, allow_insecure_local=allow_insecure_local)

    def __repr__(self) -> str:
        return self.locator()

    def locator(self) -> str:
        return self._locator

    def _full(self, key: str) -> str:
        validate_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    def read(self, key: str) -> bytes | None:
        full = self._full(key)
        try:
            return bytes(
                self._container.download_blob(full, timeout=_TIMEOUT_SECONDS).readall()
            )
        except Exception as error:
            if _is_not_found(error):
                return None
            raise _translate(error) from None

    def create_if_absent(self, key: str, data: bytes) -> CreateResult:
        full = self._full(key)
        data = bytes(data)
        try:
            self._container.upload_blob(
                full, data, overwrite=False, timeout=_TIMEOUT_SECONDS
            )
            return CreateResult.CREATED
        except Exception as error:
            if not _is_exists(error):
                raise _translate(error) from None
        existing = self.read(key)
        if existing is None:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        if existing == data:
            return CreateResult.EXISTED_IDENTICAL
        raise _error(SnapshotErrorCode.CONFLICT) from None
