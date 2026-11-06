"""Open a snapshot store from its locator string.

The closed set of locator forms the backends emit (``repr(store)``):
``s3://bucket[/prefix]``, ``gs://bucket[/prefix]``,
``azblob://host[:port]/container[/prefix]``,
``github+v2://owner/repo@branch[/prefix]`` and ``file://<path>`` (a plain path is
also accepted). GitHub grammar: the branch is percent-encoded with no raw "/" (a
"/" in the branch is ``%2F``; no other escape is accepted), followed by an optional
"/" and the prefix. The unversioned ``github://owner/repo@branch`` is accepted only
without a prefix; with a prefix it cannot be told apart from an old locator whose
branch held a raw "/" (``@feature/fix/snaps``), so it is refused with
``SNAPSHOT_LOCATOR_AMBIGUOUS_BRANCH`` and the user re-issues it as ``github+v2://``
with "/" in the branch written ``%2F``. A GitHub Enterprise store leads with its
percent-encoded API URL as a third segment
(``github+v2://https%3A%2F%2Fghe.example.com%2Fapi%2Fv3/owner/repo@branch``); the
default ``https://api.github.com`` is omitted, so github.com locators are unchanged.
A ``file://`` path and a plain path are both taken literally (never
percent-decoded) and may contain ``?`` or ``#``; ``LocalDirStore`` emits the plain
path for such names. The API URL in a GitHub Enterprise locator is only honoured when
it exactly equals the operator-configured ``TRAIGENT_GITHUB_API_URL`` (a locator can
never redirect ``GITHUB_TOKEN`` to a host the operator did not name); otherwise the
open is refused with ``SNAPSHOT_LOCATOR_API_URL_NOT_CONFIGURED``. Credentials are never
part of a locator: userinfo, a query or a fragment is refused for every URL scheme
(including inside a decoded GitHub API URL) before any credential lookup; each backend finds them through the vendor's standard chain or
environment (boto3, google-auth ADC, azure-identity, ``GITHUB_TOKEN``).
Failures carry a closed code only; the offending value is never echoed.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import unquote, urlsplit

__all__ = ["LocatorErrorCode", "SnapshotLocatorError", "open_store_from_locator"]

_GITHUB_API_URL_ENV = "TRAIGENT_GITHUB_API_URL"
_SCHEMES = ("s3", "gs", "azblob", "github", "github+v2", "file")
_EXTRA_HINT = {
    "s3": "pip install 'traigent[snapshot-s3]'",
    "gs": "pip install 'traigent[snapshot-gcs]'",
    "azblob": "pip install 'traigent[snapshot-azure]'",
}


class LocatorErrorCode:
    UNKNOWN_SCHEME = "SNAPSHOT_LOCATOR_UNKNOWN_SCHEME"
    MALFORMED = "SNAPSHOT_LOCATOR_MALFORMED"
    MISSING_EXTRA = "SNAPSHOT_LOCATOR_MISSING_EXTRA"
    CREDENTIALS_IN_LOCATOR = "SNAPSHOT_LOCATOR_CREDENTIALS_NOT_ALLOWED"
    AMBIGUOUS_BRANCH = "SNAPSHOT_LOCATOR_AMBIGUOUS_BRANCH"
    API_URL_NOT_CONFIGURED = "SNAPSHOT_LOCATOR_API_URL_NOT_CONFIGURED"
    STORE_UNAVAILABLE = "SNAPSHOT_LOCATOR_STORE_UNAVAILABLE"


class SnapshotLocatorError(Exception):
    """A locator could not be opened. Holds a closed code and a fixed message."""

    def __init__(self, code: str, hint: str | None = None) -> None:
        self.code = code
        self.hint = hint
        super().__init__(code if hint is None else f"{code} ({hint})")


def _split_path(rest: str) -> tuple[str, str]:
    head, _, tail = rest.partition("/")
    return head, unquote(tail).strip("/")


def _refuse_credential_syntax(scheme: str, rest: str, has_scheme: bool) -> None:
    """Locators are credential-free: no userinfo, query or fragment, any scheme."""
    if scheme == "file":
        # A filesystem path (plain or file://) has no URL query/fragment/userinfo:
        # "?" and "#" are legal characters of a directory name.
        if has_scheme and "@" in rest.partition("/")[0]:
            raise SnapshotLocatorError(
                LocatorErrorCode.CREDENTIALS_IN_LOCATOR
            ) from None
        return
    if "?" in rest or "#" in rest:
        raise SnapshotLocatorError(LocatorErrorCode.CREDENTIALS_IN_LOCATOR) from None
    if has_scheme and not scheme.startswith("github") and "@" in rest.partition("/")[0]:
        raise SnapshotLocatorError(LocatorErrorCode.CREDENTIALS_IN_LOCATOR) from None
    head, slash, _ = rest.partition("/")
    if scheme.startswith("github") and (":" in head or (slash and "@" in head)):
        raise SnapshotLocatorError(LocatorErrorCode.CREDENTIALS_IN_LOCATOR) from None


def open_store_from_locator(locator: str, *, allow_insecure_local: bool = False) -> Any:
    """Return the snapshot store a locator names, or raise ``SnapshotLocatorError``.

    Transport is secure by default; ``allow_insecure_local=True`` (explicit opt-in,
    local emulators only) lets loopback endpoints use plain HTTP.
    """
    from traigent.snapshot.backends import GCSStore, LocalDirStore, S3Store
    from traigent.snapshot.errors import SnapshotStoreError

    if not isinstance(locator, str) or not locator:
        raise SnapshotLocatorError(LocatorErrorCode.MALFORMED) from None
    if "://" not in locator:
        scheme, rest = "file", locator
        _refuse_credential_syntax(scheme, rest, False)
    else:
        scheme, _, rest = locator.partition("://")
        scheme = scheme.lower()
        if scheme not in _SCHEMES:
            raise SnapshotLocatorError(LocatorErrorCode.UNKNOWN_SCHEME) from None
        _refuse_credential_syntax(scheme, rest, True)
    try:
        if scheme == "file":
            if not rest:
                raise ValueError
            return LocalDirStore(rest)
        if scheme in ("s3", "gs"):
            bucket, prefix = _split_path(rest)
            cls = S3Store if scheme == "s3" else GCSStore
            return cls(bucket, prefix, allow_insecure_local=allow_insecure_local)
        if scheme == "azblob":
            return _open_azblob(rest, allow_insecure_local)
        return _open_github(scheme, rest, allow_insecure_local)
    except SnapshotLocatorError:
        raise
    except ImportError:
        raise SnapshotLocatorError(
            LocatorErrorCode.MISSING_EXTRA, _EXTRA_HINT.get(scheme)
        ) from None
    except ValueError:
        raise SnapshotLocatorError(LocatorErrorCode.MALFORMED) from None
    except SnapshotStoreError as err:
        raise SnapshotLocatorError(
            LocatorErrorCode.STORE_UNAVAILABLE, f"code={err.code.value}"
        ) from None


def _open_azblob(rest: str, allow_insecure_local: bool) -> Any:
    from traigent.snapshot.backends import AzureBlobStore

    host, tail = _split_path(rest)
    container, _, prefix = tail.partition("/")
    if not host or not container:
        raise ValueError
    # Parse the authority only; a bare "//host" yields the same hostname.
    loopback = (urlsplit(f"//{host}").hostname or "") in (
        "127.0.0.1",
        "localhost",
        "::1",
    )
    plain = allow_insecure_local and loopback
    account_url = f"{'http' if plain else 'https'}://{host}"
    return AzureBlobStore.from_account(
        account_url,
        container,
        _azure_credential(),
        prefix,
        allow_insecure_local=allow_insecure_local,
    )


def _open_github(scheme: str, rest: str, allow_insecure_local: bool) -> Any:
    from traigent.snapshot.backends import GitHubStore

    # github+v2://owner/repo@branch[/prefix]; github://owner/repo@branch only
    owner_repo_branch, prefix = _split_path_github(rest)
    if scheme == "github" and prefix:
        raise SnapshotLocatorError(
            LocatorErrorCode.AMBIGUOUS_BRANCH,
            "use github+v2:// and write '/' in the branch as %2F",
        ) from None
    repo, _, branch = owner_repo_branch.rpartition("@")
    branch = branch.replace("%2F", "/").replace("%2f", "/")
    if not repo or not branch or "%" in branch:
        raise ValueError
    segments = repo.split("/")
    if len(segments) == 2:
        return GitHubStore(repo, branch, prefix)
    if len(segments) != 3:
        raise ValueError
    api_url = unquote(segments[0])
    parts = urlsplit(api_url)
    if parts.username or parts.password or parts.query or parts.fragment:
        raise SnapshotLocatorError(LocatorErrorCode.CREDENTIALS_IN_LOCATOR) from None
    configured = os.environ.get(_GITHUB_API_URL_ENV, "").strip().rstrip("/")
    if not configured or api_url.rstrip("/") != configured:
        raise SnapshotLocatorError(LocatorErrorCode.API_URL_NOT_CONFIGURED) from None
    return GitHubStore(
        "/".join(segments[1:]),
        branch,
        prefix,
        api_url=api_url,
        allow_insecure_local=allow_insecure_local,
    )


def _split_path_github(rest: str) -> tuple[str, str]:
    # owner/repo@branch[/prefix]: "/" inside the branch is written %2F, so the
    # branch ends at the first "/" after the "@".
    at = rest.find("@")
    if at < 0:
        raise ValueError
    slash = rest.find("/", at)
    if slash < 0:
        return rest, ""
    return rest[:slash], unquote(rest[slash + 1 :]).strip("/")


def _azure_credential() -> Any:
    """azure-identity's standard chain; fail closed when it is not installed."""
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise SnapshotLocatorError(
            LocatorErrorCode.MISSING_EXTRA, "pip install azure-identity"
        ) from None
    return DefaultAzureCredential()
