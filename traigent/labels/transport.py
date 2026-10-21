"""Label-service transport: the SDK side of the Backend ``/api/v1/labels/*`` routes.

Only the routes a label WRITER needs: ``GET /labels/binding-list``, ``PUT /labels/objects``
(``base_revision`` + ``epoch``) and ``POST /labels/objects/read``. When the token store is
enabled (``TRAIGENT_TOKEN_STORE_ENABLED``) writes go instead to the negotiated route
``POST /optimization/labels`` (content-less round 1b-2b; body = the same write plus
``schema_version``), which refuses a ``object_ref`` the Backend did not issue. A name is sealed to the
project public key by :class:`~traigent.labels.writer.LabelWriter` only after the signed
binding chain verifies against the customer's pinned trust anchor, so an unconfigured or
untrusted setup sends nothing (labels are simply not sent). Requests go through the gated
aiohttp session and the caller's authenticated headers. Failures carry closed codes only;
no label text, name or Backend body is ever put in an exception or a log line.

Config (``traigent.yml``)::

    labels:
      trust_anchor:
        root_fingerprint: "<64 hex>"
        tenant_ref: "tenant_<32 hex>"
        project_ref: "project_<32 hex>"
        root_public_key: "<base64url of the raw 32-byte Ed25519 root key>"
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import yaml

from traigent.utils.logging import get_logger

from . import token_store
from ._codec import unb64u
from .transport_gate import TransportGateRefusal
from .trust_anchor import TrustAnchor, TrustAnchorError, find_config, load_trust_anchor
from .writer import LabelWriter

logger = get_logger(__name__)

HeadersProvider = Callable[[], Awaitable[Mapping[str, str]]]

#: Closed transport codes (the Backend's contract error codes plus two local ones).
CLOSED_CODES = frozenset(
    {
        "invalid_request",
        "flag_disabled",
        "not_found",
        "foreign_scope",
        "stale_revision",
        "stale_epoch",
        "stale_sequence",
        "conflict",
        "bad_signature",
        "unknown_signer",
        "internal_error",
    }
)
UNAVAILABLE = "unavailable"
BAD_RESPONSE = "bad_response"
#: Closed codes of the negotiated route (``POST /optimization/labels``, Schema
#: ``label_write_schema.json``) beyond the shared ones above.
REFERENCE_NOT_ISSUED = "reference_not_issued"
SERVICE_UNAVAILABLE = "service_unavailable"
#: The route answered the uniform unknown-route 404 (it is dark). Never falls back to the
#: legacy route: that would write an UNVALIDATED ref while the caller asked for validation.
LABEL_ROUTE_UNAVAILABLE = "label_route_unavailable"
#: ``stale_revision`` on a token's first write and no approved reader to compare (S1).
LABEL_CONFLICT_UNVERIFIED = token_store.LABEL_CONFLICT_UNVERIFIED
#: The write was acknowledged but the token journal could not record it (S1 stays pending).
TOKEN_STATE_NOT_PERSISTED = "token_state_not_persisted"
#: A label targets a token-kind ref the selected token journal does not know (issued under
#: another namespace, or no journal given): nothing is sent, so no token is left un-advanced.
TOKEN_REF_UNKNOWN = "token_ref_unknown"
UNAUTHENTICATED = "unauthenticated"
FORBIDDEN = "forbidden"
RATE_LIMITED = "rate_limited"
INTERNAL_ERROR = "internal_error"
RETRYABLE_CODES = frozenset({SERVICE_UNAVAILABLE})
_TIMEOUT_SECONDS = 15
_MAX_BATCH = 200
#: Everything a negotiated write does (retries included) ends by this many seconds.
_WRITE_DEADLINE_SECONDS = token_store.OPERATION_DEADLINE_SECONDS
_MAX_WRITE_ATTEMPTS = token_store.MAX_SERVICE_UNAVAILABLE_ATTEMPTS
_RETRY_BACKOFF_SECONDS = token_store.RETRY_BACKOFF_SECONDS
_sleep = asyncio.sleep  # indirection so tests can run the pacing without waiting


class LabelTransportError(Exception):
    """A label-service call failed. ``code`` is closed; nothing of the exchange is kept."""

    def __init__(self, code: str, status: int | None = None) -> None:
        self.code = code
        self.status = status
        super().__init__(code)

    @property
    def retryable(self) -> bool:
        """True when the same call may be tried again later (``service_unavailable``)."""
        return self.code in RETRYABLE_CODES


def _data(body: Any) -> dict[str, Any]:
    if isinstance(body, dict) and isinstance(body.get("data"), dict):
        return dict(body["data"])
    if isinstance(body, dict):
        return body
    raise LabelTransportError(BAD_RESPONSE)


def _retry_after_seconds(headers: Any) -> float | None:
    """``Retry-After`` as delta-seconds in ``[0, 3600]``; ``None`` for anything else."""
    try:
        value = headers.get("Retry-After") if headers else None
        seconds = float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, 3600.0)


# ---------------------------------------------------- negotiated route (content-less 1b-2b)

#: Closed refusal envelope of the negotiated route (Schema ``LabelWriteRefusal``):
#: code -> (HTTP status the Schema pairs with it, fixed message). Pinned against the Schema by
#: ``tests/unit/labels/test_negotiated_label_write.py``.
_NEGOTIATED_REFUSALS: dict[str, tuple[int, str]] = {
    "invalid_request": (400, "Request is invalid."),
    REFERENCE_NOT_ISSUED: (422, "Reference was not issued."),
    "stale_revision": (409, "Object revision is not current."),
    "stale_epoch": (409, "Key epoch is not current."),
    "conflict": (409, "Request conflicts with stored state."),
    INTERNAL_ERROR: (500, "Internal error."),
    SERVICE_UNAVAILABLE: (503, "Service is temporarily unavailable."),
}
_ENVELOPE_KEYS = frozenset({"success", "error", "error_code", "message"})
_RESPONSE_KEYS = frozenset({"object_ref", "purpose", "revision", "epoch", "updated_at"})
_OBJECT_REF = re.compile(
    r"(agent|agentbuild|choice|configurationversion|dataset|datasetversion|evaluationset"
    r"|evaluator|evaluatorversion|head|knob|metric|run|searchspec|snapshot|template|trial)"
    r"ref_[0-9a-f]{32}"
)
_PURPOSES = frozenset({"label", "knob", "choice", "metric", "template"})
_ENC = re.compile(r"[A-Za-z0-9_-]{43}")
#: base64url length of a 48 / 80 / 144 / 272 byte ciphertext (padding bucket + 16-byte tag)
_CIPHERTEXT = re.compile(r"[A-Za-z0-9_-]+")
_CIPHERTEXT_LENGTHS = frozenset({64, 107, 192, 363})
_MAX_REVISION = 2**53 - 1
_MAX_EPOCH = 2**32 - 1


def _int(value: Any, low: int, high: int) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and low <= value <= high
    )


def _negotiated_body(
    object_ref: Any,
    purpose: Any,
    base_revision: Any,
    epoch: Any,
    envelope: Any,
) -> dict[str, Any]:
    """The closed request of ``POST /optimization/labels``, as a private plain-data snapshot.

    Built from exact ``str`` / ``int`` / ``dict`` values only, copied out of the caller's
    objects (so nothing can change between validation and send) and checked against the Schema
    shape: six fields, ``schema_version`` 1, an opaque ``<kind>ref_<32 hex>`` reference, a
    closed purpose, integer revision and epoch in range, and an envelope that is exactly
    ``version`` / ``key_epoch`` / ``enc`` / ``ciphertext`` with ``key_epoch == epoch``. Any
    other value (in particular a plain name) is refused locally as ``invalid_request`` with
    fixed text: nothing is sent and the offending value is never quoted.
    """
    if not isinstance(envelope, dict) or any(type(k) is not str for k in envelope):
        raise LabelTransportError("invalid_request")
    env = {k: envelope[k] for k in list(envelope)}
    if (
        type(object_ref) is not str
        or not _OBJECT_REF.fullmatch(object_ref)
        or type(purpose) is not str
        or purpose not in _PURPOSES
        or type(base_revision) is not int
        or not _int(base_revision, 0, _MAX_REVISION)
        or type(epoch) is not int
        or not _int(epoch, 1, _MAX_EPOCH)
        or set(env) != {"version", "key_epoch", "enc", "ciphertext"}
        or type(env["version"]) is not int
        or env["version"] != 1
        or type(env["key_epoch"]) is not int
        or env["key_epoch"] != epoch
        or type(env["enc"]) is not str
        or not _ENC.fullmatch(env["enc"])
        or type(env["ciphertext"]) is not str
        or len(env["ciphertext"]) not in _CIPHERTEXT_LENGTHS
        or not _CIPHERTEXT.fullmatch(env["ciphertext"])
    ):
        raise LabelTransportError("invalid_request")
    return {
        "schema_version": 1,
        "object_ref": str(object_ref),
        "purpose": str(purpose),
        "base_revision": int(base_revision),
        "epoch": int(epoch),
        "envelope": {
            "version": 1,
            "key_epoch": int(env["key_epoch"]),
            "enc": str(env["enc"]),
            "ciphertext": str(env["ciphertext"]),
        },
    }


def _negotiated_refusal_code(status: int, body: Any) -> str:
    """The closed SDK code of a non-200 answer from the negotiated route; nothing is kept."""
    if isinstance(body, dict) and set(body) == _ENVELOPE_KEYS:
        raw = body.get("error_code")
        spec = _NEGOTIATED_REFUSALS.get(raw) if isinstance(raw, str) else None
        if (
            spec is not None
            and body.get("success") is False
            and body.get("error") == raw
            and body.get("message") == spec[1]
            and status == spec[0]
        ):
            return str(raw)
    if status == 404:
        return (
            LABEL_ROUTE_UNAVAILABLE  # the uniform unknown-route 404: the route is dark
        )
    if status == 401:
        return UNAUTHENTICATED
    if status == 403:
        return FORBIDDEN
    if status == 429:
        return RATE_LIMITED
    return INTERNAL_ERROR  # any other status or any envelope that is not exactly the contract


_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?(Z|[+-]00:00)")


def _is_timestamp(value: Any) -> bool:
    """An RFC 3339 UTC ``date-time`` (Schema ``label_common_schema.json#/definitions/Timestamp``)."""
    if type(value) is not str or not _TIMESTAMP.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return True


def _raise_closed(error: Exception) -> NoReturn:
    """Raise ``error`` with no ``__cause__`` / ``__context__`` (same pattern as the token-refs
    boundary in ``backend_client``): the interpreter attaches the caller's handled exception as
    ``__context__`` during the raise, so it is cleared AFTER the raise has chained."""
    try:
        raise error from None
    except Exception as raised:
        raised.__cause__ = None
        raised.__context__ = None
        raised.__suppress_context__ = True
        raise


def _negotiated_ack(body: Any, sent: Mapping[str, Any]) -> dict[str, Any]:
    """The label write response, exactly per Schema and consistent with what was sent."""
    if (
        not isinstance(body, dict)
        or set(body) != _RESPONSE_KEYS
        or body["object_ref"] != sent["object_ref"]
        or body["purpose"] != sent["purpose"]
        or not _int(body["revision"], 1, _MAX_REVISION)
        or body["revision"] != sent["base_revision"] + 1
        or type(body["epoch"]) is not int
        or body["epoch"] != sent["epoch"]
        or not _is_timestamp(body["updated_at"])
    ):
        raise LabelTransportError(INTERNAL_ERROR)
    return dict(body)


class LabelTransport:
    """Authenticated calls to the label service routes."""

    def __init__(self, api_base_url: str, headers: HeadersProvider) -> None:
        self._base = api_base_url.rstrip("/")
        self._headers = headers

    async def _call(
        self, method: str, path: str, payload: Any = None
    ) -> dict[str, Any]:
        status, body, _retry_after = await self._request(
            method, f"{self._base}/labels{path}", payload, _TIMEOUT_SECONDS
        )
        if status not in (200, 201):
            code = body.get("error_code") if isinstance(body, dict) else None
            raise LabelTransportError(
                code if code in CLOSED_CODES else UNAVAILABLE, status
            )
        return _data(body)

    async def _request(
        self,
        method: str,
        url: str,
        payload: Any,
        total_timeout: float,
        *,
        bounded_auth: bool = False,
    ) -> tuple[int, Any, float | None]:
        """One gated, authenticated request: ``(status, parsed JSON | None, retry_after)``.

        ``retry_after`` is the ``Retry-After`` header as delta-seconds in ``[0, 3600]``
        (``None`` otherwise); no other header or body text is kept.

        ``bounded_auth`` (the negotiated write only): authentication and the HTTP call share ONE
        budget. Off (the legacy routes), the headers provider is awaited unbounded and the HTTP
        call then gets the full ``total_timeout``, exactly as before the negotiated route.
        """
        import aiohttp

        from .transport_gate import gate_session

        if bounded_auth:
            # A stalled or slow headers provider cannot outlive the budget; the HTTP timeout
            # is what is left afterwards.
            started = time.monotonic()
            headers = dict(await asyncio.wait_for(self._headers(), total_timeout))
            headers.setdefault("Content-Type", "application/json")
            left = total_timeout - (time.monotonic() - started)
            if left <= 0:
                raise TimeoutError
        else:
            headers = dict(await self._headers())
            headers.setdefault("Content-Type", "application/json")
            left = total_timeout
        timeout = aiohttp.ClientTimeout(total=left)
        try:
            async with (
                gate_session(
                    aiohttp.ClientSession(trust_env=True),
                    origins=[self._base],
                    strict=True,
                ) as session,
                session.request(
                    method,
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as response,
            ):
                try:
                    body = await response.json(content_type=None)
                except Exception:  # noqa: BLE001 - unreadable body is a closed code
                    body = None
                return response.status, body, _retry_after_seconds(response.headers)
        except LabelTransportError:
            raise
        except aiohttp.ClientError:
            raise LabelTransportError(UNAVAILABLE) from None

    async def read_binding_list(self) -> dict[str, Any]:
        return await self._call("GET", "/binding-list")

    async def read_labels(self, object_refs: list[str]) -> list[dict[str, Any]]:
        if not object_refs or len(object_refs) > _MAX_BATCH:
            raise LabelTransportError("invalid_request")
        data = await self._call("POST", "/objects/read", {"object_refs": object_refs})
        items = data.get("items")
        if not isinstance(items, list):
            raise LabelTransportError(BAD_RESPONSE)
        return [item for item in items if isinstance(item, dict)]

    async def write_label(
        self,
        *,
        object_ref: str,
        purpose: str,
        base_revision: int,
        epoch: int,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        """Write one sealed label.

        Token store enabled: ``POST /optimization/labels`` (see :meth:`write_label_negotiated`).
        Otherwise the legacy ``PUT /labels/objects``, unchanged.
        """
        if token_store.is_enabled():
            return await self.write_label_negotiated(
                object_ref=object_ref,
                purpose=purpose,
                base_revision=base_revision,
                epoch=epoch,
                envelope=envelope,
            )
        return await self._call(
            "PUT",
            "/objects",
            {
                "object_ref": object_ref,
                "purpose": purpose,
                "base_revision": base_revision,
                "epoch": epoch,
                "envelope": envelope,
            },
        )

    async def write_label_negotiated(
        self,
        *,
        object_ref: str,
        purpose: str,
        base_revision: int,
        epoch: int,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        """``POST /api/v1/optimization/labels`` (Schema ``label_write_schema.json``).

        The body is validated as a private snapshot first, then sent once per attempt. Every
        refusal is a closed :class:`LabelTransportError` code and nothing of the answer is
        kept: ``reference_not_issued`` and the other 4xx refusals are final (no retry);
        ``service_unavailable`` (503) is retried honoring ``Retry-After`` until the operation
        deadline, then raised (``retryable``); the dark route's 404 is
        ``label_route_unavailable`` and is NEVER retried on the legacy route (that would write
        an unvalidated ref while validation was asked for). Returns the validated response
        (``object_ref``, ``purpose``, ``revision``, ``epoch``, ``updated_at``).

        Closed boundary: whatever fails inside (a headers provider, the transport, a local
        validation) leaves as a :class:`LabelTransportError` with NO ``__cause__`` /
        ``__context__`` (the caller's handled exception included), so nothing a lower layer
        or the caller was holding can reach a traceback.
        """
        error: Exception
        try:
            return await self._write_label_negotiated(
                object_ref=object_ref,
                purpose=purpose,
                base_revision=base_revision,
                epoch=epoch,
                envelope=envelope,
            )
        except LabelTransportError as exc:
            error = LabelTransportError(exc.code, exc.status)
        except TransportGateRefusal as exc:
            error = exc  # the gate's own closed refusal (counts only) keeps its type
        except Exception:  # noqa: BLE001 - closed boundary: nothing else may escape
            error = LabelTransportError(INTERNAL_ERROR)
        _raise_closed(error)

    async def _write_label_negotiated(
        self,
        *,
        object_ref: str,
        purpose: str,
        base_revision: int,
        epoch: int,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        sent = _negotiated_body(object_ref, purpose, base_revision, epoch, envelope)
        url = f"{self._base}/optimization/labels"
        deadline = time.monotonic() + _WRITE_DEADLINE_SECONDS
        attempt = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LabelTransportError(SERVICE_UNAVAILABLE, 503)
            try:
                status, body, retry_after = await self._request(
                    "POST",
                    url,
                    sent,
                    min(_TIMEOUT_SECONDS, remaining),
                    bounded_auth=True,
                )
            except TimeoutError:
                if (
                    deadline - time.monotonic() <= 0
                ):  # authentication + call spent the budget
                    raise LabelTransportError(SERVICE_UNAVAILABLE, 503) from None
                raise LabelTransportError(UNAVAILABLE) from None
            if status == 200:
                return _negotiated_ack(body, sent)
            code = _negotiated_refusal_code(status, body)
            if code != SERVICE_UNAVAILABLE:
                raise LabelTransportError(code, status)
            delay = (
                retry_after
                if retry_after is not None
                else _RETRY_BACKOFF_SECONDS[
                    min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)
                ]
            )
            attempt += 1
            if attempt >= _MAX_WRITE_ATTEMPTS or delay >= deadline - time.monotonic():
                raise LabelTransportError(code, status)
            await _sleep(delay)


# ------------------------------------------------------------------------ configuration


def load_root_public_key(config_path: Path | None = None) -> bytes:
    """The raw Ed25519 root public key from ``labels.trust_anchor.root_public_key``."""
    path = config_path or find_config()
    if path is None:
        raise TrustAnchorError("no Traigent config found")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = data["labels"]["trust_anchor"]["root_public_key"]
        key = unb64u(str(raw))
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise TrustAnchorError("root_public_key is absent or malformed") from exc
    if len(key) != 32:
        raise TrustAnchorError("root_public_key is not a 32-byte key")
    return key


def _state_path() -> Path:
    base = os.getenv("TRAIGENT_RESULTS_FOLDER")
    root = Path(base).expanduser() if base else Path.home() / ".traigent"
    return root / "labels_state.json"


SEQUENCE_NOT_PERSISTED = "sequence_not_persisted"


def _state_dir() -> Path:
    """The resolved state directory: the parent of ``_state_path()``."""
    return _state_path().parent.resolve()


def _require_local_file(path: Path) -> None:
    """Refuse a state/lock/temp file unless it lies inside the resolved state directory.

    The path is SDK-built (``TRAIGENT_RESULTS_FOLDER`` or ``~/.traigent`` plus a fixed file
    name), never taken from a server response. Fails closed unless the file is not a
    symlink, its resolved location sits directly in its own parent, and that resolved
    location is inside the resolved state directory. A planted link, a ``..`` component
    or an override pointing elsewhere therefore cannot redirect the replay-protection
    journal, its lock, or its temp file.
    """
    try:
        resolved = path.resolve()
        parent = path.parent.resolve()
        if (
            path.is_symlink()
            or resolved.parent != parent
            or not resolved.is_relative_to(_state_dir())
        ):
            raise LabelTransportError(SEQUENCE_NOT_PERSISTED)
    except (OSError, RuntimeError):
        raise LabelTransportError(SEQUENCE_NOT_PERSISTED) from None


def _read_state(path: Path) -> dict[str, Any]:
    """The retained-sequence map. Missing file = empty; unreadable/corrupt = closed refusal."""
    _require_local_file(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        raise LabelTransportError(SEQUENCE_NOT_PERSISTED) from None
    try:
        data = json.loads(raw)
    except ValueError:
        raise LabelTransportError(SEQUENCE_NOT_PERSISTED) from None
    if not isinstance(data, dict):
        raise LabelTransportError(SEQUENCE_NOT_PERSISTED)
    return data


def _retained_sequence(path: Path, scope: str) -> int:
    value = _read_state(path).get(scope, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LabelTransportError(SEQUENCE_NOT_PERSISTED)
    return value


def _retain_sequence(path: Path, scope: str, sequence: int) -> None:
    """Durably record the highest verified binding-list sequence, or raise.

    Monotonic (never lowers), serialized across processes by an exclusive file lock, and
    atomic (temp file + fsync + rename + directory fsync). Any failure raises a closed
    ``sequence_not_persisted`` error: the caller must not hand out a writer, because a later
    process would otherwise accept a replayed older binding list.
    """
    try:
        import fcntl
    except ImportError:  # no advisory locks on this platform: fail closed
        raise LabelTransportError(SEQUENCE_NOT_PERSISTED) from None
    try:
        lock_path = path.with_suffix(".lock")
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        for candidate in (path, lock_path, tmp):
            _require_local_file(candidate)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = _read_state(path)
            current = data.get(scope, 0)
            if not isinstance(current, int) or isinstance(current, bool):
                raise LabelTransportError(SEQUENCE_NOT_PERSISTED)
            if sequence < current:
                raise LabelTransportError(SEQUENCE_NOT_PERSISTED)
            data[scope] = max(sequence, current)
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(data, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except LabelTransportError:
        raise
    except OSError:
        raise LabelTransportError(SEQUENCE_NOT_PERSISTED) from None


# ---------------------------------------------------------------------------- sender


_TOKEN_REF = re.compile(r"(knob|choice|metric|template)ref_[0-9a-f]{32}")


class LabelSender:
    """Seals names and writes them through the transport, once per object."""

    def __init__(
        self,
        transport: LabelTransport,
        anchor: TrustAnchor,
        root_public_key: bytes,
        *,
        state_path: Path | None = None,
        token_store_: token_store.TokenStore | None = None,
    ) -> None:
        self._transport = transport
        self._anchor = anchor
        self._root = root_public_key
        self._state = state_path or _state_path()
        self._writer: LabelWriter | None = None
        #: The journal of the agent whose tokens are labelled (None: sessions' own labels)
        self._tokens = token_store_

    async def _verified_writer(self) -> LabelWriter:
        if self._writer is not None:
            return self._writer
        blist = await self._transport.read_binding_list()
        bindings = [b for b in blist.get("bindings", []) if isinstance(b, dict)]
        admin = next(
            (b for b in bindings if b.get("purpose") == "admin_identity"), None
        )
        projects = [
            b
            for b in bindings
            if b.get("purpose") == "project_key"
            and b.get("project_ref") == self._anchor.project_ref
        ]
        if admin is None or not projects:
            raise LabelTransportError(BAD_RESPONSE)
        project = max(projects, key=lambda b: int(b.get("epoch") or 0))
        scope = f"{self._anchor.tenant_ref}/{self._anchor.project_ref}"
        writer = (
            LabelWriter(  # verifies the chain against the pinned root; raises if not
                self._anchor,
                root_public_key=self._root,
                admin_binding=admin,
                project_binding=project,
                binding_list=blist,
                retained_sequence=_retained_sequence(self._state, scope),
                now=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        )
        _retain_sequence(self._state, scope, writer.list_sequence)
        self._writer = writer
        return writer

    def _journal_state(self, object_ref: str) -> str | None:
        """The token journal state of ``object_ref`` (None: not a token of the journal)."""
        if self._tokens is None or not _TOKEN_REF.fullmatch(object_ref):
            return None
        try:
            return self._tokens.state_if_known(object_ref)
        except token_store.TokenStoreError:
            raise LabelTransportError(TOKEN_STATE_NOT_PERSISTED) from None

    async def send_new(self, labels: Mapping[str, tuple[str, str]]) -> dict[str, int]:
        """Write ``{object_ref: (purpose, text)}`` for objects that have no label yet.

        An object that already holds a revision is left alone (a rename made elsewhere is
        never overwritten). Returns ``{object_ref: written revision}``.

        With the token store enabled and a journal given, a token (S1, ``pending_label``)
        whose write is acknowledged with revision >= 1 moves to S2 (``label_acked``); a
        ``stale_revision`` answer means another writer won and, with no approved reader to
        compare, the token becomes ``label_conflict_unverified`` and so does the raised code.
        """
        if not labels:
            return {}
        if token_store.is_enabled():
            # fail closed BEFORE any network or write: a token-kind ref the selected journal
            # does not know would be written yet never advance (silent S1 forever)
            for object_ref in labels:
                if _TOKEN_REF.fullmatch(object_ref) and (
                    self._journal_state(object_ref) is None
                ):
                    _raise_closed(LabelTransportError(TOKEN_REF_UNKNOWN))
        writer = await self._verified_writer()
        existing = {
            item.get("object_ref"): item.get("revision")
            for item in await self._transport.read_labels(list(labels))
        }
        done: dict[str, int] = {}
        for object_ref, (purpose, text) in labels.items():
            negotiated = token_store.is_enabled()
            state = self._journal_state(object_ref) if negotiated else None
            if existing.get(object_ref):
                if state in (
                    token_store.S1_PENDING_LABEL,
                    token_store.S_LABEL_CONFLICT,
                ):
                    # a token whose label was never acknowledged to us already holds a
                    # revision: a lost-ack write of ours and another writer's label look the
                    # same, and without an approved reader they cannot be told apart. Never
                    # skip silently (S1 would stay pending forever): closed unverified state.
                    self._mark_conflict(object_ref)
                    _raise_closed(LabelTransportError(LABEL_CONFLICT_UNVERIFIED))
                continue
            envelope = writer.seal(
                text, purpose=purpose, object_ref=object_ref, base_revision=0
            )
            try:
                ack = await self._transport.write_label(
                    object_ref=object_ref,
                    purpose=purpose,
                    base_revision=0,
                    epoch=writer.epoch,
                    envelope=envelope,
                )
            except LabelTransportError as exc:
                if exc.code == "stale_revision" and state in (
                    token_store.S1_PENDING_LABEL,
                    token_store.S_LABEL_CONFLICT,
                ):
                    self._mark_conflict(object_ref)
                    _raise_closed(
                        LabelTransportError(LABEL_CONFLICT_UNVERIFIED, exc.status)
                    )
                raise
            done[object_ref] = 1
            if state in (token_store.S1_PENDING_LABEL, token_store.S2_LABEL_ACKED):
                self._mark_acked(object_ref, ack)
        return done

    def _mark_acked(self, object_ref: str, ack: Mapping[str, Any]) -> None:
        revision = ack.get("revision")
        if self._tokens is None or not _is_revision(revision):
            raise LabelTransportError(TOKEN_STATE_NOT_PERSISTED)
        try:
            self._tokens.mark_label_acked(object_ref, int(revision))  # type: ignore[arg-type]
        except token_store.TokenStoreError:
            raise LabelTransportError(TOKEN_STATE_NOT_PERSISTED) from None

    def _mark_conflict(self, object_ref: str) -> None:
        if self._tokens is None:
            return
        try:
            self._tokens.mark_label_stale(object_ref)
        except token_store.TokenStoreError:
            raise LabelTransportError(TOKEN_STATE_NOT_PERSISTED) from None


def _is_revision(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def sender_from_config(
    api_base_url: str,
    headers: HeadersProvider,
    config_path: Path | None = None,
    *,
    token_store_: token_store.TokenStore | None = None,
) -> LabelSender | None:
    """A :class:`LabelSender` when a trust anchor and root key are configured, else None."""
    try:
        anchor = load_trust_anchor(config_path)
        root = load_root_public_key(config_path)
    except TrustAnchorError:
        return None
    return LabelSender(
        LabelTransport(api_base_url, headers),
        anchor,
        root,
        token_store_=token_store_,
    )


def _token_store_for(
    api_base_url: str, refs: Mapping[str, str]
) -> token_store.TokenStore | None:
    """The token journal of this session's agent, when the token store is enabled.

    The namespace is (normalized backend origin, the session's issued ``agent`` reference):
    the same pair the tokens were issued under. Flag off, no agent reference or an invalid
    namespace: ``None`` and nothing is opened (``TokenStore`` itself touches no file until a
    token is looked up).
    """
    if not token_store.is_enabled():
        return None
    try:
        agent_ref = refs.get("agent")
        if not isinstance(agent_ref, str):
            return None
        return token_store.TokenStore(
            token_store.TokenNamespace.for_backend(api_base_url, agent_ref)
        )
    except token_store.TokenStoreError:
        return None


async def send_session_labels(
    texts: Mapping[str, str | None],
    refs: Mapping[str, str],
    api_base_url: str,
    headers: HeadersProvider,
    config_path: Path | None = None,
    *,
    token_store_: token_store.TokenStore | None = None,
) -> dict[str, int]:
    """Best-effort: send the human names of a session's agent / dataset / evaluator.

    ``texts`` maps a reference kind to its local name; ``refs`` maps kinds to the opaque
    references the Backend issued. Without a configured trust anchor nothing is sent.
    ``token_store_`` is the exact issuance journal for token-kind refs; when given it wins
    over the one derived from the session's ``agent`` reference.
    Never raises: any failure means "label not sent" (closed code logged at debug).
    """
    labels = {
        refs[kind]: ("label", text)
        for kind, text in texts.items()
        if isinstance(text, str) and text.strip() and kind in refs
    }
    if not labels:
        return {}
    try:
        sender = sender_from_config(
            api_base_url,
            headers,
            config_path,
            token_store_=(
                token_store_
                if token_store_ is not None
                else _token_store_for(api_base_url, refs)
            ),
        )
        if sender is None:
            return {}
        return await sender.send_new(labels)
    except LabelTransportError as exc:
        logger.debug("label not sent: %s", exc.code)
    except Exception as exc:  # noqa: BLE001 - verification / crypto failure: not sent
        logger.debug("label not sent: %s", type(exc).__name__)
    return {}
