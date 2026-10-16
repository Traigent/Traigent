"""Label-service transport: the SDK side of the Backend ``/api/v1/labels/*`` routes.

Only the routes a label WRITER needs: ``GET /labels/binding-list``, ``PUT /labels/objects``
(``base_revision`` + ``epoch``) and ``POST /labels/objects/read``. A name is sealed to the
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

import json
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from traigent.utils.logging import get_logger

from ._codec import unb64u
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
_TIMEOUT_SECONDS = 15
_MAX_BATCH = 200


class LabelTransportError(Exception):
    """A label-service call failed. ``code`` is closed; nothing of the exchange is kept."""

    def __init__(self, code: str, status: int | None = None) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


def _data(body: Any) -> dict[str, Any]:
    if isinstance(body, dict) and isinstance(body.get("data"), dict):
        return dict(body["data"])
    if isinstance(body, dict):
        return body
    raise LabelTransportError(BAD_RESPONSE)


class LabelTransport:
    """Authenticated calls to the label service routes."""

    def __init__(self, api_base_url: str, headers: HeadersProvider) -> None:
        self._base = api_base_url.rstrip("/")
        self._headers = headers

    async def _call(
        self, method: str, path: str, payload: Any = None
    ) -> dict[str, Any]:
        import aiohttp

        from .transport_gate import gate_session

        headers = dict(await self._headers())
        headers.setdefault("Content-Type", "application/json")
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
        try:
            async with (
                gate_session(
                    aiohttp.ClientSession(trust_env=True),
                    origins=[self._base],
                    strict=True,
                ) as session,
                session.request(
                    method,
                    f"{self._base}/labels{path}",
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as response,
            ):
                try:
                    body = await response.json(content_type=None)
                except Exception:  # noqa: BLE001 - unreadable body is a closed code
                    body = None
                if response.status not in (200, 201):
                    code = body.get("error_code") if isinstance(body, dict) else None
                    raise LabelTransportError(
                        code if code in CLOSED_CODES else UNAVAILABLE, response.status
                    )
                return _data(body)
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


def _read_state(path: Path) -> dict[str, Any]:
    """The retained-sequence map. Missing file = empty; unreadable/corrupt = closed refusal."""
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = _read_state(path)
            current = data.get(scope, 0)
            if not isinstance(current, int) or isinstance(current, bool):
                raise LabelTransportError(SEQUENCE_NOT_PERSISTED)
            if sequence < current:
                raise LabelTransportError(SEQUENCE_NOT_PERSISTED)
            data[scope] = max(sequence, current)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
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


class LabelSender:
    """Seals names and writes them through the transport, once per object."""

    def __init__(
        self,
        transport: LabelTransport,
        anchor: TrustAnchor,
        root_public_key: bytes,
        *,
        state_path: Path | None = None,
    ) -> None:
        self._transport = transport
        self._anchor = anchor
        self._root = root_public_key
        self._state = state_path or _state_path()
        self._writer: LabelWriter | None = None

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

    async def send_new(self, labels: Mapping[str, tuple[str, str]]) -> dict[str, int]:
        """Write ``{object_ref: (purpose, text)}`` for objects that have no label yet.

        An object that already holds a revision is left alone (a rename made elsewhere is
        never overwritten). Returns ``{object_ref: written revision}``.
        """
        if not labels:
            return {}
        writer = await self._verified_writer()
        existing = {
            item.get("object_ref"): item.get("revision")
            for item in await self._transport.read_labels(list(labels))
        }
        done: dict[str, int] = {}
        for object_ref, (purpose, text) in labels.items():
            if existing.get(object_ref):
                continue
            envelope = writer.seal(
                text, purpose=purpose, object_ref=object_ref, base_revision=0
            )
            await self._transport.write_label(
                object_ref=object_ref,
                purpose=purpose,
                base_revision=0,
                epoch=writer.epoch,
                envelope=envelope,
            )
            done[object_ref] = 1
        return done


def sender_from_config(
    api_base_url: str,
    headers: HeadersProvider,
    config_path: Path | None = None,
) -> LabelSender | None:
    """A :class:`LabelSender` when a trust anchor and root key are configured, else None."""
    try:
        anchor = load_trust_anchor(config_path)
        root = load_root_public_key(config_path)
    except TrustAnchorError:
        return None
    return LabelSender(LabelTransport(api_base_url, headers), anchor, root)


async def send_session_labels(
    texts: Mapping[str, str | None],
    refs: Mapping[str, str],
    api_base_url: str,
    headers: HeadersProvider,
    config_path: Path | None = None,
) -> dict[str, int]:
    """Best-effort: send the human names of a session's agent / dataset / evaluator.

    ``texts`` maps a reference kind to its local name; ``refs`` maps kinds to the opaque
    references the Backend issued. Without a configured trust anchor nothing is sent.
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
        sender = sender_from_config(api_base_url, headers, config_path)
        if sender is None:
            return {}
        return await sender.send_new(labels)
    except LabelTransportError as exc:
        logger.debug("label not sent: %s", exc.code)
    except Exception as exc:  # noqa: BLE001 - verification / crypto failure: not sent
        logger.debug("label not sent: %s", type(exc).__name__)
    return {}
