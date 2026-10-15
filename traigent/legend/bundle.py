"""Canonical byte containers: the stored capture bundle and the source-tree archive."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from traigent.identity.content_identity import ContentIdentityError, canonical_bytes
from traigent.legend.errors import LegendError

BUNDLE_MAGIC: Final = b"TRGLEG1\n"
TREE_MAGIC: Final = b"TRGTREE1\n"


@dataclass(frozen=True)
class Bundle:
    header: dict[str, Any]  # {"manifest": ..., "pillar_inputs": {...}}
    representations: tuple[bytes, ...]


def pack_bundle(header: dict[str, Any], reps: Sequence[bytes]) -> bytes:
    head = canonical_bytes(header)
    out = [
        BUNDLE_MAGIC,
        struct.pack(">I", len(head)),
        head,
        struct.pack(">I", len(reps)),
    ]
    for rep in reps:
        out += [struct.pack(">Q", len(rep)), rep]
    return b"".join(out)


def unpack_bundle(data: bytes) -> Bundle:
    """Parse a bundle; any truncation or malformation raises ``restore_failed``."""
    import json

    try:
        if not data.startswith(BUNDLE_MAGIC):
            raise ValueError
        pos = len(BUNDLE_MAGIC)
        (hlen,) = struct.unpack_from(">I", data, pos)
        pos += 4
        raw = data[pos : pos + hlen]
        if len(raw) != hlen:
            raise ValueError
        pos += hlen
        header = json.loads(raw)
        (count,) = struct.unpack_from(">I", data, pos)
        pos += 4
        reps = []
        for _ in range(count):
            (rlen,) = struct.unpack_from(">Q", data, pos)
            pos += 8
            rep = data[pos : pos + rlen]
            if len(rep) != rlen:
                raise ValueError
            pos += rlen
            reps.append(rep)
        if pos != len(data) or not isinstance(header, dict):
            raise ValueError
        return Bundle(header, tuple(reps))
    except (ValueError, struct.error):
        raise LegendError(
            "restore_failed", "bundle is malformed or truncated"
        ) from None


def pack_tree(entries: Sequence[tuple[str, bool, bytes]]) -> bytes:
    """Deterministic archive of ``(posix_path, executable, data)`` sorted by path."""
    out = [TREE_MAGIC]
    seen = set()
    for path, executable, data in sorted(entries, key=lambda e: e[0]):
        if path in seen:
            raise ContentIdentityError("duplicate path in tree")
        seen.add(path)
        raw = path.encode("utf-8")
        out += [
            struct.pack(">IBQ", len(raw), 1 if executable else 0, len(data)),
            raw,
            data,
        ]
    return b"".join(out)


def unpack_tree(data: bytes) -> list[tuple[str, bool, bytes]]:
    try:
        if not data.startswith(TREE_MAGIC):
            raise ValueError
        pos = len(TREE_MAGIC)
        entries: list[tuple[str, bool, bytes]] = []
        seen: set[str] = set()
        while pos < len(data):
            plen, exe, size = struct.unpack_from(">IBQ", data, pos)
            pos += struct.calcsize(">IBQ")
            path = data[pos : pos + plen].decode("utf-8")
            pos += plen
            body = data[pos : pos + size]
            if len(body) != size:
                raise ValueError
            pos += size
            parts = path.split("/")
            if (
                not path
                or path.startswith("/")
                or "\\" in path
                or any(p in ("", ".", "..") for p in parts)
                or path in seen
            ):
                raise ValueError
            seen.add(path)
            entries.append((path, bool(exe), body))
        return entries
    except (ValueError, struct.error, UnicodeDecodeError):
        raise LegendError("restore_failed", "tree archive is malformed") from None
