"""Local alias map: human names -> opaque entity refs, stored client-side only.

Names live in ``.traigent/legend/aliases.json`` and never appear in any payload.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from traigent.legend import _contract as c
from traigent.legend.errors import LegendError

DEFAULT_ALIAS_PATH = Path(".traigent") / "legend" / "aliases.json"


class AliasMap:
    def __init__(self, path: str | os.PathLike[str] = DEFAULT_ALIAS_PATH) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, dict[str, str]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            raise LegendError("local_alias_file_unreadable") from None
        aliases = data.get("aliases") if isinstance(data, dict) else None
        if not isinstance(aliases, dict):
            raise LegendError("local_alias_file_unreadable")
        return {k: dict(v) for k, v in aliases.items() if isinstance(v, dict)}

    def _save(self, aliases: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {"schema_version": c.SCHEMA_VERSION, "aliases": aliases},
                    handle,
                    sort_keys=True,
                    indent=2,
                )
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def set(self, name: str, kind: str, entity_ref: str) -> None:
        if (
            kind not in c.ENTITY_KINDS
            or not name
            or not c.ENTITY_REF_RE.fullmatch(entity_ref)
        ):
            raise LegendError(
                "invalid_request", "alias needs a name, a kind and an opaque entity ref"
            )
        aliases = self._load()
        aliases.setdefault(kind, {})[name] = entity_ref
        self._save(aliases)

    def resolve(self, name: str, kind: str) -> str | None:
        return self._load().get(kind, {}).get(name)

    def names(self, kind: str) -> list[str]:
        return sorted(self._load().get(kind, {}))

    def remove(self, name: str, kind: str) -> bool:
        aliases = self._load()
        removed = aliases.get(kind, {}).pop(name, None) is not None
        if removed:
            self._save(aliases)
        return removed
