"""Trust-anchor pinning for the label service.

The customer pins the tenant root key's fingerprint OUTSIDE Traigent, in the repo's
committed Traigent config (``traigent.yml`` / ``.traigent.yml``, searched upward from
the working directory)::

    labels:
      trust_anchor:
        root_fingerprint: "<64 lowercase hex: SHA-256 of the raw Ed25519 root public key>"
        tenant_ref: "tenant_<32 hex>"
        project_ref: "project_<32 hex>"

A missing or malformed anchor raises :class:`TrustAnchorError`; callers must treat that
as "refuse to encrypt" (fail closed), never as "skip verification".
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_FILE_NAMES = ("traigent.yml", "traigent.yaml", ".traigent.yml", ".traigent.yaml")
CONFIG_KEY_PATH = "labels.trust_anchor"

_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_TENANT = re.compile(r"^tenant_[0-9a-f]{32}$")
_PROJECT = re.compile(r"^project_[0-9a-f]{32}$")


class TrustAnchorError(ValueError):
    """The pinned trust anchor is absent or malformed (fail closed)."""


@dataclass(frozen=True)
class TrustAnchor:
    root_fingerprint: str
    tenant_ref: str
    project_ref: str

    def __post_init__(self) -> None:
        if not _FINGERPRINT.match(self.root_fingerprint):
            raise TrustAnchorError(
                "root_fingerprint must be 64 lowercase hex characters"
            )
        if not _TENANT.match(self.tenant_ref):
            raise TrustAnchorError("tenant_ref is not a tenant reference")
        if not _PROJECT.match(self.project_ref):
            raise TrustAnchorError("project_ref is not a project reference")


def find_config(start: Path | None = None) -> Path | None:
    """First Traigent config file found walking up from ``start`` (default: cwd)."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        for name in CONFIG_FILE_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_trust_anchor(config_path: Path | None = None) -> TrustAnchor:
    path = config_path or find_config()
    if path is None:
        raise TrustAnchorError(f"no Traigent config found with {CONFIG_KEY_PATH}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise TrustAnchorError("Traigent config is unreadable") from exc
    labels = data.get("labels") if isinstance(data, dict) else None
    section = labels.get("trust_anchor") if isinstance(labels, dict) else None
    if not isinstance(section, dict):
        raise TrustAnchorError(f"config has no {CONFIG_KEY_PATH} section")
    try:
        return TrustAnchor(
            root_fingerprint=str(section["root_fingerprint"]).strip().lower(),
            tenant_ref=str(section["tenant_ref"]),
            project_ref=str(section["project_ref"]),
        )
    except KeyError as exc:
        raise TrustAnchorError("trust anchor is missing a required key") from exc
