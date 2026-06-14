"""Per-provider quickstart examples for the Traigent SDK.

Each module in this package is a runnable, copy-pasteable example that
optimizes a tiny Q&A agent across models/temperatures for one LLM
provider. They all share the same shape as the bundled quickstart
(:mod:`traigent.examples.quickstart`):

* mock by default (no API keys, no provider spend), so ``python -m
  traigent.examples.providers.<provider>`` runs out of the box;
* a real call when ``TRAIGENT_MOCK_LLM=false`` and the provider's
  credentials are present.

:data:`MANIFEST` is the single source of truth for the provider table:
which implementation each provider uses (LangChain vs LiteLLM), the
import line, models, required env vars, and the pip package a bare
``pip install traigent`` would still need. Both these example scripts
and the portal Quick Start page derive their code snippets from this
manifest, and tests in both repos assert they stay in sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


def load_manifest() -> dict[str, Any]:
    """Return the parsed provider manifest (the single source of truth)."""
    with _MANIFEST_PATH.open(encoding="utf-8") as fh:
        return cast("dict[str, Any]", json.load(fh))


def get_provider(provider_id: str) -> dict[str, Any]:
    """Return the manifest entry for ``provider_id`` or raise ``KeyError``."""
    for entry in load_manifest()["providers"]:
        if entry["id"] == provider_id:
            return cast("dict[str, Any]", entry)
    raise KeyError(f"unknown provider id: {provider_id!r}")


MANIFEST: dict[str, Any] = load_manifest()

__all__ = ["MANIFEST", "load_manifest", "get_provider"]
