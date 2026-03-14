"""Tests for scripts/auth/get_api_key.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_get_api_key_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "auth" / "get_api_key.py"
    spec = importlib.util.spec_from_file_location("get_api_key_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_backend_url_strips_trailing_slash() -> None:
    module = _load_get_api_key_module()

    assert module.normalize_backend_url("https://api.traigent.ai/") == "https://api.traigent.ai"


def test_normalize_backend_url_allows_localhost_http() -> None:
    module = _load_get_api_key_module()

    assert module.normalize_backend_url("http://localhost:5000/") == "http://localhost:5000"


@pytest.mark.parametrize(
    ("backend_url", "error"),
    [
        ("ftp://api.traigent.ai", "must start with http:// or https://"),
        ("https://api.traigent.ai?token=1", "must not include params, query strings, or fragments"),
        ("https://", "must include a hostname"),
        ("http://api.traigent.ai", "Non-local backend URLs must use https"),
        (
            "https://user:pass@api.traigent.ai",  # pragma: allowlist secret
            "must not include embedded credentials",
        ),
        ("https://192.168.1.10", "must not target private or loopback IPs"),
    ],
)
def test_normalize_backend_url_rejects_unsafe_values(
    backend_url: str, error: str
) -> None:
    module = _load_get_api_key_module()

    with pytest.raises(ValueError, match=error):
        module.normalize_backend_url(backend_url)
