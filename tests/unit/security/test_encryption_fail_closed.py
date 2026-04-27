"""Fail-closed import guard tests for traigent.security.encryption.

Codex review (B7) flagged the prior silent Fernet mock fallback at
``traigent/security/encryption.py:38`` (``encrypt`` returning ``b"encrypted_" + data``).
Real ``cryptography`` is now a hard dependency and the import block re-raises
``ImportError`` with a clear remediation message instead of installing a fake
class. These tests pin that behavior so it cannot regress silently.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_encryption_module_imports_cleanly_with_cryptography_installed():
    """Module must import without falling back to a mock when crypto is available."""
    module = importlib.import_module("traigent.security.encryption")
    # CRYPTO_AVAILABLE flag must be true and Fernet must be the real class.
    assert getattr(module, "CRYPTO_AVAILABLE", False) is True
    fernet_cls = getattr(module, "Fernet")
    # Real Fernet lives in cryptography.fernet — module path check is a
    # cheap proxy for "this is not the in-file mock class".
    assert fernet_cls.__module__.startswith("cryptography.")


def test_encryption_module_reraises_importerror_when_cryptography_missing(monkeypatch):
    """If cryptography import fails, we fail closed with a clear message."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("cryptography"):
            raise ImportError(f"simulated missing dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Force a fresh import so the try/except at module top runs.
    sys.modules.pop("traigent.security.encryption", None)

    with pytest.raises(ImportError, match=r"cryptography>=46\.0"):
        importlib.import_module("traigent.security.encryption")

    # Cleanup: restore the real module so other tests aren't poisoned.
    monkeypatch.setattr(builtins, "__import__", real_import)
    sys.modules.pop("traigent.security.encryption", None)
    importlib.import_module("traigent.security.encryption")
