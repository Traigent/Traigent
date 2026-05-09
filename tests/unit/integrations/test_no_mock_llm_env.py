"""Verify dangerous env-toggles are not honored.

Sprint 2 cleanup (S2-B) removed the ``TRAIGENT_MOCK_LLM`` env-toggle
from three high-severity sites that previously shipped to customers:

* ``traigent/integrations/utils/mock_adapter.py`` — ``with_mock_support``
  decorator was deleted entirely. ``MockAdapter.is_mock_enabled`` now
  consults the in-code flag set by
  :func:`traigent.testing.enable_mock_mode_for_quickstart`. The legacy
  ``TRAIGENT_MOCK_LLM=true`` env var is honored only outside production
  (and is hard-blocked when ``ENVIRONMENT=production``); see
  ``test_mock_adapter_safety.py`` for those guarantees.
* ``traigent/security/encryption.py`` — encrypt() / decrypt() fallback
  branches (env-toggle produced ``b"mock_" + plaintext`` "ciphertext"
  and stripped the prefix on decrypt). These were deleted with no
  fallback at all.
* ``traigent/evaluators/local.py`` — ``_compute_mock_accuracy`` (env
  toggle fabricated accuracy via random.uniform() + string-length
  heuristics). Deleted with no fallback.

These tests pin the surviving guarantees: provider-specific
``*_MOCK`` env vars are completely ignored everywhere, the
``with_mock_support`` decorator is gone, and the encryption /
evaluation paths fail closed regardless of any mock-style env var.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from traigent.integrations.utils.mock_adapter import MockAdapter
from traigent.security.encryption import EncryptionManager, KeyManager
