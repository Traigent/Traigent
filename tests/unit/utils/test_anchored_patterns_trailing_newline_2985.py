"""Anchored validators must not accept a value with a trailing newline (#2985)."""

from __future__ import annotations

import importlib

import pytest

H64 = "a" * 64
H32 = "b" * 32

CASES = [
    ("traigent.analytics_mcp.tools", "_DIRECTOR_INSTRUCTION_ID_PATTERN", "di_" + H32),
    (
        "traigent.cloud.analytics_client",
        "_EXAMPLE_INSIGHT_REF_PATTERN",
        "exref_" + "c" * 16,
    ),
    ("traigent.cloud.dataset_converter", "_EXAMPLE_SET_ID_PATTERN", "set-1"),
    ("traigent.cloud.sync_manager", "_EXAMPLE_METRIC_KEY_RE", "accuracy"),
    ("traigent.cloud.validators", "_IDENTIFIER_PATTERN", "name_1"),
    ("traigent.hooks.validator", "_SNAPSHOT_SUFFIX", "-20250101"),
    ("traigent.observability.otel.contract", "_SAFE_TOKEN", "tok.en"),
    ("traigent.observability.otel.contract", "_SAFE_SCOPE_VERSION", "1.2.3"),
    ("traigent.observability.otel.contract", "_SAFE_KEY", "a.b"),
    ("traigent.providers.validation", "_HF_ORG_MODEL_PATTERN", "org/model"),
    ("traigent.utils.module_loading", "_DOTTED_MODULE_RE", "pkg.mod"),
    ("traigent.wrapper.service", "_IDENTIFIER_RE", "fn_name"),
    ("traigent.security.auth.helpers", "EMAIL_PATTERN", "a@b.co"),
    ("traigent.security.auth.helpers", "ROLE_PATTERN", "admin"),
    ("traigent.labels.types", "_DIGEST", H64),
    ("traigent.labels.types", "_KEY_MATERIAL", "key_1"),
    ("traigent.labels.types", "_REL_PATH", "pkg/mod.py"),
    ("traigent.labels.types", "_CODE_ID", "pkg.mod:fn"),
    ("traigent.labels.trust_anchor", "_FINGERPRINT", H64),
    ("traigent.labels.trust_anchor", "_TENANT", "tenant_" + H32),
    ("traigent.labels.trust_anchor", "_PROJECT", "project_" + H32),
]


@pytest.mark.parametrize(("module", "name", "valid"), CASES)
def test_trailing_newline_is_rejected(module: str, name: str, valid: str) -> None:
    pattern = getattr(importlib.import_module(module), name)
    assert pattern.match(valid), "control: the plain value must match"
    assert pattern.match(valid + "\n") is None


@pytest.mark.parametrize("name", ["KEY_PATTERN"])
def test_dto_key_patterns_reject_trailing_newline(name: str) -> None:
    from traigent.cloud import dtos

    for cls in (c for c in vars(dtos).values() if isinstance(c, type)):
        pattern = vars(cls).get(name)
        if pattern is not None:
            assert pattern.match("key_1")
            assert pattern.match("key_1\n") is None
