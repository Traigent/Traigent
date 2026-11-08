"""Regression tests for example-script defect #3204."""

from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples"


# --- #3204: run.py must not overwrite an explicit TRAIGENT_MOCK_LLM ---------

CS_DIR = EXAMPLES / "gallery/page-inline/ai-agents/ex01-customer-support"


@pytest.mark.parametrize("preset", ["false", "0"])
def test_customer_support_run_respects_explicit_mock_setting(preset, monkeypatch):
    stub = types.ModuleType("customer_support")
    stub.demonstrate_customer_support = lambda: None
    monkeypatch.setitem(sys.modules, "customer_support", stub)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", preset)
    runpy.run_path(str(CS_DIR / "run.py"), run_name="__main__")
    assert os.environ["TRAIGENT_MOCK_LLM"] == preset


def test_customer_support_run_defaults_to_mock_when_unset(monkeypatch):
    stub = types.ModuleType("customer_support")
    stub.demonstrate_customer_support = lambda: None
    monkeypatch.setitem(sys.modules, "customer_support", stub)
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    runpy.run_path(str(CS_DIR / "run.py"), run_name="__main__")
    assert os.environ["TRAIGENT_MOCK_LLM"] == "true"
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
