"""Regression tests for example-script defect #3202."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples"


def _load(path: Path, name: str, monkeypatch: pytest.MonkeyPatch):
    """Import an example file as a module with mock mode off (no import side effects)."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


# --- #3202: BM25 wrappers must use the public invoke() API -------------------


def _docs():
    from langchain_core.documents import Document

    return [
        Document(page_content="the budget was approved in march"),
        Document(page_content="the timeline slips to june"),
        Document(page_content="unrelated text about lunch"),
    ]


@pytest.mark.parametrize(
    ("rel", "builder"),
    [
        ("core/rag-optimization/run.py", "_build_retriever"),
        ("core/chunking-long-context/run.py", "_bm25_from_documents"),
    ],
)
def test_bm25_wrapper_retrieves_with_installed_langchain(rel, builder, monkeypatch):
    pytest.importorskip("langchain_community.retrievers")
    pytest.importorskip("langchain_anthropic")
    module = _load(EXAMPLES / rel, "ex_" + builder + rel.split("/")[1], monkeypatch)
    retriever = getattr(module, builder)(_docs())
    retriever.k = 1
    hits = retriever.get_relevant_documents("budget approved")
    assert [d.page_content for d in hits] == ["the budget was approved in march"]


def test_chunking_example_exits_nonzero_when_no_best_config(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    module = _load(
        EXAMPLES / "core/chunking-long-context/run.py", "ex_chunk_guard", monkeypatch
    )
    failed = types.SimpleNamespace(best_config=None, best_score=None)
    with pytest.raises(SystemExit) as exc:
        module._require_best_config(failed)
    assert exc.value.code not in (0, None)
    ok = types.SimpleNamespace(best_config={"k": 1}, best_score=1.0)
    module._require_best_config(ok)
