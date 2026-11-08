"""Shipped example scripts must run in mock mode on the dependency set the SDK installs."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _run(script: Path, tmp_path: Path, *, drop: tuple[str, ...] = ()):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(
        TRAIGENT_MOCK_LLM="true",
        TRAIGENT_OFFLINE_MODE="true",
        PYTHONPATH=str(ROOT),
    )
    # Constructing the LangChain client needs a key even though mock mode
    # intercepts every call; this value is never sent anywhere.
    env.setdefault("OPENAI_API_KEY", "sk-mock-not-a-real-key")
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


def test_examples_do_not_import_removed_langchain_schema() -> None:
    offenders = []
    for path in EXAMPLES.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "langchain.schema":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


@pytest.mark.timeout(180)
def test_local_search_example_builds_messages_langchain_accepts(
    tmp_path: Path,
) -> None:
    """The message object must be a real LangChain message (#2934).

    Standalone, this script still reaches the provider-host guard in mock mode
    (it calls the LLM outside an optimization run), so only the message
    coercion failure is asserted absent here.
    """
    script = (
        EXAMPLES / "advanced/optimization-routing/ex01-local-search/local_search.py"
    )
    result = _run(script, tmp_path)
    assert "Unsupported message type" not in result.stderr, result.stderr[-2000:]
    assert "MESSAGE_COERCION_FAILURE" not in result.stderr
