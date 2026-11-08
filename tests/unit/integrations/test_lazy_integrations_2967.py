"""Regression tests for #2967: optional integrations must import lazily.

``traigent.optimize`` must not import optional framework adapters (DSPy,
LangChain, PydanticAI, Langfuse, OpenAI/LangChain LLM helpers) as a side effect.
Those adapters load on first access of their public names instead.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_FAKE_FRAMEWORKS = ("dspy", "langchain_core", "pydantic_ai", "langfuse")

_ADAPTER_MODULES = (
    "traigent.integrations.dspy_adapter",
    "traigent.integrations.pydantic_ai",
    "traigent.integrations.langchain",
    "traigent.integrations.langfuse",
    "traigent.integrations.llms.langchain.base",
    "traigent.integrations.llms.openai",
)


def _make_fake_frameworks(root: Path) -> Path:
    """Create stub third-party packages that record being imported."""
    marker = root / "imported.txt"
    for name in _FAKE_FRAMEWORKS:
        pkg = root / "fakes" / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(
            f"from pathlib import Path\n"
            f"with open({str(marker)!r}, 'a') as fh:\n"
            f"    fh.write({name!r} + '\\n')\n"
        )
    return marker


def _run(code: str, root: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "PYTHONPATH": str(root / "fakes") + ":" + ":".join(sys.path),
        "TRAIGENT_MOCK_LLM": "true",
        "PATH": "/usr/bin:/bin",
    }
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def test_optimize_access_does_not_import_optional_adapters(tmp_path: Path) -> None:
    marker = _make_fake_frameworks(tmp_path)
    result = _run(
        f"""
        import sys
        import traigent
        traigent.optimize
        loaded = [m for m in {_ADAPTER_MODULES!r} if m in sys.modules]
        print("ADAPTERS:" + ",".join(loaded))
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "ADAPTERS:\n" in result.stdout + "\n", result.stdout
    assert result.stdout.strip().endswith("ADAPTERS:"), result.stdout
    assert not marker.exists(), marker.read_text()


def test_public_names_still_resolve_lazily(tmp_path: Path) -> None:
    marker = _make_fake_frameworks(tmp_path)
    result = _run(
        """
        import sys
        import traigent
        traigent.optimize
        import traigent.integrations as intg
        assert "traigent.integrations.dspy_adapter" not in sys.modules
        assert intg.DSPY_AVAILABLE is True
        assert callable(intg.create_dspy_integration)
        assert "traigent.integrations.dspy_adapter" in sys.modules
        missing = [n for n in intg.__all__ if not hasattr(intg, n)]
        assert not missing, missing
        print("OK")
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "dspy" in marker.read_text().split()


def test_unavailable_group_keeps_raising_stub() -> None:
    import traigent.integrations as intg

    if not intg.PYDANTICAI_AVAILABLE:
        with pytest.raises(ImportError, match="PydanticAI"):
            intg.create_pydantic_ai_handler()
    with pytest.raises(AttributeError):
        intg.definitely_not_an_export  # noqa: B018
