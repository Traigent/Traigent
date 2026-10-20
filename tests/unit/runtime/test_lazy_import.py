"""The TVL compiler is imported lazily; the kwargs path never needs it."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import traigent
from traigent.runtime import tvl_fragment
from traigent.runtime.model import RuntimeRequirementError

from .conftest import FIXTURES


_ROOT = Path(__file__).resolve().parents[3]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter, fixed code
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=_ROOT,  # sys.path[0] == repo root, so this worktree is what imports
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_import_traigent_and_declare_without_tvl_installed() -> None:
    done = _run(
        """
        import sys
        sys.modules['tvl'] = None
        sys.modules['tvl.runtime_compiler'] = None
        import traigent
        r = traigent.requirement(latency_ms=2000, max_violation_rate=0.05)

        @r
        def f(): ...

        assert f.requirements.to_tvl()
        issues = f.requirements.validate()
        assert not [i for i in issues if i.severity == 'error'], issues
        loaded = [m for m in sys.modules if m == 'tvl' or m.startswith('tvl.')]
        assert not [m for m in loaded if sys.modules[m] is not None], loaded
        print('ok')
        """
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert done.stdout.strip().endswith("ok")


def test_plain_import_does_not_load_runtime_package() -> None:
    done = _run(
        """
        import sys
        import traigent
        assert 'traigent.runtime' not in sys.modules
        print('ok')
        """
    )
    assert done.returncode == 0, done.stderr[-800:]


def test_fragment_path_raises_closed_error_without_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tvl", None)
    monkeypatch.setitem(sys.modules, "tvl.runtime_compiler", None)
    req = tvl_fragment.load_fragment(FIXTURES / "latency.tvl.yml", max_fix_cost_usd=1.0)
    with pytest.raises(RuntimeRequirementError) as err:
        tvl_fragment.compile_hosted_subset(req, {"tokens": []}, "m")
    assert err.value.code == "tvl_compiler_unavailable"


def test_kwargs_apply_dry_run_needs_no_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tvl", None)
    monkeypatch.setitem(sys.modules, "tvl.runtime_compiler", None)

    @traigent.requirement(latency_ms=2000, max_violation_rate=0.05)
    def f() -> None: ...

    assert f.requirements.apply("p", agent_id="a", dry_run=True) is None
