"""Anti-drift tests for the per-provider quickstart examples.

These are the teeth that catch an SDK change silently breaking the
bundled per-provider examples (and, by extension, the portal Quick Start
snippets that are derived from the same manifest):

* The manifest stays well-formed and in sync with the example files.
* Every runnable example still imports against the live SDK surface
  (``traigent.optimize``, ``EvaluationOptions``, ``traigent.get_config``,
  the LangChain / LiteLLM entry points) without ``ImportError`` /
  ``AttributeError`` — so a rename in the SDK fails here loudly.
* The ``ensure_packages`` bootstrap behaves safely (never installs
  silently; prints the command and exits in non-interactive shells).

The examples themselves are exercised end-to-end (a real mock run) in
``test_quickstart_entry`` style by the dedicated subprocess test below.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from traigent.examples.providers import MANIFEST, get_provider, load_manifest
from traigent.testing import _reset_for_tests

_PROVIDERS_DIR = Path(__file__).resolve()
for _parent in _PROVIDERS_DIR.parents:
    _candidate = _parent / "traigent" / "examples" / "providers"
    if _candidate.is_dir():
        _PROVIDERS_DIR = _candidate
        break

_REQUIRED_KEYS = {
    "id",
    "label",
    "impl",
    "runnable_module",
    "import_check",
    "pip_package",
    "import_line",
    "models",
    "env_vars",
    "creds_note",
    "run_command",
}

_PROVIDER_IDS = [p["id"] for p in MANIFEST["providers"]]
_RUNNABLE = [p for p in MANIFEST["providers"] if p["runnable_module"]]


@pytest.fixture(autouse=True)
def _reset_mock_mode():
    _reset_for_tests()
    yield
    _reset_for_tests()


# --------------------------------------------------------------------------- #
# Manifest shape + parity with the example files                              #
# --------------------------------------------------------------------------- #


def test_manifest_loads_and_is_stable():
    assert load_manifest() == MANIFEST
    assert MANIFEST["schema_version"] == 1
    assert MANIFEST["dataset"] == "qa_samples.jsonl"


def test_manifest_provider_ids_are_the_expected_set():
    # Locks the public provider set; the portal selector mirrors these ids.
    assert set(_PROVIDER_IDS) == {
        "mock",
        "openai",
        "anthropic",
        "openrouter",
        "nous",
        "azure",
        "gemini",
        "groq",
        "aws",
        "gcp",
        "litellm",
    }


@pytest.mark.parametrize("provider", MANIFEST["providers"], ids=_PROVIDER_IDS)
def test_provider_entry_has_required_keys(provider):
    missing = _REQUIRED_KEYS - provider.keys()
    assert not missing, f"{provider['id']} missing keys: {missing}"
    assert provider["impl"] in {"langchain", "litellm"}
    assert provider["models"], f"{provider['id']} has no models"
    assert provider["env_vars"], f"{provider['id']} has no env_vars"
    # LiteLLM is a core dependency, so it must never carry an extra pip package.
    if provider["impl"] == "litellm":
        assert provider["pip_package"] is None


def test_bundled_dataset_exists():
    assert (_PROVIDERS_DIR / MANIFEST["dataset"]).is_file()


@pytest.mark.parametrize("provider", _RUNNABLE, ids=[p["id"] for p in _RUNNABLE])
def test_example_file_exists_and_matches_manifest(provider):
    path = _PROVIDERS_DIR / f"{provider['runnable_module']}.py"
    assert path.is_file(), f"missing example file for {provider['id']}: {path}"
    source = path.read_text(encoding="utf-8")

    # The example must use the implementation the manifest advertises.
    assert provider["import_line"] in source, (
        f"{provider['id']} example does not contain manifest import_line "
        f"{provider['import_line']!r}"
    )
    if provider["impl"] == "litellm":
        assert "litellm.completion(" in source, (
            f"{provider['id']} must call litellm.completion(...) at the module "
            "level so Traigent's interceptor is used (not a `from litellm import "
            "completion` binding)."
        )
    else:
        assert ".invoke(" in source


# --------------------------------------------------------------------------- #
# Anti-drift: each runnable example imports against the live SDK surface       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", _RUNNABLE, ids=[p["id"] for p in _RUNNABLE])
def test_example_module_imports_against_live_sdk(provider):
    """Importing the example exercises @traigent.optimize, EvaluationOptions,
    get_config and the provider entry point. A rename in the SDK surface makes
    this raise ImportError/AttributeError — the drift signal we want.

    Run in a SUBPROCESS: importing an example has deliberate global side effects
    (it enables mock mode and seeds TRAIGENT_DATASET_ROOT + placeholder provider
    keys so its module-level @traigent.optimize can validate the dataset). Doing
    that in-process would leak into sibling tests (it did — it broke the MCP
    scaffold test under random ordering). This mirrors test_quickstart_entry,
    which never imports the example __main__ in-process for the same reason.
    """
    # Skip cleanly if the provider's package isn't installed in this env
    # (e.g. core-only CI without the LangChain extras). find_spec does not
    # import the module, so it has no side effects.
    if importlib.util.find_spec(provider["import_check"]) is None:
        pytest.skip(f"{provider['import_check']} not installed")

    module_name = f"traigent.examples.providers.{provider['runnable_module']}"
    code = (
        "import importlib;"
        f"m = importlib.import_module({module_name!r});"
        "assert hasattr(m, 'answer'), 'missing answer';"
        "assert hasattr(m.answer, 'optimize'), "
        "'answer is not @traigent.optimize-decorated (SDK surface changed?)';"
        "assert callable(getattr(m, 'main', None)), 'missing main';"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "ENVIRONMENT": "development",
        "TRAIGENT_MOCK_LLM": "true",
        "TRAIGENT_EXAMPLES_NO_INSTALL": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, (
        f"importing {module_name} failed:\n{proc.stdout}\n{proc.stderr}"
    )


def test_get_provider_unknown_raises():
    with pytest.raises(KeyError):
        get_provider("does-not-exist")


# --------------------------------------------------------------------------- #
# ensure_packages bootstrap — never install silently                          #
# --------------------------------------------------------------------------- #


def test_ensure_packages_noop_for_present_package():
    from traigent.examples.providers._bootstrap import ensure_packages

    # sys is always importable; must be a no-op (no SystemExit).
    ensure_packages([("sys", "should-never-be-installed")])


def test_ensure_packages_skips_core_pip_none():
    from traigent.examples.providers._bootstrap import ensure_packages

    # pip_package=None means "core dep" — never install even if absent.
    ensure_packages([("a_module_that_does_not_exist_xyz", None)])


def test_ensure_packages_non_interactive_prints_and_exits(monkeypatch, capsys):
    from traigent.examples.providers import _bootstrap

    monkeypatch.setattr(_bootstrap.sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.delenv("TRAIGENT_EXAMPLES_NO_INSTALL", raising=False)

    with pytest.raises(SystemExit):
        _bootstrap.ensure_packages([("a_module_that_does_not_exist_xyz", "ghost-pkg")])

    err = capsys.readouterr().err
    assert "ghost-pkg" in err
    assert "pip install" in err


def test_ensure_packages_no_install_env_blocks_install(monkeypatch, capsys):
    from traigent.examples.providers import _bootstrap

    # Even on a TTY, the opt-out must prevent any prompt/install.
    monkeypatch.setattr(_bootstrap.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("TRAIGENT_EXAMPLES_NO_INSTALL", "1")

    def _fail_input(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("input() must not be called when NO_INSTALL is set")

    monkeypatch.setattr("builtins.input", _fail_input)

    with pytest.raises(SystemExit):
        _bootstrap.ensure_packages([("a_module_that_does_not_exist_xyz", "ghost-pkg")])
    assert "pip install" in capsys.readouterr().err


def test_ensure_packages_declined_prompt_does_not_install(monkeypatch, capsys):
    from traigent.examples.providers import _bootstrap

    monkeypatch.setattr(_bootstrap.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.delenv("TRAIGENT_EXAMPLES_NO_INSTALL", raising=False)
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")

    def _fail_check_call(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("pip install must not run when the user declines")

    monkeypatch.setattr(_bootstrap.subprocess, "check_call", _fail_check_call)

    with pytest.raises(SystemExit):
        _bootstrap.ensure_packages([("a_module_that_does_not_exist_xyz", "ghost-pkg")])
    assert "pip install" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# End-to-end: a representative mock run actually produces ranked results       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module_name", ["openai", "groq", "nous"])
def test_example_runs_end_to_end_in_mock(module_name):
    """One LangChain (openai), one LiteLLM (groq) and the OAuth Nous Portal
    (nous) example run keyless in mock mode and print a ranked results table
    with a non-zero best score.

    ``nous`` (#1978) additionally exercises the JWT-refresh helper's offline
    path: ``configure_demo_env`` seeds ``NOUS_API_KEY`` (via ``setdefault``),
    so ``get_nous_api_key()`` returns it verbatim with ZERO network. The
    module id == runnable_module for all three, so ``get_provider(module_name)``
    resolves the manifest entry whose ``import_check`` gates the skip (nous ->
    ``langchain_openai``, like openai; groq -> ``litellm``).
    """
    provider = get_provider(module_name)
    if importlib.util.find_spec(provider["import_check"]) is None:
        pytest.skip(f"{provider['import_check']} not installed")

    env = {
        "PATH": __import__("os").environ.get("PATH", ""),
        "HOME": __import__("os").environ.get("HOME", ""),
        "ENVIRONMENT": "development",
        "TRAIGENT_MOCK_LLM": "true",
    }
    proc = subprocess.run(
        [sys.executable, "-m", f"traigent.examples.providers.{module_name}"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"mock run failed:\n{combined[-2000:]}"
    assert "Trial Results" in combined
    assert "★" in combined
