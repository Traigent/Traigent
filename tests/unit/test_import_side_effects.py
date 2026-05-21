from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_traigent_does_not_emit_optional_dependency_warnings():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )

    result = subprocess.run(
        [sys.executable, "-c", "import traigent"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined_output
    assert (
        "No LangChain models could be patched for metadata capture"
        not in combined_output
    )
    assert (
        "scikit-learn not available. Bayesian optimization will not work."
        not in combined_output
    )


def test_local_evaluator_patches_langchain_lazily_once(monkeypatch):
    import traigent.evaluators.local as local_module

    calls: list[str] = []

    def fake_patch() -> bool:
        calls.append("patched")
        return False

    monkeypatch.setattr(local_module, "_METADATA_PATCHES_ATTEMPTED", False)
    monkeypatch.setattr(
        local_module, "patch_langchain_for_metadata_capture", fake_patch
    )
    monkeypatch.setattr(
        local_module, "patch_litellm_for_metadata_capture", lambda: True
    )

    local_module.LocalEvaluator(metrics=["accuracy"])
    local_module.LocalEvaluator(metrics=["accuracy"])

    assert calls == ["patched"]


def test_local_evaluator_init_does_not_emit_optional_langchain_warning():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from traigent.evaluators.local import LocalEvaluator; "
            "LocalEvaluator(metrics=['accuracy'])",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined_output
    assert (
        "No LangChain models could be patched for metadata capture"
        not in combined_output
    )


def _run_subprocess(
    code: str, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a clean subprocess with the worktree on PYTHONPATH."""
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    # Strip any inherited setting so each test exercises the default it intends.
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    env.pop("TRAIGENT_OFFLINE_MODE", None)
    if env_overrides:
        env.update(env_overrides)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_traigent_does_not_load_litellm():
    """Regression for #912: a plain `import traigent` must not transitively
    import litellm. Previously the eager import chain (decorators → orchestrator
    → cost_estimator → cost_calculator → litellm) caused `import traigent` to
    make an outbound HTTPS fetch to raw.githubusercontent.com for the model
    pricing map. The chain has since been made lazy, and this test exists to
    prevent it from regressing.
    """
    result = _run_subprocess(
        "import sys; import traigent; "
        "print('litellm_loaded=' + str('litellm' in sys.modules)); "
        "print('cost_calculator_loaded=' + "
        "str('traigent.utils.cost_calculator' in sys.modules))"
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined
    assert "litellm_loaded=False" in result.stdout, combined
    assert "cost_calculator_loaded=False" in result.stdout, combined
    # Defensive: the LiteLLM remote-fetch warning is the most visible symptom
    # in offline/blocked environments if the lazy chain regresses.
    assert "Failed to fetch remote model cost map" not in combined


def test_cost_calculator_defaults_litellm_to_local_cost_map():
    """Regression for #912: when cost_calculator IS imported (e.g. via the
    @optimize decorator path), it must default litellm to its bundled local
    pricing map BEFORE litellm itself is imported. Previously this default
    fired only if the user had set TRAIGENT_OFFLINE_MODE=true, which meant
    a user importing the decorator in a regulated/air-gapped environment
    would silently leak an outbound HTTPS request on the very first import.
    """
    result = _run_subprocess(
        "import os; "
        "from traigent.utils import cost_calculator; "
        "print('LITELLM_LOCAL_MODEL_COST_MAP=' + "
        "str(os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP')))"
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined
    assert "LITELLM_LOCAL_MODEL_COST_MAP=True" in result.stdout, combined
    assert "Failed to fetch remote model cost map" not in combined


def test_cost_calculator_honors_user_opt_out_of_local_cost_map():
    """If a user explicitly sets LITELLM_LOCAL_MODEL_COST_MAP=false in their
    environment before importing the cost calculator, we must NOT override
    their choice. setdefault() guarantees this; this test guards against a
    regression that might switch to plain assignment.
    """
    result = _run_subprocess(
        "import os; "
        "from traigent.utils import cost_calculator; "
        "print('LITELLM_LOCAL_MODEL_COST_MAP=' + "
        "str(os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP')))",
        env_overrides={"LITELLM_LOCAL_MODEL_COST_MAP": "false"},
    )
    assert result.returncode == 0, result.stderr
    assert "LITELLM_LOCAL_MODEL_COST_MAP=false" in result.stdout
