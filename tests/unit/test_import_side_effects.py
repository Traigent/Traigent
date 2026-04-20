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

    monkeypatch.setattr(local_module, "_LANGCHAIN_PATCH_ATTEMPTED", False)
    monkeypatch.setattr(
        local_module, "patch_langchain_for_metadata_capture", fake_patch
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
