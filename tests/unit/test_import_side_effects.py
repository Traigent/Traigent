from __future__ import annotations

import json
import os
import socket
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
        timeout=30,
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
        "import os; import sys; import types; "
        "sys.modules['litellm'] = types.SimpleNamespace(); "
        "from traigent.utils import cost_calculator; "
        "print('LITELLM_LOCAL_MODEL_COST_MAP=' + "
        "str(os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP')))",
        env_overrides={"LITELLM_LOCAL_MODEL_COST_MAP": "false"},
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined
    assert "LITELLM_LOCAL_MODEL_COST_MAP=false" in result.stdout, combined


# --- litellm pricing-map pin, checked against litellm's own report -----------
# litellm fetches its pricing map at import through a native client that Python
# socket patches cannot see, so these tests ask litellm where its map came from
# instead of watching the network. They complement the cost_calculator tests
# above (second pin) by covering the pin set by ``import traigent`` itself.

_PIN_VARS = ("LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS")

_PIN_PROBE = f"""
import json, os
import traigent
import litellm
from litellm.litellm_core_utils.get_model_cost_map import (
    get_model_cost_map_source_info,
)
print("PROBE:" + json.dumps({{
    "traigent_file": traigent.__file__,
    "env": {{k: os.environ.get(k) for k in {_PIN_VARS!r}}},
    "source": get_model_cost_map_source_info(),
}}))
"""


def _probe_litellm_pin(tmp_path: Path, extra_env: dict[str, str]) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    # Bound but never listening: connects to this port are refused every time,
    # so even a broken pin cannot reach the internet from this test.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as dead:
        dead.bind(("127.0.0.1", 0))
        dead_url = f"http://127.0.0.1:{dead.getsockname()[1]}"
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(repo_root),
            # No checkout or home .env may mask a missing pin. python-dotenv's
            # PYTHON_DOTENV_DISABLED (dotenv/main.py: _load_dotenv_disabled,
            # honoured at the top of load_dotenv) turns off both Traigent's and
            # litellm's .env loading without taking the SDK's TRAIGENT_SKIP_DOTENV
            # branch, so the ordinary import path is what runs.
            "PYTHON_DOTENV_DISABLED": "1",
            # Dead map URL and proxies for every case (litellm checks the
            # forcing env var before the URL, so pin detection is unaffected).
            "LITELLM_MODEL_COST_MAP_URL": f"{dead_url}/x.json",
            "HTTP_PROXY": dead_url,
            "HTTPS_PROXY": dead_url,
            "ALL_PROXY": dead_url,
            "NO_PROXY": "",
            **extra_env,
        }
        proc = subprocess.run(
            [sys.executable, "-c", _PIN_PROBE],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-2000:]}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE:")]
    assert lines, f"no probe output:\n{proc.stdout[-1000:]}\n{proc.stderr[-1000:]}"
    result = json.loads(lines[-1][len("PROBE:") :])
    # The worktree copy, not an installed one, is what was tested.
    assert (
        Path(result["traigent_file"]).resolve()
        == (repo_root / "traigent" / "__init__.py").resolve()
    )
    return result


def test_import_traigent_pins_litellm_to_local_cost_map(tmp_path: Path) -> None:
    result = _probe_litellm_pin(tmp_path, {})
    for name in _PIN_VARS:
        assert result["env"][name] == "True", f"{name} not pinned by import traigent"
    assert result["source"]["source"] == "local"
    assert result["source"]["is_env_forced"] is True


def test_live_prices_opt_out_leaves_litellm_pin_unset(tmp_path: Path) -> None:
    result = _probe_litellm_pin(tmp_path, {"TRAIGENT_LITELLM_LIVE_PRICES": "1"})
    for name in _PIN_VARS:
        assert result["env"][name] is None, f"{name} set despite live-prices opt-out"
    source = result["source"]
    assert source["is_env_forced"] is False
    # litellm tried the remote path: it names the dead URL and why it fell back.
    assert source["url"].endswith("/x.json")
    assert source["fallback_reason"]
