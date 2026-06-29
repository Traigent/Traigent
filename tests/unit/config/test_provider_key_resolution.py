"""Provider API-key resolution through the shared canonical chain (#1568).

The concrete bug fixed here: ``traigent.get_api_key("google")`` and
``("mistral")`` previously returned ``None`` even though the validator already
read ``GOOGLE_API_KEY``/``GEMINI_API_KEY`` and ``MISTRAL_API_KEY``. Both key
maps now derive their env chains from the canonical provider-support table, so
google/mistral are first-class and the two maps cannot drift.

Covers all three resolution entry points so they stay in lockstep:
  - APIKeyManager.get_api_key       (traigent/config/api_keys.py)
  - traigent.get_api_key (public)   (traigent/api/functions.py -> APIKeyManager)
  - env_config.get_api_key          (traigent/utils/env_config.py)
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from traigent.api.functions import get_api_key as public_get_api_key
from traigent.config.api_keys import APIKeyManager
from traigent.utils.env_config import get_api_key as env_get_api_key

# Repo root: tests/unit/config/<this file> -> parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROVIDER_SUPPORT_SRC = _REPO_ROOT / "traigent" / "config" / "provider_support.py"

_PROVIDER_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "CO_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_API_KEY",
    "TRAIGENT_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Remove every provider env var before each test for full isolation."""
    for var in _PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def manager():
    return APIKeyManager()


@pytest.mark.unit
class TestGoogleKeyResolution:
    """google was the headline #1568 gap in the key manager."""

    def test_google_api_key_via_manager(self, manager, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "g-primary")
        assert manager.get_api_key("google") == "g-primary"

    def test_google_api_key_via_public(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "g-public")
        assert public_get_api_key("google") == "g-public"

    def test_google_api_key_via_env_config(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "g-env")
        assert env_get_api_key("google") == "g-env"

    def test_google_falls_back_to_gemini_api_key(self, manager, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-fallback")
        assert manager.get_api_key("google") == "gemini-fallback"
        assert env_get_api_key("google") == "gemini-fallback"

    def test_google_primary_wins_over_gemini(self, manager, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "g-primary")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-secondary")
        assert manager.get_api_key("google") == "g-primary"

    def test_google_none_when_unset(self, manager):
        assert manager.get_api_key("google") is None
        assert env_get_api_key("google") is None


@pytest.mark.unit
class TestMistralKeyResolution:
    """mistral was the second #1568 gap in the key manager."""

    def test_mistral_api_key_via_manager(self, manager, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "m-key")
        assert manager.get_api_key("mistral") == "m-key"

    def test_mistral_api_key_via_public(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "m-public")
        assert public_get_api_key("mistral") == "m-public"

    def test_mistral_api_key_via_env_config(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "m-env")
        assert env_get_api_key("mistral") == "m-env"

    def test_mistral_none_when_unset(self, manager):
        assert manager.get_api_key("mistral") is None
        assert env_get_api_key("mistral") is None


@pytest.mark.unit
class TestExistingProvidersUnregressed:
    """Same-or-better: previously-working providers keep working."""

    def test_openai_unchanged(self, manager, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        assert manager.get_api_key("openai") == "sk-openai"
        assert env_get_api_key("openai") == "sk-openai"
        assert public_get_api_key("openai") == "sk-openai"

    def test_anthropic_unchanged(self, manager, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        assert manager.get_api_key("anthropic") == "sk-ant"
        assert env_get_api_key("anthropic") == "sk-ant"

    def test_cohere_primary_unchanged(self, manager, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "co-primary")
        assert manager.get_api_key("cohere") == "co-primary"
        assert env_get_api_key("cohere") == "co-primary"

    def test_cohere_co_api_key_fallback_now_reconciled(self, manager, monkeypatch):
        # Additive: the key manager now reads CO_API_KEY too, matching the
        # validator. Previously only COHERE_API_KEY was honored here.
        monkeypatch.setenv("CO_API_KEY", "co-fallback")
        assert manager.get_api_key("cohere") == "co-fallback"
        assert env_get_api_key("cohere") == "co-fallback"

    def test_huggingface_alias_chain_preserved(self, manager, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf-native")
        assert manager.get_api_key("huggingface") == "hf-native"
        assert env_get_api_key("huggingface") == "hf-native"
        assert public_get_api_key("huggingface") == "hf-native"

    def test_huggingface_hub_token_fallback_preserved(self, manager, monkeypatch):
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf-hub")
        assert manager.get_api_key("huggingface") == "hf-hub"

    def test_huggingface_legacy_key_fallback_preserved(self, manager, monkeypatch):
        monkeypatch.setenv("HF_API_KEY", "hf-legacy")
        assert manager.get_api_key("huggingface") == "hf-legacy"


@pytest.mark.unit
class TestBackendAndUnknown:
    """Backend key + unknown/mapping-only providers."""

    def test_env_config_traigent_backend_key_preserved(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_API_KEY", "traigent-backend")
        assert env_get_api_key("traigent") == "traigent-backend"

    def test_manager_environment_priority_preserved(self, manager, monkeypatch):
        # Env must still win over an explicitly-set key (regression guard).
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manager.set_api_key("openai", "direct-key", source="code")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        assert manager.get_api_key("openai") == "env-key"

    def test_mapping_only_provider_has_no_key(self, manager):
        # azure_openai / bedrock are not key-managed.
        assert manager.get_api_key("azure_openai") is None
        assert env_get_api_key("azure_openai") is None
        assert env_get_api_key("bedrock") is None

    def test_unknown_provider_returns_none(self, manager):
        assert manager.get_api_key("nonexistent-provider") is None
        assert env_get_api_key("nonexistent-provider") is None


@pytest.mark.unit
class TestCoreKeyApiHasNoYamlImportDependency:
    """#1568 review fix: the CORE key/config API must import and resolve keys
    with PyYAML absent.

    ``provider_support`` is in the import chain of the public key API
    (``traigent.get_api_key`` -> ``APIKeyManager`` / ``env_config.get_api_key``
    via ``resolve_api_key_from_env``). PyYAML is NOT a core dependency (only
    ``types-PyYAML`` is a dev dep), so a module-level ``import yaml`` there would
    make a minimal install fail importing ``get_api_key`` before any behavior
    runs. The canonical ``PROVIDER_SPECS`` table is a static literal; the only
    yaml-dependent path (``load_models_yaml``) imports yaml lazily and is used
    solely by the drift test and ``model_discovery`` (integrations extra).
    """

    def test_provider_support_has_no_module_level_yaml_import(self) -> None:
        """Static guard: no top-level ``import yaml`` in provider_support.py."""
        tree = ast.parse(_PROVIDER_SUPPORT_SRC.read_text())
        offending: list[str] = []
        for node in tree.body:  # module-level statements only
            if isinstance(node, ast.Import):
                offending += [
                    alias.name
                    for alias in node.names
                    if alias.name.split(".")[0] == "yaml"
                ]
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "yaml":
                    offending.append(node.module or "")
        assert not offending, (
            "provider_support.py must NOT import yaml at module level (found "
            f"{offending}); it is in the core key-API import chain and PyYAML "
            "is not a core dependency (#1568 review)."
        )

    def test_core_key_api_resolves_google_with_yaml_blocked(self) -> None:
        """Behavioral proof: import + resolve a key in a fresh interpreter
        where ``import yaml`` raises ImportError.

        Runs in a subprocess so the (process-global) ``sys.modules`` block and a
        truly-fresh import chain cannot pollute the in-process suite. Setting
        ``sys.modules['yaml'] = None`` makes any subsequent ``import yaml`` raise
        ImportError, exactly simulating a minimal install without PyYAML.
        """
        script = textwrap.dedent(
            """
            import sys
            # A None entry makes any `import yaml` raise ImportError, simulating
            # a minimal install with PyYAML absent.
            sys.modules["yaml"] = None
            from traigent.config.api_keys import APIKeyManager
            val = APIKeyManager().get_api_key("google")
            assert val == "g-subproc", repr(val)
            print("RESOLVED", val)
            """
        )
        env = dict(os.environ)
        env["GOOGLE_API_KEY"] = "g-subproc"
        env.pop("GEMINI_API_KEY", None)
        # Hermetic: don't let the repo's .env perturb resolution.
        env["TRAIGENT_SKIP_DOTENV"] = "1"
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{_REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(_REPO_ROOT)
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert result.returncode == 0, (
            "core key API failed to import/resolve with yaml blocked:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "RESOLVED g-subproc" in result.stdout, result.stdout
