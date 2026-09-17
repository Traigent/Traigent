"""Tests for environment configuration utilities."""

import os
import sys
from types import SimpleNamespace

import pytest

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = SimpleNamespace(
        load_dotenv=lambda *_args, **_kwargs: None,
        find_dotenv=lambda *_args, **_kwargs: "",
    )

from traigent.utils import env_config


def _reset_env(monkeypatch):
    """Clear critical environment variables to isolate test cases."""
    for key in (
        "JWT_SECRET_KEY",
        "TRAIGENT_MOCK_LLM",
        "TRAIGENT_OFFLINE_MODE",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
        "TRAIGENT_ENVIRONMENT",
        "TRAIGENT_DEV_JWT_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env_config, "_GENERATED_DEV_JWT_SECRET", None, raising=False)


@pytest.mark.parametrize(
    ("env", "expect_error"),
    [("production", True), ("development", False), ("Dev", False)],
)
def test_get_jwt_secret_missing_secret(monkeypatch, env, expect_error):
    """Missing JWT secret should be disallowed in production but generate a secure dev secret otherwise."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", env)

    if expect_error:
        with pytest.raises(ValueError):
            env_config.get_jwt_secret()
    else:
        with pytest.warns(UserWarning):
            secret = env_config.get_jwt_secret()
        assert secret
        assert len(secret) >= env_config._MIN_JWT_SECRET_LENGTH
        # Subsequent calls should return the same generated secret for stability.
        assert env_config.get_jwt_secret() == secret


def test_get_jwt_secret_accepts_mock_mode(monkeypatch):
    """Mock LLM mode should allow fallback secret even if environment not set."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")

    with pytest.warns(UserWarning):
        secret = env_config.get_jwt_secret()
    assert secret
    assert len(secret) >= env_config._MIN_JWT_SECRET_LENGTH
    assert env_config.get_jwt_secret() == secret


def test_get_jwt_secret_returns_existing_value(monkeypatch):
    """Explicit JWT secret should be returned unchanged."""
    _reset_env(monkeypatch)
    value = "a" * 40
    monkeypatch.setenv("JWT_SECRET_KEY", value)
    monkeypatch.setenv("ENVIRONMENT", "production")

    secret = env_config.get_jwt_secret()
    assert secret == value


def test_resolve_environment_name_uses_canonical_key_order(monkeypatch):
    """Environment resolution should honor the documented compatibility keys."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_ENV", "staging")
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", "development")

    assert env_config.resolve_environment_name(default=None) == "staging"

    monkeypatch.setenv("ENVIRONMENT", "production")
    assert env_config.resolve_environment_name(default=None) == "production"


def test_resolve_environment_label_drops_unknown_values(monkeypatch):
    """Telemetry labels should not copy arbitrary env var content."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("TRAIGENT_ENV", "alice@example.com")

    assert env_config.resolve_environment_label(default=None) is None
    assert env_config.resolve_environment_label(default="production") == "production"


def test_treat_as_production_policy_fails_closed_for_unknown_or_unset(monkeypatch):
    """Policy surfaces deny unless the deployment is explicitly non-production."""
    _reset_env(monkeypatch)
    assert env_config.treat_as_production_policy() is True

    monkeypatch.setenv("ENVIRONMENT", "qa")
    assert env_config.treat_as_production_policy() is True


@pytest.mark.parametrize("key", ["ENVIRONMENT", "TRAIGENT_ENV", "TRAIGENT_ENVIRONMENT"])
def test_treat_as_production_policy_accepts_explicit_non_prod_aliases(monkeypatch, key):
    """Legacy environment keys should opt policy checks into non-production only explicitly."""
    _reset_env(monkeypatch)
    monkeypatch.setenv(key, "development")

    assert env_config.treat_as_production_policy() is False


def test_legacy_policy_alias_warns(monkeypatch):
    """The old helper name remains compatible but points callers to the explicit policy name."""
    _reset_env(monkeypatch)

    with pytest.deprecated_call(match="treat_as_production_policy"):
        assert env_config.treat_as_production() is True


def test_get_jwt_secret_warns_on_short_secret(monkeypatch):
    """Short JWT secrets trigger a warning to encourage rotation."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("JWT_SECRET_KEY", "short-secret")

    with pytest.warns(UserWarning):
        assert env_config.get_jwt_secret() == "short-secret"


def test_get_jwt_secret_uses_override(monkeypatch):
    """Custom development override should take precedence."""
    _reset_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("TRAIGENT_DEV_JWT_SECRET", "custom-dev-key")

    jwt_value = env_config.get_jwt_secret()  # pragma: allowlist secret
    assert jwt_value == "custom-dev-key"


def test_get_env_var_masks_value_in_logs(monkeypatch, caplog):
    """Sensitive environment variables should be masked in logs."""
    key = "SECRET_TEST_VAR"
    value = "supersecretvalue"
    monkeypatch.setenv(key, value)

    with caplog.at_level("INFO"):
        retrieved = env_config.get_env_var(key, mask_in_logs=True)

    assert retrieved == value
    assert key in caplog.text
    assert value not in caplog.text


def test_database_and_redis_default_to_local_only_outside_production(monkeypatch):
    """Local service defaults are allowed for SDK development only."""
    _reset_env(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")

    assert env_config.get_database_url() == "postgresql://localhost:5432/traigent"
    assert env_config.get_redis_url() == "redis://localhost:6379"


def test_database_and_redis_are_required_in_production(monkeypatch):
    """Production must not silently fall back to localhost services."""
    _reset_env(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(ValueError, match="DATABASE_URL"):
        env_config.get_database_url()

    with pytest.raises(ValueError, match="REDIS_URL"):
        env_config.get_redis_url()


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        (" TRUE ", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("", False),
    ],
)
def test_is_strict_cost_accounting(monkeypatch, raw_value, expected):
    """Strict cost accounting flag should parse bool-like env values."""
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", raw_value)
    assert env_config.is_strict_cost_accounting() is expected


@pytest.mark.parametrize("raw_value", ["true", "1", "yes", "on", " TRUE "])
def test_backend_offline_accepts_standard_truthy_values(monkeypatch, raw_value):
    """Offline mode should honor the standard truthy env vocabulary."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", raw_value)
    assert env_config.is_backend_offline() is True


def test_backend_offline_accepts_consolidated_offline_alias(monkeypatch):
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE", "1")
    assert env_config.is_backend_offline() is True


class TestLoadDotenvFiles:
    """Traigent/Traigent#1830: the SDK's own dotenv loader must also read a
    project-root ``.env`` discovered from the caller's cwd, not only the
    package-adjacent path (repo root in a dev checkout, ``site-packages/``
    when pip-installed).
    """

    _MARKER_VAR = "TRAIGENT_TEST_1830_PROJECT_MARKER"

    def _write_project_env(self, tmp_path, value):
        (tmp_path / ".env").write_text(f"{self._MARKER_VAR}={value}\n")

    def test_reads_project_root_env_via_cwd(self, tmp_path, monkeypatch):
        """A .env in the caller's project root (not package-adjacent) is
        loaded once the SDK's own loader runs — the pip-installed-user
        scenario the issue reports as silently unmet before this fix."""
        _reset_env(monkeypatch)
        self._write_project_env(tmp_path, "from-project-root")
        monkeypatch.delenv(self._MARKER_VAR, raising=False)
        monkeypatch.chdir(tmp_path)

        try:
            env_config._load_dotenv_files()
            assert os.environ.get(self._MARKER_VAR) == "from-project-root"
        finally:
            os.environ.pop(self._MARKER_VAR, None)

    def test_explicit_env_var_wins_over_project_dotenv(self, tmp_path, monkeypatch):
        """An already-exported real env var is never overridden by the
        project .env (load_dotenv's default, non-overriding behavior)."""
        _reset_env(monkeypatch)
        self._write_project_env(tmp_path, "from-project-root")
        monkeypatch.setenv(self._MARKER_VAR, "from-real-shell-env")
        monkeypatch.chdir(tmp_path)

        env_config._load_dotenv_files()
        assert os.environ.get(self._MARKER_VAR) == "from-real-shell-env"

    def test_skip_dotenv_opts_out_of_project_dotenv_too(self, tmp_path, monkeypatch):
        """TRAIGENT_SKIP_DOTENV must still suppress the new cwd-discovered
        file, exactly as it already suppresses the package-adjacent one."""
        self._write_project_env(tmp_path, "from-project-root")
        monkeypatch.delenv(self._MARKER_VAR, raising=False)
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "1")
        monkeypatch.chdir(tmp_path)

        try:
            env_config._load_dotenv_files()
            assert self._MARKER_VAR not in os.environ
        finally:
            os.environ.pop(self._MARKER_VAR, None)

    def test_survives_deleted_cwd(self, tmp_path, monkeypatch):
        """A deleted/unmounted cwd must degrade to 'no project .env found',
        never crash the loader (review finding: os.getcwd() raising
        FileNotFoundError inside find_dotenv(usecwd=True) used to propagate
        straight out of _load_dotenv_files(), i.e. out of `import traigent`).
        """
        _reset_env(monkeypatch)
        gone = tmp_path / "deleted"
        gone.mkdir()
        monkeypatch.chdir(gone)
        gone.rmdir()

        # Must not raise.
        env_config._load_dotenv_files()

    def test_project_dotenv_bounded_at_marker_directory(self, tmp_path, monkeypatch):
        """A .env at the project marker directory (one level above cwd) is
        still found — the bound is inclusive of the marker directory."""
        _reset_env(monkeypatch)
        project = tmp_path / "project"
        subdir = project / "subdir"
        subdir.mkdir(parents=True)
        (project / "pyproject.toml").write_text("")
        self._write_project_env(project, "from-marker-dir")
        monkeypatch.delenv(self._MARKER_VAR, raising=False)
        monkeypatch.chdir(subdir)

        try:
            env_config._load_dotenv_files()
            assert os.environ.get(self._MARKER_VAR) == "from-marker-dir"
        finally:
            os.environ.pop(self._MARKER_VAR, None)

    def test_project_dotenv_never_crosses_marker_into_ancestor(
        self, tmp_path, monkeypatch
    ):
        """A .env belonging to an unrelated ancestor (past the project
        marker, e.g. a monorepo/workspace root) must never be loaded —
        the walk stops at the marker, it does not cross it."""
        _reset_env(monkeypatch)
        workspace = tmp_path / "workspace"
        project = workspace / "project"
        subdir = project / "subdir"
        subdir.mkdir(parents=True)
        self._write_project_env(workspace, "from-unrelated-ancestor")
        (project / "pyproject.toml").write_text("")
        monkeypatch.delenv(self._MARKER_VAR, raising=False)
        monkeypatch.chdir(subdir)

        env_config._load_dotenv_files()
        assert self._MARKER_VAR not in os.environ

    def test_no_marker_anywhere_checks_cwd_only(self, tmp_path, monkeypatch):
        """When no project marker exists in the whole ancestry, there is no
        trusted boundary, so only cwd itself is checked — an ancestor .env
        with no marker between it and cwd is not loaded either."""
        _reset_env(monkeypatch)
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        self._write_project_env(parent, "from-markerless-ancestor")
        monkeypatch.delenv(self._MARKER_VAR, raising=False)
        monkeypatch.chdir(child)

        env_config._load_dotenv_files()
        assert self._MARKER_VAR not in os.environ
