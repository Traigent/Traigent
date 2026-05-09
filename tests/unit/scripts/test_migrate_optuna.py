"""Tests for the Optuna migration runner."""

from __future__ import annotations

import sqlite3

from scripts.db import migrate_optuna
from scripts.db.migrate_optuna import apply_migrations


def test_migration_dry_run(tmp_path):
    db_url = f"sqlite:///{tmp_path/'dryrun.db'}"
    result = apply_migrations(db_url, tmp_path, dry_run=True)
    assert result is None or isinstance(
        result, (int, list)
    )  # Returns None or migration count/list


def test_migration_applies_to_sqlite(tmp_path):
    db_file = tmp_path / "optuna.db"
    db_url = f"sqlite:///{db_file}"

    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    migration = migrations_dir / "0001.sql"
    migration.write_text("""
        CREATE TABLE IF NOT EXISTS optuna_demo (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            created TIMESTAMPTZ DEFAULT NOW()
        );
        """)

    apply_migrations(db_url, migrations_dir, dry_run=False)

    conn = sqlite3.connect(db_file)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='optuna_demo'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_redact_database_url_hides_password_and_query():
    password = "secret"  # pragma: allowlist secret
    database_url = (
        f"postgresql://user:{password}@db.example.com:5432/traigent?sslmode=require"
    )
    redacted = migrate_optuna._redact_database_url(database_url)

    assert "secret" not in redacted
    assert "sslmode=require" not in redacted
    assert redacted == "postgresql://user:***@db.example.com:5432/traigent?***"


def test_apply_migrations_logs_redacted_database_url(tmp_path, monkeypatch, caplog):
    class FakeConnection:
        def close(self):
            pass

    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "0001.sql").write_text("SELECT 1;")
    password = "secret"  # pragma: allowlist secret
    database_url = f"postgresql://user:{password}@db.example.com/traigent"

    monkeypatch.setattr(
        migrate_optuna,
        "_connect",
        lambda url: ("sqlite", FakeConnection()),
    )

    caplog.set_level("INFO")
    migrate_optuna.apply_migrations(database_url, migrations_dir, dry_run=True)

    assert "secret" not in caplog.text
    assert "postgresql://user:***@db.example.com/traigent" in caplog.text
