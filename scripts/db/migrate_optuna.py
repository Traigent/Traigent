#!/usr/bin/env python3
"""Apply Optuna migration scripts to the Traigent database."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple
from urllib.parse import urlparse

from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Optuna database migrations")
    parser.add_argument(
        "--database-url",
        default=os.getenv("TRAIGENT_DATABASE_URL", "sqlite:///:memory:"),
        help="Database connection URL",
    )
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=Path("traigent_schema/optuna_migrations"),
        help="Directory containing migration SQL files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print migration steps without executing them",
    )
    return parser.parse_args()


def _load_migrations(directory: Path) -> Iterable[Path]:
    return sorted(directory.glob("*.sql"))


def _adapt_sql_for_sqlite(sql: str) -> str:
    replacements = {
        "UUID": "TEXT",
        "TIMESTAMPTZ": "TEXT",
        "JSONB": "JSON",
        "DOUBLE PRECISION": "REAL",
        "DEFAULT gen_random_uuid()": "DEFAULT (lower(hex(randomblob(16))))",
        "DEFAULT NOW()": "DEFAULT CURRENT_TIMESTAMP",
    }
    for src, dst in replacements.items():
        sql = sql.replace(src, dst)
    return sql


def _connect(database_url: str) -> Tuple[str, object]:
    parsed = urlparse(database_url)
    scheme = parsed.scheme

    if scheme.startswith("sqlite"):
        if parsed.path in ("", "/") and parsed.netloc:
            path = f"{parsed.netloc}"
        else:
            path = parsed.path or ":memory:"
        if path.startswith("/"):
            db_path = path
        elif path in ("", ":memory:"):
            db_path = ":memory:"
        else:
            db_path = path
        conn = sqlite3.connect(db_path)
        return "sqlite", conn

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "sqlalchemy is required for non-sqlite migrations"
        ) from exc

    engine = create_engine(database_url, future=True)
    return "sqlalchemy", engine


def _execute_sql(engine_type: str, handle: object, sql: str) -> None:
    if engine_type == "sqlite":
        conn = handle
        conn.executescript(sql)
        conn.commit()
        return

    from sqlalchemy import text

    engine = handle
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    with engine.begin() as connection:
        for statement in statements:
            connection.exec_driver_sql(statement)


def apply_migrations(database_url: str, migrations_dir: Path, dry_run: bool) -> None:
    engine_type, handle = _connect(database_url)
    migrations = list(_load_migrations(migrations_dir))

    if not migrations:
        logger.info("No migrations found in %s", migrations_dir)
        return

    logger.info("Applying %s migrations to %s", len(migrations), database_url)

    for migration in migrations:
        sql = migration.read_text()
        adapted_sql = _adapt_sql_for_sqlite(sql) if engine_type == "sqlite" else sql

        if dry_run:
            logger.info("[DRY-RUN] Would apply migration %s", migration.name)
            continue

        logger.info("Applying migration %s", migration.name)
        _execute_sql(engine_type, handle, adapted_sql)

    if engine_type == "sqlite":
        handle.close()


def main() -> None:
    args = parse_args()
    apply_migrations(args.database_url, args.migrations_dir, args.dry_run)


if __name__ == "__main__":
    main()
