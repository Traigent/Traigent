"""Top-level ``traigent sync`` command.

First-class, idempotent sync of locally-logged optimization runs to the
Traigent portal, following the ``wandb sync`` convention: running it with no
arguments prints a status summary (what is synced vs. still pending) and uploads
nothing; an already-synced, unchanged run is skipped so re-running never creates
duplicate portal experiments.

    traigent sync                 # status only — no upload
    traigent sync <session_id>    # sync one run (idempotent)
    traigent sync --all           # sync every run still pending
    traigent sync --dry-run ...   # validate + preview, upload nothing
    traigent sync --clean ...     # delete local only after a verified sync
"""

from __future__ import annotations

import json as json_module
import sys
from typing import Any

import click

from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig

_PENDING_STATES = {"unsynced", "partial", "failed"}


def _resolve_api_key(api_key: str | None) -> str | None:
    if api_key:
        return api_key
    from traigent.config.backend_config import BackendConfig

    return BackendConfig.get_api_key()


def _print_status(sync_manager: SyncManager, *, as_json: bool) -> dict[str, Any]:
    status = sync_manager.get_sync_status()
    if as_json:
        click.echo(json_module.dumps(status, indent=2, default=str))
        return status

    click.echo("\n📊 Traigent sync status")
    click.echo("=" * 32)
    click.echo(f"Completed runs : {status['completed_sessions']}")
    click.echo(f"  ✅ synced    : {status.get('synced', 0)}")
    click.echo(f"  ⬆️  unsynced  : {status.get('unsynced', 0)}")
    click.echo(f"  ◐ partial    : {status.get('partial', 0)}")
    click.echo(f"  ❌ failed     : {status.get('failed', 0)}")
    pending = status.get("sync_eligible", 0)
    if pending:
        click.echo(f"\n💡 {pending} run(s) pending. Run `traigent sync --all` to push.")
    else:
        click.echo("\n✨ All completed runs are synced.")
    return status


def _emit_session_result(result: dict[str, Any], *, dry_run: bool) -> None:
    status = result.get("status")
    sid = result.get("session_id")
    if status == "success":
        if dry_run:
            click.echo(
                f"🔍 Ready to sync {sid} ({result.get('trials_converted', 0)} trials)"
            )
        else:
            click.echo(f"✅ Synced {sid}")
            if result.get("cloud_url"):
                click.echo(f"   Portal URL: {result['cloud_url']}")
    elif status == "already_synced":
        click.echo(f"⏭️  Skipped {sid} (already synced — use --force to re-upload)")
    else:
        click.echo(f"❌ {sid}: {status}")
        for error in result.get("errors", []):
            click.echo(f"   • {error}")


@click.command(name="sync")
@click.argument("session_id", required=False)
@click.option("--all", "sync_all", is_flag=True, help="Sync every pending run")
@click.option("--dry-run", is_flag=True, help="Preview without uploading")
@click.option(
    "--force", is_flag=True, help="Re-upload even if already synced and unchanged"
)
@click.option(
    "--clean",
    is_flag=True,
    help="Delete local copies of runs after a verified sync (backup kept)",
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@click.option("--api-key", help="API key (else TRAIGENT_API_KEY / stored credentials)")
def sync_command(
    session_id: str | None,
    sync_all: bool,
    dry_run: bool,
    force: bool,
    clean: bool,
    as_json: bool,
    api_key: str | None,
) -> None:
    """Sync locally-logged optimization runs to the Traigent portal."""
    try:
        config = TraigentConfig.from_environment()
        api_key = _resolve_api_key(api_key)
        sync_manager = SyncManager(config, api_key)

        # No target → status only (wandb `sync` with no args). Needs no key.
        if not session_id and not sync_all:
            _print_status(sync_manager, as_json=as_json)
            return

        # Only real uploads require a key; status and --dry-run do not.
        if not api_key and not dry_run:
            click.echo(
                "❌ API key required for portal sync. Use --api-key or set "
                "TRAIGENT_API_KEY.",
                err=True,
            )
            click.echo("💡 Configure a key with `traigent auth login`.")
            sys.exit(1)

        synced_ids: list[str] = []
        had_errors = False
        if session_id:
            result = sync_manager.sync_session_to_cloud(
                session_id, dry_run=dry_run, force=force
            )
            if as_json:
                click.echo(json_module.dumps(result, indent=2, default=str))
            else:
                _emit_session_result(result, dry_run=dry_run)
            if result.get("status") in {"success", "already_synced"}:
                synced_ids.append(session_id)
            elif not dry_run:
                # A real sync that did not succeed is a failure — exit non-zero
                # so callers (CI, scripts) do not assume the session was uploaded.
                had_errors = True
        else:
            result = sync_manager.sync_all_sessions(dry_run, force=force)
            if as_json:
                click.echo(json_module.dumps(result, indent=2, default=str))
            else:
                click.echo(
                    f"\n📈 Synced {result['synced_successfully']}, "
                    f"skipped {result.get('skipped', 0)}, "
                    f"errors {result['sync_errors']} "
                    f"(of {result['eligible_sessions']} eligible)"
                )
                for session_result in result["session_results"]:
                    _emit_session_result(session_result, dry_run=dry_run)
            synced_ids = [
                r["session_id"]
                for r in result["session_results"]
                if r.get("status") in {"success", "already_synced"}
            ]
            if not dry_run and result.get("sync_errors", 0) > 0:
                had_errors = True

        # --clean: delete local copies only for runs verified synced, with backup.
        if clean and not dry_run and synced_ids:
            verified = _verified_synced(sync_manager, synced_ids)
            if verified:
                cleanup = sync_manager.cleanup_after_sync(verified, keep_backup=True)
                click.echo(
                    f"🧹 Cleaned {cleanup['sessions_deleted']} synced run(s) "
                    "(backed up first)."
                )

        if had_errors:
            sys.exit(1)

    except Exception as e:  # noqa: BLE001 - CLI boundary surfaces a clean message
        click.echo(f"Error syncing to portal: {e}", err=True)
        sys.exit(1)


def _verified_synced(sync_manager: SyncManager, session_ids: list[str]) -> list[str]:
    """Keep only sessions whose persisted sync_state confirms a full sync."""
    verified: list[str] = []
    for sid in session_ids:
        session = sync_manager.storage.load_session(sid)
        if session and (session.sync_state or {}).get("status") == "synced":
            verified.append(sid)
    return verified


def register_sync_command(cli: click.Group) -> None:
    """Register the top-level ``traigent sync`` command on the CLI group."""
    cli.add_command(sync_command)
