"""
CLI commands for Traigent Edge Analytics mode operations.
Inspired by DeepEval's approach to local command interface.
"""

# Traceability: CONC-Layer-API CONC-Quality-Usability CONC-Quality-Reliability FUNC-API-ENTRY FUNC-STORAGE REQ-API-001 REQ-STOR-007 SYNC-OptimizationFlow

import json
import os
import sys
from datetime import UTC
from pathlib import Path

import click

from ..cloud.sync_manager import SyncManager
from ..config.types import TraigentConfig
from ..storage.local_storage import LocalStorageManager
from ..utils.incentives import show_upgrade_hint
from ..utils.local_analytics import collect_and_submit_analytics
from ..utils.logging import get_logger

logger = get_logger(__name__)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@click.group(name="edge-analytics")
def edge_analytics_commands() -> None:
    """Edge Analytics mode operations for Traigent optimization sessions."""
    pass


@edge_analytics_commands.command(name="list")
@click.option(
    "--status",
    type=click.Choice(["pending", "running", "completed", "failed"]),
    help="Filter sessions by status",
)
@click.option("--limit", default=10, help="Maximum number of sessions to show")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_sessions(status: str | None, limit: int, output_format: str) -> None:
    """List local optimization sessions."""
    try:
        config = TraigentConfig.from_environment()
        storage = LocalStorageManager(config.get_local_storage_path())

        sessions = storage.list_sessions(status=status)[:limit]

        # Submit analytics when user checks their sessions (privacy-safe)
        if config.enable_usage_analytics and sessions:
            try:
                collect_and_submit_analytics(config)
            except Exception as e:
                logger.debug(f"Analytics submission failed: {e}")

        if output_format == "json":
            summaries = []
            for session in sessions:
                summary = storage.get_session_summary(session.session_id)
                if summary:
                    summaries.append(summary)
            click.echo(json.dumps(summaries, indent=2, default=str))
        else:
            # Table format
            if not sessions:
                click.echo("No sessions found.")
                return

            click.echo(
                f"{'Session ID':<30} {'Function':<20} {'Status':<12} {'Trials':<8} {'Best Score':<12} {'Created'}"
            )
            click.echo("-" * 100)

            for session in sessions:
                summary = storage.get_session_summary(session.session_id)
                if summary:
                    created_at = str(summary["created_at"])[:10]  # Just the date
                    best_score = (
                        f"{summary['best_score']:.3f}"
                        if summary["best_score"]
                        else "N/A"
                    )
                    click.echo(
                        f"{session.session_id:<30} "
                        f"{summary['function_name']:<20} "
                        f"{summary['status']:<12} "
                        f"{summary['completed_trials']:<8} "
                        f"{best_score:<12} "
                        f"{created_at}"
                    )

    except Exception as e:
        click.echo(f"Error listing sessions: {e}", err=True)
        sys.exit(1)


@edge_analytics_commands.command(name="show")
@click.argument("session_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["summary", "detailed", "json"]),
    default="summary",
    help="Detail level",
)
def show_session(session_id: str, output_format: str) -> None:
    """Show details of a specific optimization session."""
    try:
        config = TraigentConfig.from_environment()
        storage = LocalStorageManager(config.get_local_storage_path())

        session = storage.load_session(session_id)
        if not session:
            click.echo(f"Session '{session_id}' not found.", err=True)
            sys.exit(1)

        if output_format == "json":
            session_data = {
                "session_id": session.session_id,
                "function_name": session.function_name,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "status": session.status,
                "total_trials": session.total_trials,
                "completed_trials": session.completed_trials,
                "best_config": session.best_config,
                "best_score": session.best_score,
                "baseline_score": session.baseline_score,
                "optimization_config": session.optimization_config,
                "trials": [
                    {
                        "trial_id": trial.trial_id,
                        "config": trial.config,
                        "score": trial.score,
                        "timestamp": trial.timestamp,
                        "error": trial.error,
                    }
                    for trial in (session.trials or [])
                ],
            }
            click.echo(json.dumps(session_data, indent=2, default=str))

        elif output_format == "detailed":
            summary = storage.get_session_summary(session_id)
            click.echo(f"\n📊 Session Details: {session_id}")
            click.echo("=" * 60)
            click.echo(f"Function: {session.function_name}")
            click.echo(f"Status: {session.status}")
            click.echo(f"Created: {session.created_at}")
            click.echo(f"Updated: {session.updated_at}")
            click.echo(f"Trials: {session.completed_trials}/{session.total_trials}")

            if session.best_score is not None:
                click.echo(f"Best Score: {session.best_score:.4f}")
                improvement = (summary or {}).get("improvement")
                if improvement:
                    click.echo(f"Improvement: {improvement * 100:.1f}%")

            if session.best_config:
                click.echo("\n🎯 Best Configuration:")
                for key, value in session.best_config.items():
                    click.echo(f"  {key}: {value}")

            if session.trials and output_format == "detailed":
                click.echo("\n📈 Trial History:")
                for trial in session.trials[-5:]:  # Show last 5 trials
                    status_icon = "✅" if trial.error is None else "❌"
                    click.echo(
                        f"  {status_icon} Trial {trial.trial_id}: {trial.score:.4f}"
                    )

        else:
            # Summary format
            summary = storage.get_session_summary(session_id)
            click.echo(f"\n📊 {session.function_name} Optimization")
            click.echo(f"Status: {session.status} | Trials: {session.completed_trials}")

            if session.best_score is not None:
                click.echo(f"Best Score: {session.best_score:.4f}")
                improvement = (summary or {}).get("improvement")
                if improvement:
                    click.echo(f"Improvement: {improvement * 100:.1f}%")

            if session.best_config:
                click.echo(f"Best Config: {session.best_config}")

            # Show context-specific upgrade hints for Edge Analytics mode
            if config.is_edge_analytics_mode() and session.status == "completed":
                try:
                    show_upgrade_hint(
                        "cli_usage",
                        trial_count=session.completed_trials,
                        best_score=session.best_score,
                    )
                except Exception as e:
                    logger.debug(f"Failed to show upgrade hint: {e}")

    except Exception as e:
        click.echo(f"Error showing session: {e}", err=True)
        sys.exit(1)


@edge_analytics_commands.command(name="export")
@click.argument("session_id")
@click.option("--output", "-o", help="Output file path")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["json"]),
    default="json",
    help="Export format",
)
def export_session(session_id: str, output: str | None, export_format: str) -> None:
    """Export a session to a file."""
    try:
        config = TraigentConfig.from_environment()
        storage = LocalStorageManager(config.get_local_storage_path())
        storage_root = storage.storage_path.resolve()
        exports_root = (storage_root / "exports").resolve()

        session = storage.load_session(session_id)
        if not session:
            click.echo(f"Session '{session_id}' not found.", err=True)
            sys.exit(1)

        # Determine output path
        if output:
            candidate = Path(output).expanduser()
            if not candidate.is_absolute():
                candidate = exports_root / candidate
        else:
            candidate = exports_root / f"{session_id}.json"

        candidate = candidate.resolve()

        try:
            candidate.relative_to(storage_root)
        except ValueError:
            click.echo(
                f"Error: Output path must reside within the local storage directory ({storage_root})",
                err=True,
            )
            sys.exit(1)

        candidate.parent.mkdir(parents=True, exist_ok=True)

        # Export session
        success = storage.export_session(session_id, str(candidate), export_format)

        if success:
            click.echo(f"✅ Session exported to: {candidate}")
        else:
            click.echo("❌ Failed to export session", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error exporting session: {e}", err=True)
        sys.exit(1)


@edge_analytics_commands.command(name="delete")
@click.argument("session_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def delete_session(session_id: str, force: bool) -> None:
    """Delete a local optimization session."""
    try:
        config = TraigentConfig.from_environment()
        storage = LocalStorageManager(config.get_local_storage_path())

        session = storage.load_session(session_id)
        if not session:
            click.echo(f"Session '{session_id}' not found.", err=True)
            sys.exit(1)

        if not force:
            click.confirm(
                f"Are you sure you want to delete session '{session_id}' ({session.function_name})?",
                abort=True,
            )

        success = storage.delete_session(session_id)

        if success:
            click.echo(f"✅ Session '{session_id}' deleted")
        else:
            click.echo("❌ Failed to delete session", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error deleting session: {e}", err=True)
        sys.exit(1)


@edge_analytics_commands.command(name="cleanup")
@click.option("--days", default=30, help="Delete sessions older than N days")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be deleted without deleting"
)
def cleanup_sessions(days: int, dry_run: bool) -> None:
    """Clean up old optimization sessions."""
    try:
        config = TraigentConfig.from_environment()
        storage = LocalStorageManager(config.get_local_storage_path())

        if dry_run:
            # Show what would be deleted
            from datetime import datetime, timedelta

            cutoff_date = datetime.now(UTC) - timedelta(days=days)

            old_sessions = []
            for session in storage.list_sessions():
                session_date = datetime.fromisoformat(session.created_at)
                # Normalize to timezone-aware for safe comparison
                if session_date.tzinfo is None:
                    session_date = session_date.replace(tzinfo=UTC)
                if session_date < cutoff_date:
                    old_sessions.append(session)

            if old_sessions:
                click.echo(
                    f"Would delete {len(old_sessions)} sessions older than {days} days:"
                )
                for session in old_sessions:
                    click.echo(f"  - {session.session_id} ({session.function_name})")
            else:
                click.echo(f"No sessions older than {days} days found.")
        else:
            deleted_count = storage.cleanup_old_sessions(days)
            click.echo(f"✅ Cleaned up {deleted_count} sessions older than {days} days")

    except Exception as e:
        click.echo(f"Error cleaning up sessions: {e}", err=True)
        sys.exit(1)


@edge_analytics_commands.command(name="info")
def storage_info() -> None:
    """Show information about local storage."""
    try:
        config = TraigentConfig.from_environment()
        storage = LocalStorageManager(config.get_local_storage_path())

        info = storage.get_storage_info()

        click.echo("\n📁 Traigent Local Storage")
        click.echo("=" * 40)
        click.echo(f"Storage Path: {info['storage_path']}")
        click.echo(f"Total Sessions: {info['total_sessions']}")
        click.echo(f"Total Trials: {info['total_trials']}")
        click.echo(f"Storage Size: {info['storage_size_mb']} MB")

        # Show session breakdown by status
        sessions = storage.list_sessions()
        status_counts: dict[str, int] = {}
        for session in sessions:
            status_counts[session.status] = status_counts.get(session.status, 0) + 1

        if status_counts:
            click.echo("\n📊 Session Breakdown:")
            for status, count in status_counts.items():
                click.echo(f"  {status}: {count}")

        # Show upgrade hint
        completed_count = status_counts.get("completed", 0)
        if completed_count > 0:
            click.echo("\n💡 Upgrade to Traigent Cloud:")
            click.echo(f"   • Sync your {completed_count} completed sessions")
            click.echo("   • Get advanced analytics and visualizations")
            click.echo("   • Enable team collaboration")
            click.echo("   Run 'traigent login' to get started")

    except Exception as e:
        click.echo(f"Error getting storage info: {e}", err=True)
        sys.exit(1)


@edge_analytics_commands.command(name="config")
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--set-path", help="Set custom storage path")
@click.option("--reset", is_flag=True, help="Reset to default configuration")
def manage_config(show: bool, set_path: str | None, reset: bool) -> None:
    """Manage Edge Analytics mode configuration."""
    try:
        if show:
            config = TraigentConfig.from_environment()
            click.echo("\n⚙️  Traigent Local Mode Configuration")
            click.echo("=" * 45)
            click.echo(f"Execution Mode: {config.execution_mode}")
            click.echo(f"Storage Path: {config.get_local_storage_path()}")
            click.echo(f"Minimal Logging: {config.minimal_logging}")
            click.echo(f"Auto Sync: {config.auto_sync}")

            # Show environment variables
            click.echo("\n🌍 Environment Variables:")
            env_vars = [
                (
                    "TRAIGENT_EDGE_ANALYTICS_MODE",
                    os.getenv("TRAIGENT_EDGE_ANALYTICS_MODE", "not set"),
                ),
                (
                    "TRAIGENT_RESULTS_FOLDER",
                    os.getenv("TRAIGENT_RESULTS_FOLDER", "not set"),
                ),
                (
                    "TRAIGENT_MINIMAL_LOGGING",
                    os.getenv("TRAIGENT_MINIMAL_LOGGING", "not set"),
                ),
                ("TRAIGENT_AUTO_SYNC", os.getenv("TRAIGENT_AUTO_SYNC", "not set")),
            ]

            for name, value in env_vars:
                click.echo(f"  {name}: {value}")

        elif set_path:
            # Set custom storage path
            path = Path(set_path).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)

            click.echo("# Add to your shell profile (.bashrc, .zshrc, etc.):")
            click.echo(f"export TRAIGENT_RESULTS_FOLDER='{path}'")
            click.echo("\n# Or set for current session:")
            click.echo(f"export TRAIGENT_RESULTS_FOLDER='{path}'")

        elif reset:
            click.echo(
                "# To reset to default configuration, unset environment variables:"
            )
            click.echo("unset TRAIGENT_RESULTS_FOLDER")
            click.echo("unset TRAIGENT_EDGE_ANALYTICS_MODE")
            click.echo("unset TRAIGENT_MINIMAL_LOGGING")
            click.echo("unset TRAIGENT_AUTO_SYNC")

        else:
            click.echo("Use --show to display current configuration")
            click.echo("Use --set-path PATH to set custom storage path")
            click.echo("Use --reset for reset instructions")

    except Exception as e:
        click.echo(f"Error managing configuration: {e}", err=True)
        sys.exit(1)


@edge_analytics_commands.command(name="sync")
@click.option("--api-key", help="API key for cloud service")
@click.option("--session-id", help="Sync specific session (if not provided, syncs all)")
@click.option("--dry-run", is_flag=True, help="Preview sync without uploading")
@click.option(
    "--cleanup", is_flag=True, help="Clean up local sessions after successful sync"
)
def sync_to_cloud(
    api_key: str | None, session_id: str | None, dry_run: bool, cleanup: bool
) -> None:
    """Sync local optimization sessions to Traigent Cloud."""
    try:
        config = TraigentConfig.from_environment()

        # Get API key from environment if not provided
        if not api_key:
            from traigent.config.backend_config import BackendConfig

            api_key = BackendConfig.get_api_key()

        # Check for API key before creating SyncManager
        if not api_key and not dry_run:
            click.echo(
                "❌ API key required for cloud sync. Use --api-key or set TRAIGENT_API_KEY environment variable.",
                err=True,
            )
            click.echo("💡 Configure a key with `traigent auth login` or export TRAIGENT_API_KEY.")
            sys.exit(1)

        sync_manager = SyncManager(config, api_key)

        # Show sync status first
        status = sync_manager.get_sync_status()
        click.echo("\n📊 Sync Status")
        click.echo("=" * 30)
        click.echo(f"Total sessions: {status['total_sessions']}")
        click.echo(f"Completed sessions: {status['completed_sessions']}")
        click.echo(f"Sync eligible: {status['sync_eligible']}")
        click.echo(f"Total trials: {status['total_trials']}")

        if status["sync_eligible"] == 0:
            click.echo(
                "\n💡 No completed sessions to sync. Complete some optimizations first!"
            )
            return

        if dry_run:
            click.echo("\n🔍 Dry Run Mode - Preview Only")

        # Sync sessions
        if session_id:
            # Sync specific session
            click.echo(f"\n🔄 Syncing session: {session_id}")
            result = sync_manager.sync_session_to_cloud(session_id, dry_run)

            if result["status"] == "success":
                click.echo("✅ Session synced successfully!")
                if not dry_run:
                    click.echo(f"   Cloud URL: {result.get('cloud_url', 'N/A')}")
            else:
                click.echo("❌ Sync failed:")
                for error in result["errors"]:
                    click.echo(f"   • {error}")
        else:
            # Sync all sessions
            click.echo(
                f"\n🔄 Syncing all {status['sync_eligible']} eligible sessions..."
            )

            if not dry_run and not click.confirm("Continue with sync to cloud?"):
                click.echo("Sync cancelled.")
                return

            result = sync_manager.sync_all_sessions(dry_run)

            click.echo("\n📈 Sync Results:")
            click.echo(f"   Total sessions: {result['total_sessions']}")
            click.echo(f"   Synced successfully: {result['synced_successfully']}")
            click.echo(f"   Errors: {result['sync_errors']}")

            if result["sync_errors"] > 0:
                click.echo("\n❌ Sessions with errors:")
                for session_result in result["session_results"]:
                    if session_result["status"] != "success":
                        click.echo(
                            f"   • {session_result['session_id']}: {session_result.get('errors', ['Unknown error'])}"
                        )

            # Cleanup if requested and successful
            if cleanup and not dry_run and result["synced_successfully"] > 0:
                successful_sessions = [
                    r["session_id"]
                    for r in result["session_results"]
                    if r["status"] == "success"
                ]

                if successful_sessions and click.confirm(
                    f"\nClean up {len(successful_sessions)} successfully synced local sessions?"
                ):
                    cleanup_result = sync_manager.cleanup_after_sync(
                        successful_sessions, keep_backup=True
                    )
                    click.echo(
                        f"✅ Cleaned up {cleanup_result['sessions_deleted']} sessions (backed up first)"
                    )

        # Show cloud analytics preview
        if not dry_run and status["sync_eligible"] > 0:
            sync_manager.get_cloud_analytics_preview()
            click.echo("\n🎯 Available in Traigent Cloud:")
            click.echo("   • Cross-function optimization insights")
            click.echo("   • Performance trend analysis")
            click.echo("   • Team collaboration features")
            click.echo("   • Advanced visualization dashboard")

    except Exception as e:
        click.echo(f"Error syncing to cloud: {e}", err=True)
        sys.exit(1)


@edge_analytics_commands.command(name="preview-cloud")
def preview_cloud_benefits() -> None:
    """Preview benefits of upgrading to Traigent Cloud."""
    try:
        config = TraigentConfig.from_environment()
        sync_manager = SyncManager(config)

        # Get current status
        status = sync_manager.get_sync_status()
        analytics = sync_manager.get_cloud_analytics_preview()

        click.echo("\n🚀 Traigent Cloud Benefits Preview")
        click.echo("=" * 50)

        if status["completed_sessions"] > 0:
            click.echo("\n📊 Your Current Local Usage:")
            click.echo(f"   • {status['completed_sessions']} completed optimizations")
            click.echo(f"   • {status['total_trials']} total trials executed")
            click.echo(
                f"   • {analytics.get('functions_optimized', 0)} functions optimized"
            )

            improvement = analytics.get("average_improvement")
            if improvement:
                click.echo(
                    f"   • {improvement * 100:.1f}% average improvement achieved"
                )

        click.echo("\n🌟 Upgrade to Traigent Cloud for:")

        cloud_features = [
            (
                "🧠 Advanced Algorithms",
                "Bayesian optimization vs. random search (3-5x faster)",
            ),
            ("📈 Web Dashboard", "Beautiful visualizations and progress tracking"),
            ("👥 Team Collaboration", "Share optimizations across your organization"),
            ("🔄 Unlimited Trials", "No 20-trial limit like Edge Analytics mode"),
            ("⚡ Parallel Execution", "Run multiple trials simultaneously"),
            ("📊 Advanced Analytics", "Cross-function insights and trend analysis"),
            ("🔗 API Integration", "REST API for programmatic access"),
            ("🛡️ Enterprise Security", "SOC 2 compliance and data encryption"),
            ("📞 Priority Support", "Direct access to optimization experts"),
        ]

        for feature, description in cloud_features:
            click.echo(f"   {feature}")
            click.echo(f"     {description}")

        if status["completed_sessions"] > 0:
            click.echo("\n💰 Estimated Value for Your Usage:")
            time_saved = analytics.get("estimated_dashboard_value", {}).get(
                "time_saved_reviewing_results", "0 minutes"
            )
            click.echo(f"   • Time saved reviewing results: {time_saved}")
            click.echo(f"   • Advanced insights from {status['total_trials']} trials")
            click.echo(
                f"   • Team collaboration on {status['completed_sessions']} optimizations"
            )

        click.echo("\n🎯 Ready to upgrade?")
        click.echo("   1. Get API key: https://traigent.ai/signup")
        click.echo("   2. Run: traigent local sync --api-key YOUR_KEY")
        click.echo("   3. Access dashboard: https://app.traigent.ai")

    except Exception as e:
        click.echo(f"Error previewing cloud benefits: {e}", err=True)
        sys.exit(1)


# Add to main CLI in traigent/cli/main.py
def register_edge_analytics_commands(cli_group: click.Group) -> None:
    """Register Edge Analytics commands with the main CLI."""
    cli_group.add_command(edge_analytics_commands)
