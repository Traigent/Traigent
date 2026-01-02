#!/usr/bin/env python3
"""Live progress display for the test review orchestrator.

Shows real-time progress updates in the terminal.

Usage:
    python -m tests.optimizer_validation.tools.live_display
    python -m tests.optimizer_validation.tools.live_display --watch
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not available. Install with: pip install rich")


def load_tracking(tracking_file: Path) -> dict:
    """Load the tracking file."""
    with open(tracking_file) as f:
        return json.load(f)


def create_header() -> Text:
    """Create the header."""
    header = Text()
    header.append(
        "╔═══════════════════════════════════════════════════════════════╗\n",
        style="blue",
    )
    header.append("║           ", style="blue")
    header.append("Test Review Orchestrator", style="bold cyan")
    header.append("           ║\n", style="blue")
    header.append(
        "╚═══════════════════════════════════════════════════════════════╝",
        style="blue",
    )
    return header


def create_progress_bar(completed: int, total: int, width: int = 50) -> str:
    """Create a text progress bar."""
    if total == 0:
        return "░" * width
    filled = int(width * completed / total)
    return "█" * filled + "░" * (width - filled)


def create_status_panel(tracking_data: dict) -> Panel:
    """Create the status panel."""
    metadata = tracking_data["metadata"]
    review = metadata["review_progress"]
    validation = metadata["validation_progress"]
    total = metadata["total_tests"]

    completed = review.get("completed", 0)
    in_progress = review.get("in_progress", 0)
    not_started = review.get("not_started", 0)
    flagged = review.get("flagged", 0)

    pct = (completed / total * 100) if total > 0 else 0
    bar = create_progress_bar(completed, total)

    content = Text()
    content.append(f"\nProgress: [{bar}] {pct:.1f}%\n", style="white")
    content.append(f"          {completed}/{total} reviews completed\n\n", style="dim")

    content.append("Review Status:\n", style="bold")
    content.append(f"  ✓ Completed:   {completed:4d}\n", style="green")
    content.append(f"  → In Progress: {in_progress:4d}\n", style="yellow")
    content.append(f"  ○ Not Started: {not_started:4d}\n", style="dim")
    content.append(f"  ⚠ Flagged:     {flagged:4d}\n", style="red")

    content.append("\nValidation Status:\n", style="bold")
    content.append(
        f"  ✓ Approved:      {validation.get('approved', 0):4d}\n", style="green"
    )
    content.append(
        f"  ✗ Rejected:      {validation.get('rejected', 0):4d}\n", style="red"
    )
    content.append(
        f"  → Needs Revision:{validation.get('needs_revision', 0):4d}\n", style="yellow"
    )
    content.append(
        f"  ? Pending:       {validation.get('pending', 0):4d}\n", style="dim"
    )

    return Panel(content, title="Overall Progress", border_style="blue")


def create_category_table(tracking_data: dict) -> Table:
    """Create the category breakdown table."""
    tests = tracking_data["tests"]

    categories: dict[str, dict[str, int]] = {}
    for test in tests:
        file_path = test["identity"]["file"]
        if file_path.startswith("dimensions/"):
            cat = "dimensions"
        elif file_path.startswith("failures/"):
            cat = "failures"
        elif file_path.startswith("interactions/"):
            cat = "interactions"
        else:
            cat = "other"

        if cat not in categories:
            categories[cat] = {
                "completed": 0,
                "in_progress": 0,
                "not_started": 0,
                "total": 0,
            }

        status = test["review"]["status"]
        if status == "completed":
            categories[cat]["completed"] += 1
        elif status == "in_progress":
            categories[cat]["in_progress"] += 1
        else:
            categories[cat]["not_started"] += 1
        categories[cat]["total"] += 1

    table = Table(
        title="Category Breakdown", show_header=True, header_style="bold cyan"
    )
    table.add_column("Category", style="white", width=15)
    table.add_column("Progress", width=25)
    table.add_column("Completed", justify="right", width=10)
    table.add_column("Total", justify="right", width=8)
    table.add_column("%", justify="right", width=6)

    for cat in ["dimensions", "failures", "interactions", "other"]:
        if cat not in categories:
            continue
        data = categories[cat]
        pct = (data["completed"] / data["total"] * 100) if data["total"] > 0 else 0
        bar = create_progress_bar(data["completed"], data["total"], 20)

        style = "green" if pct == 100 else ("yellow" if pct > 0 else "dim")
        table.add_row(
            cat,
            bar,
            str(data["completed"]),
            str(data["total"]),
            f"{pct:.0f}%",
            style=style,
        )

    return table


def create_recent_activity(tracking_data: dict, limit: int = 5) -> Table:
    """Create recent activity table."""
    tests = tracking_data["tests"]

    # Find recently reviewed tests
    recent = []
    for test in tests:
        if test["review"]["reviewed_at"]:
            recent.append(
                {
                    "id": test["id"],
                    "reviewed_at": test["review"]["reviewed_at"],
                    "verdict": test["review"]["overall_verdict"],
                    "reviewer": test["review"]["reviewer_id"],
                }
            )

    # Sort by review time
    recent.sort(key=lambda x: x["reviewed_at"], reverse=True)
    recent = recent[:limit]

    table = Table(title="Recent Activity", show_header=True, header_style="bold cyan")
    table.add_column("Time", width=20)
    table.add_column("Test", width=40, overflow="ellipsis")
    table.add_column("Verdict", width=12)
    table.add_column("Reviewer", width=15)

    for item in recent:
        # Parse and format time
        try:
            dt = datetime.fromisoformat(item["reviewed_at"].replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M:%S")
        except Exception:
            time_str = item["reviewed_at"][:19]

        # Short test name
        test_name = item["id"].split("::")[-1] if "::" in item["id"] else item["id"]

        verdict = item["verdict"] or "?"
        verdict_style = (
            "green" if verdict == "PASS" else ("red" if verdict == "FAIL" else "yellow")
        )

        table.add_row(
            time_str,
            test_name,
            verdict,
            item["reviewer"] or "unknown",
            style=verdict_style if verdict != "PASS" else None,
        )

    if not recent:
        table.add_row("-", "No reviews yet", "-", "-", style="dim")

    return table


def create_dashboard(tracking_data: dict, console: Console) -> None:
    """Create and display the full dashboard."""
    console.clear()

    # Header
    console.print(create_header())
    console.print()

    # Main status panel
    console.print(create_status_panel(tracking_data))
    console.print()

    # Category breakdown
    console.print(create_category_table(tracking_data))
    console.print()

    # Recent activity
    console.print(create_recent_activity(tracking_data))

    # Footer
    console.print(
        f"\n[dim]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
    )
    console.print("[dim]Press Ctrl+C to exit[/dim]")


def watch_mode(tracking_file: Path, interval: float = 2.0) -> None:
    """Watch mode - continuously update display."""
    console = Console()

    try:
        while True:
            tracking_data = load_tracking(tracking_file)
            create_dashboard(tracking_data, console)
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching.[/yellow]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live progress display for test review orchestrator"
    )
    parser.add_argument(
        "--watch", "-w", action="store_true", help="Watch mode - continuously update"
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=2.0, help="Update interval in seconds"
    )
    parser.add_argument(
        "--tracking-file",
        "-f",
        type=Path,
        default=Path(__file__).parent.parent / "test_tracking.json",
        help="Path to tracking file",
    )

    args = parser.parse_args()

    if not RICH_AVAILABLE:
        print(
            "Error: 'rich' library required for display. Install with: pip install rich"
        )
        return

    if not args.tracking_file.exists():
        print(f"Error: Tracking file not found: {args.tracking_file}")
        return

    if args.watch:
        watch_mode(args.tracking_file, args.interval)
    else:
        console = Console()
        tracking_data = load_tracking(args.tracking_file)
        create_dashboard(tracking_data, console)


if __name__ == "__main__":
    main()
