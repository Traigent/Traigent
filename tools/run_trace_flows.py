"""Runner for TraceSync instrumentation flows.

Usage:
    python tools/run_trace_flows.py                # run all flows
    python tools/run_trace_flows.py --flows invoker_basic
    python tools/run_trace_flows.py --list         # list available flows

Options:
    --log-path PATH           Override TRACE_SYNC_LOG (default: runtime/traces/runtime.log)
    --sample-rate FLOAT       Override TRACE_SYNC_SAMPLE_RATE (default: 1)
    --trace-enabled/--no-trace-enabled  Toggle TRACE_SYNC_ENABLED (default: true)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TraceSync instrumentation flows")
    parser.add_argument(
        "--flows",
        nargs="*",
        help="Subset of flows to run (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available flows and exit",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=ROOT / "runtime" / "traces" / "runtime.log",
        help="Path for TRACE_SYNC_LOG",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="TRACE_SYNC_SAMPLE_RATE value",
    )
    parser.add_argument(
        "--trace-enabled",
        dest="trace_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable TRACE_SYNC_ENABLED",
    )
    return parser.parse_args()


def configure_env(log_path: Path, sample_rate: float, trace_enabled: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRACE_SYNC_ENABLED", "true" if trace_enabled else "false")
    os.environ.setdefault("TRACE_SYNC_SAMPLE_RATE", str(sample_rate))
    os.environ.setdefault("TRACE_SYNC_LOG", str(log_path))
    os.environ.setdefault("TRAIGENT_TRACE_SYNC", "1")


def select_flows(selected: Iterable[str] | None) -> dict[str, Any]:
    from tools.trace_flows import FLOWS
    from tools.flow_generation_utils import CODE_SYNC_FLOWS

    combined = dict(FLOWS)
    combined.update(CODE_SYNC_FLOWS)

    if not selected:
        return combined

    names = set(selected)
    missing = names - set(combined.keys())
    if missing:
        raise SystemExit(f"Unknown flows: {', '.join(sorted(missing))}")
    return {name: combined[name] for name in selected}


def main() -> None:
    args = parse_args()
    configure_env(args.log_path, args.sample_rate, args.trace_enabled)

    from tools.trace_sync_bootstrap import TRACE_RUNTIME_AVAILABLE, enable_trace_sync
    from tools.trace_flows import list_flows
    from tools.flow_generation_utils import list_codesync_flows

    if args.list:
        flow_names = set(list_flows().keys()) | set(list_codesync_flows().keys())
        for name in sorted(flow_names):
            print(name)
        return

    enable_trace_sync()

    flows = select_flows(args.flows)

    print(f"Trace runtime available: {TRACE_RUNTIME_AVAILABLE}")
    print(f"Trace log: {Path(os.environ['TRACE_SYNC_LOG'])}")
    print(f"Running flows: {', '.join(flows.keys()) or 'none'}")

    for name, flow in flows.items():
        print(f"\n=== Running flow: {name} ===")
        result = flow()
        if asyncio.iscoroutine(result):
            asyncio.run(result)
        print(f"✔ completed: {name}")

    print("\nAll selected flows completed.")


if __name__ == "__main__":
    main()
