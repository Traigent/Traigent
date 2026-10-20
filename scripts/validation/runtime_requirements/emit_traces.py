#!/usr/bin/env python3
"""Emit synthetic latency traces for one agent (step 3/5 of the PY-1 live-dev script).

Content-free: only a trace name, environment, agent id and timestamps are sent.
``--dry-run`` captures batches locally and sends nothing.

    emit_traces.py --agent-id AGENT --count 250 --latency-ms 3500        # slow
    emit_traces.py --agent-id AGENT --count 250 --latency-ms 200         # healthy
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, UTC

from traigent.observability import ObservabilityClient, ObservabilityConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument(
        "--count", type=int, default=250, help=">= the requirement's min_samples"
    )
    parser.add_argument("--latency-ms", type=int, required=True)
    parser.add_argument("--environment", default="production")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    captured: list[dict] = []
    client = (
        ObservabilityClient(sender=lambda batch: captured.extend(batch))
        if args.dry_run
        else ObservabilityClient(ObservabilityConfig())
    )
    now = datetime.now(UTC)
    for index in range(args.count):
        ended = now - timedelta(seconds=index % 60)
        started = ended - timedelta(milliseconds=args.latency_ms)
        trace_id = client.start_trace(
            "py1-live-dev-latency",
            environment=args.environment,
            execution_context={"agent_id": args.agent_id},
            started_at=started,
        )
        client.end_trace(trace_id, status="completed", ended_at=ended)
    client.flush()
    client.close()
    print(
        f"emitted {args.count} traces at {args.latency_ms} ms for agent {args.agent_id}"
    )
    if args.dry_run:
        print(f"dry run: {len(captured)} payload(s) captured, nothing sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
