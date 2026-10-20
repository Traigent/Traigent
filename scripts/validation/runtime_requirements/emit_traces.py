#!/usr/bin/env python3
"""Emit synthetic latency traces for one agent (step 3/5 of the PY-1 live-dev script).

Content-free: only a trace name, environment, agent id and timestamps are sent.
``--dry-run`` captures batches locally and sends nothing.

    emit_traces.py --agent-key KEY --count 250 --latency-ms 3500             # slow
    emit_traces.py --agent-key KEY --count 250 --latency-ms 200 --window-ago 1   # healthy

``--agent-key`` is the TRACE identity (the agent key the backend attributes traces to), which is
not the registration id (the server ``Agent.id``) used by ``requirements apply``.
``--window-ago N`` spreads the traces over the N-th most recent COMPLETE UTC hour (N >= 1);
the default spreads them over the last minute.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, UTC

from traigent.observability import ObservabilityClient, ObservabilityConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-key", required=True, help="trace attribution key, NOT Agent.id"
    )
    parser.add_argument("--window-ago", type=int, default=0)
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
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    for index in range(args.count):
        if args.window_ago >= 1:
            window_start = hour_start - timedelta(hours=args.window_ago)
            ended = window_start + timedelta(seconds=(index + 1) * 3590 / args.count)
        else:
            ended = now - timedelta(seconds=index % 60)
        started = ended - timedelta(milliseconds=args.latency_ms)
        trace_id = client.start_trace(
            "py1-live-dev-latency",
            environment=args.environment,
            execution_context={"agent_id": args.agent_key},
            started_at=started,
        )
        client.end_trace(trace_id, status="completed", ended_at=ended)
    client.flush()
    client.close()
    print(
        f"emitted {args.count} traces at {args.latency_ms} ms for agent key {args.agent_key}"
    )
    if args.dry_run:
        print(f"dry run: {len(captured)} payload(s) captured, nothing sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
