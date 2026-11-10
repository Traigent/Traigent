"""Issue #3145: a fully closed ObservabilityClient releases its atexit hook, so
it can be garbage collected; an open (or not fully closed) client keeps it, so
the documented best-effort exit flush still runs."""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
import weakref

from traigent.observability import ObservabilityClient, ObservabilityConfig


def _config(**overrides) -> ObservabilityConfig:
    values = {
        "backend_origin": "http://localhost:5000",
        "api_key": "test-key",  # pragma: allowlist secret
        "batch_size": 10,
        "max_buffer_age": 60.0,
        "max_queue_size": 10,
    }
    values.update(overrides)
    return ObservabilityConfig(**values)


def test_closed_clients_are_collectable() -> None:
    refs = []
    for _ in range(5):
        client = ObservabilityClient(_config(), sender=lambda traces: None)
        client.close()
        refs.append(weakref.ref(client))
        del client
    gc.collect()

    assert [ref() for ref in refs] == [None] * 5


def test_open_client_keeps_its_exit_flush_registration() -> None:
    client = ObservabilityClient(_config(), sender=lambda traces: None)
    ref = weakref.ref(client)
    del client
    gc.collect()
    client = ref()
    assert client is not None  # kept alive for the exit flush
    client.close()


def test_exit_flush_still_delivers_for_an_unclosed_dropped_client() -> None:
    script = textwrap.dedent(
        """
        import gc, sys
        from traigent.observability import ObservabilityClient, ObservabilityConfig

        def sender(traces):
            sys.stdout.write(f"delivered={len(traces)}\\n")
            sys.stdout.flush()

        client = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=60.0,
                max_queue_size=10,
            ),
            sender=sender,
        )
        trace_id = client.start_trace("t", trace_id="trace_exit_3145")
        client.end_trace(trace_id)
        del client
        gc.collect()

        closed = ObservabilityClient(
            ObservabilityConfig(
                backend_origin="http://localhost:5000",
                api_key="test-key",  # pragma: allowlist secret
                batch_size=10,
                max_buffer_age=60.0,
                max_queue_size=10,
            ),
            sender=sender,
        )
        closed.close()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "delivered=1" in completed.stdout
