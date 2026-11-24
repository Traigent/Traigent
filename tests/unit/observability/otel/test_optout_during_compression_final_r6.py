"""An opt-out raised during request preparation still blocks the send (SDK-129 final r6)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.observability.otel import transport as mod
from traigent.observability.otel.transport import TransportSuppressed, UrllibTransport


def test_opt_out_during_gzip_compression_never_opens_a_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(mod, "is_backend_offline", lambda: False)
    real_compress = mod.gzip.compress

    def compress(data: bytes, *args, **kwargs) -> bytes:
        monkeypatch.setenv("TRAIGENT_DISABLE_TELEMETRY", "true")
        return real_compress(data, *args, **kwargs)

    monkeypatch.setattr(mod.gzip, "compress", compress)
    transport = UrllibTransport("https://example.invalid/v1/traces", {})
    transport._opener = MagicMock()  # noqa: SLF001
    with pytest.raises(TransportSuppressed):
        transport.post(b"x", timeout=1.0)
    assert transport._opener.open.call_count == 0  # noqa: SLF001
