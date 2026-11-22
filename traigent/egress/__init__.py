"""Local egress relay: a log of what the SDK sends to the Traigent backend.

``traigent egress relay`` runs a loopback-only forwarding proxy that records, for
every request, the method, route shape, body size, body checksum and response
status. Never values, never headers. See :mod:`traigent.egress.log` for the log
format.

This package is imported lazily by the CLI; importing ``traigent`` does not load
:mod:`traigent.egress.relay`.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY
