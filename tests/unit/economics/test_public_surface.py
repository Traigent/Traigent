"""Public-surface / import-contract for the economics emitter (WI-B).

Policy: the emitter ships under the documented SUBPACKAGE-ONLY surface
``traigent.economics`` and is intentionally NOT a ``traigent`` root export,
because root symbols are governed by the schema-owned Python/JS parity manifest
(pinned target SHA) and this WI-B Python emitter has no JS counterpart yet. This
test pins that policy so a future root promotion is a deliberate, coordinated
change rather than an accident.
"""

from __future__ import annotations

import traigent
import traigent.economics as economics


def test_client_is_importable_from_the_subpackage_surface() -> None:
    from traigent.economics import EconomicsTelemetryClient

    assert EconomicsTelemetryClient is economics.EconomicsTelemetryClient


def test_client_is_not_a_root_export_by_policy() -> None:
    # Deliberate: kept off the root surface until the schema-owned parity
    # manifest classifies it. See the traigent.economics module docstring.
    assert "EconomicsTelemetryClient" not in traigent.__all__


def test_economics_all_names_are_importable() -> None:
    for name in economics.__all__:
        assert hasattr(economics, name), name


def test_public_error_hierarchy_is_coherent() -> None:
    base = economics.EconomicsTelemetryError
    for name in (
        "EconomicsBatchTooLarge",
        "EconomicsIdempotencyConflict",
        "EconomicsResponseError",
        "EconomicsSchemaUnavailable",
        "EconomicsTelemetryAuthError",
        "EconomicsTelemetryContractError",
        "EconomicsTelemetryTransportError",
        "EgressPolicyError",
    ):
        assert issubclass(getattr(economics, name), base), name
