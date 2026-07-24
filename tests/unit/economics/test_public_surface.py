"""Public-surface / import-contract for the economics emitter (WI-B).

Policy: ``EconomicsTelemetryClient`` is exported both as a ``traigent`` root LAZY
export and from the ``traigent.economics`` subpackage. The root export is true
Python/JS parity — the schema-owned parity manifest classifies the symbol
``matched`` and the JS SDK root-exports it — and it is lazy, so plain
``import traigent`` does not eagerly import ``traigent.economics``. These tests
pin all three properties (root ``__all__`` membership, same-object resolution,
and import laziness).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import traigent
import traigent.economics as economics


def test_client_is_importable_from_the_subpackage_surface() -> None:
    from traigent.economics import EconomicsTelemetryClient

    assert EconomicsTelemetryClient is economics.EconomicsTelemetryClient


def test_client_is_a_root_export() -> None:
    # Parity: the schema-owned manifest classifies EconomicsTelemetryClient as a
    # `matched` root export and the JS SDK root-exports it, so Python does too.
    assert "EconomicsTelemetryClient" in traigent.__all__


def test_root_client_resolves_to_the_same_object_as_the_subpackage() -> None:
    # The lazy root attribute and the subpackage attribute are the SAME class.
    assert traigent.EconomicsTelemetryClient is economics.EconomicsTelemetryClient


def test_root_import_does_not_eagerly_import_economics_subpackage() -> None:
    # Laziness (mirrors tests/unit/test_init_imports.py::test_main_module_import
    # _stays_cold): a plain `import traigent` must not pull in traigent.economics;
    # only accessing the symbol does. Run in a fresh interpreter so an already
    # -imported economics module (from other tests) cannot mask the regression.
    script = textwrap.dedent(
        """
        import sys

        import traigent

        assert "traigent.economics" not in sys.modules, (
            "import traigent eagerly imported traigent.economics"
        )
        # Accessing the lazy root export resolves it (and imports the subpackage).
        client = traigent.EconomicsTelemetryClient
        assert "traigent.economics" in sys.modules
        import traigent.economics as economics
        assert client is economics.EconomicsTelemetryClient
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
