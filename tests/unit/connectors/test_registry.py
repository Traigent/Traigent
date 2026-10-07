from importlib.metadata import EntryPoint

from traigent.connectors.registry import discover_connectors


def test_dummy_connector_registers_via_entry_point(monkeypatch):
    entry = EntryPoint(
        name="dummy",
        value="tests.unit.connectors.fixtures.dummy:Connector",
        group="traigent.connectors",
    )
    monkeypatch.setattr(
        "traigent.connectors.registry.entry_points",
        lambda: {"traigent.connectors": [entry]},
    )
    found = discover_connectors()
    assert "dummy" in found
    assert found["dummy"]().kind == "dummy"


def test_registry_contains_failing_entry_point(monkeypatch):
    healthy = EntryPoint(
        name="dummy",
        value="tests.unit.connectors.fixtures.dummy:Connector",
        group="traigent.connectors",
    )

    class BrokenEntry:
        name = "broken"

        @staticmethod
        def load():
            raise ImportError("/customer/private/connector.py")

    monkeypatch.setattr(
        "traigent.connectors.registry.entry_points",
        lambda: {"traigent.connectors": [healthy, BrokenEntry()]},
    )
    found = discover_connectors()
    assert found["dummy"]().kind == "dummy"
    assert found["broken"].reason == "entry_point_load_failed"
    assert "/customer/private" not in repr(found["broken"])
