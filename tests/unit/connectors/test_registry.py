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
