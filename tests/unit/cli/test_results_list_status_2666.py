"""`traigent results list` shows run status; legacy results show unknown (#2666)."""

from __future__ import annotations

from unittest.mock import Mock, patch

from click.testing import CliRunner

from traigent.cli.main import _result_status_label, cli


def _info(name: str, **extra):
    base = {
        "name": name,
        "function_name": "fn",
        "algorithm": "tpe",
        "best_score": 0.5,
        "total_trials": 2,
        "success_rate": 1.0,
        "created_at": "2024-01-01T00:00:00",
    }
    base.update(extra)
    return base


def test_status_label_never_guesses():
    assert _result_status_label("cancelled") == "cancelled"
    assert _result_status_label("CANCELLED") == "cancelled"
    assert _result_status_label(None) == "unknown"
    assert _result_status_label("") == "unknown"
    assert _result_status_label(3) == "unknown"


def test_results_list_shows_status_column(tmp_path):
    with patch("traigent.cli.main.PersistenceManager") as pm:
        pm.return_value = Mock(
            list_results=Mock(
                return_value=[
                    _info("cancelled_run", status="cancelled"),
                    _info("legacy_run"),
                ]
            ),
            base_dir=tmp_path,
        )
        result = CliRunner().invoke(
            cli, ["results", "list", "-d", str(tmp_path / "empty")]
        )
    assert result.exit_code == 0
    assert "Status" in result.output
    assert "cancelled" in result.output
    assert "unknown" in result.output


def test_list_results_reads_status_from_saved_result(tmp_path):
    """A saved cancelled result lists as cancelled; stripped metadata as no status."""
    import json

    from traigent.api.types import OptimizationStatus
    from traigent.utils.persistence import PersistenceManager
    from tests.unit.utils.test_persistence import _sentinel_result

    pm = PersistenceManager(base_dir=tmp_path)
    result = _sentinel_result()
    result.status = OptimizationStatus.CANCELLED
    pm.save_result(result, "cancelled_run")
    pm.save_result(_sentinel_result(), "legacy_run")
    legacy_meta = tmp_path / "legacy_run" / "metadata.json"
    data = json.loads(legacy_meta.read_text())
    data["result_fields"].pop("status", None)
    legacy_meta.write_text(json.dumps(data))

    by_name = {r["name"]: r for r in pm.list_results()}
    assert by_name["cancelled_run"]["status"] == "cancelled"
    assert "status" not in by_name["legacy_run"]
    assert _result_status_label(by_name["legacy_run"].get("status")) == "unknown"
