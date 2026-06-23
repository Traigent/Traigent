"""Unit tests for the local chart-render helper.

Asserts a file is produced from canonical payloads (run_pareto / run_correlations)
and that the helper renders FROM the payload rather than recomputing analytics.
"""

from __future__ import annotations

import importlib.util

import pytest

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

pytestmark = pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed"
)


@pytest.fixture()
def pareto_payload() -> dict[str, object]:
    """A frozen-v0 run_pareto contract instance."""
    return {
        "run_id": "run_123",
        "project_id": "proj_abc",
        "measures": {
            "quality": {"label": "Accuracy", "unit": "%"},
            "cost": {"label": "Cost", "unit": "USD"},
            "latency": {"label": "Latency", "unit": "ms"},
        },
        "frontier": [
            {
                "config_id": "cfg_1",
                "rank_on_frontier": 1,
                "metrics": {"quality": 0.9, "cost": 0.02, "latency": 800},
                "is_knee": True,
                "tradeoff_summary": "Best balance.",
            },
            {
                "config_id": "cfg_2",
                "rank_on_frontier": 2,
                "metrics": {"quality": 0.95, "cost": 0.05, "latency": 1200},
                "is_knee": False,
                "tradeoff_summary": "Highest quality.",
            },
        ],
        "dominated": [
            {
                "config_id": "cfg_3",
                "dominated_by": "cfg_1",
                "reason": "Worse on both axes.",
                "metrics": {"quality": 0.8, "cost": 0.04, "latency": 1500},
            }
        ],
        "shape": "convex",
        "warnings": [],
    }


@pytest.fixture()
def correlations_payload() -> dict[str, object]:
    """A frozen-v0 run_correlations contract instance."""
    return {
        "run_id": "run_123",
        "method": "pearson",
        "sample_size": 42,
        "measure_correlations": [
            {
                "x": "quality",
                "y": "cost",
                "r": 0.62,
                "p_value": 0.01,
                "strength": "moderate",
            },
            {
                "x": "quality",
                "y": "latency",
                "r": -0.31,
                "p_value": 0.2,
                "strength": "weak",
            },
        ],
        "parameter_correlations": [],
        "warnings": [],
    }


class TestRenderPareto:
    def test_writes_png_file(self, pareto_payload, tmp_path) -> None:
        from traigent.analytics.render import render_chart

        out = tmp_path / "pareto.png"
        path = render_chart(pareto_payload, "run_pareto", str(out))

        assert path == str(out.resolve())
        assert out.exists()
        assert out.stat().st_size > 0
        # PNG magic bytes.
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_writes_svg_when_requested(self, pareto_payload, tmp_path) -> None:
        from traigent.analytics.render import render_chart

        out = tmp_path / "pareto.svg"
        path = render_chart(pareto_payload, "run_pareto", str(out))

        assert out.exists()
        assert "<svg" in out.read_text(encoding="utf-8")[:2000]
        assert path == str(out.resolve())

    def test_default_output_path_under_cwd(
        self, pareto_payload, tmp_path, monkeypatch
    ) -> None:
        from traigent.analytics.render import render_chart

        monkeypatch.chdir(tmp_path)
        path = render_chart(pareto_payload, "run_pareto")

        assert path.endswith(".png")
        assert "run_123" in path
        from pathlib import Path

        assert Path(path).exists()

    def test_empty_frontier_raises(self, tmp_path) -> None:
        from traigent.analytics.render import ChartRenderError, render_chart

        payload = {"run_id": "r", "frontier": [], "dominated": []}
        with pytest.raises(ChartRenderError, match="no plottable points"):
            render_chart(payload, "run_pareto", str(tmp_path / "x.png"))

    def test_does_not_recompute__skips_points_missing_metrics(self, tmp_path) -> None:
        """A frontier point with no usable metrics is skipped, not invented."""
        from traigent.analytics.render import render_chart

        payload = {
            "run_id": "r",
            "measures": {},
            "frontier": [
                {"config_id": "good", "metrics": {"quality": 0.9, "cost": 0.01}},
                {"config_id": "bad", "metrics": {"quality": None}},
            ],
            "dominated": [],
        }
        out = tmp_path / "p.png"
        # Renders fine using only the one plottable point; the incomplete point
        # is dropped rather than back-filled with a computed value.
        path = render_chart(payload, "run_pareto", str(out))
        assert out.exists()
        assert path == str(out.resolve())


class TestRenderCorrelations:
    def test_writes_file_from_correlations(
        self, correlations_payload, tmp_path
    ) -> None:
        from traigent.analytics.render import render_chart

        out = tmp_path / "corr.png"
        path = render_chart(correlations_payload, "run_correlations", str(out))

        assert out.exists()
        assert out.stat().st_size > 0
        assert path == str(out.resolve())

    def test_empty_correlations_raises(self, tmp_path) -> None:
        from traigent.analytics.render import ChartRenderError, render_chart

        payload = {"run_id": "r", "measure_correlations": []}
        with pytest.raises(ChartRenderError, match="no plottable"):
            render_chart(payload, "run_correlations", str(tmp_path / "x.png"))


class TestRenderValidation:
    def test_unsupported_kind_raises(self, tmp_path) -> None:
        from traigent.analytics.render import ChartRenderError, render_chart

        with pytest.raises(ChartRenderError, match="Unsupported chart kind"):
            render_chart({}, "histogram", str(tmp_path / "x.png"))  # type: ignore[arg-type]

    def test_unsupported_extension_raises(self, pareto_payload, tmp_path) -> None:
        from traigent.analytics.render import ChartRenderError, render_chart

        with pytest.raises(ChartRenderError, match="output_path must end in"):
            render_chart(pareto_payload, "run_pareto", str(tmp_path / "x.pdf"))

    def test_non_mapping_payload_raises(self, tmp_path) -> None:
        from traigent.analytics.render import ChartRenderError, render_chart

        with pytest.raises(ChartRenderError, match="must be a mapping"):
            render_chart(["not", "a", "dict"], "run_pareto")  # type: ignore[arg-type]
