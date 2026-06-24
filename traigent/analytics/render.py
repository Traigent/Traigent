"""Local chart rendering for backend analytics payloads.

This module turns a canonical, backend-produced analytics payload (a
``run_pareto`` or ``run_correlations`` document, per the frozen v0 contracts)
into an image file on disk and returns the path.

It is a **pure presentation layer**: it renders pixels *from the numbers the
backend already computed*. It never recomputes a frontier, a correlation, a
p-value, or any other analytic — doing so would risk the chart disagreeing with
the backend's own report. If a value is missing from the payload, the point is
skipped, not invented.

``matplotlib`` is imported lazily inside :func:`render_chart` so it stays an
optional dependency (``pip install 'traigent[analytics]'`` or
``traigent[visualization]``) and the base SDK install is not bloated.
"""

# Traceability: CONC-Layer-Infra FUNC-ANALYTICS

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

ChartKind = Literal["run_pareto", "run_correlations"]

_SUPPORTED_KINDS: frozenset[str] = frozenset({"run_pareto", "run_correlations"})
_DEFAULT_FORMAT = "png"
_SUPPORTED_FORMATS: frozenset[str] = frozenset({"png", "svg"})

_MATPLOTLIB_INSTALL_MESSAGE = (
    "Chart rendering requires matplotlib. "
    "Install it with: pip install 'traigent[analytics]'"
)


class ChartRenderError(ValueError):
    """Raised when a payload cannot be rendered into a chart."""


def _import_pyplot() -> Any:
    """Import matplotlib's pyplot lazily with a non-interactive backend."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - covered by import guard test
        raise ChartRenderError(_MATPLOTLIB_INSTALL_MESSAGE) from exc

    # Force a headless backend so rendering works in agent/CI environments with
    # no display. Set before pyplot is imported.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _resolve_output_path(
    output_path: str | Path | None,
    *,
    kind: str,
    run_id: str | None,
    fmt: str,
) -> Path:
    if output_path is not None:
        path = Path(output_path).expanduser()
        if path.suffix.lower().lstrip(".") not in _SUPPORTED_FORMATS:
            raise ChartRenderError(
                f"output_path must end in one of {sorted(_SUPPORTED_FORMATS)}: {path}"
            )
        return path

    safe_run = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in (run_id or "run")
    )
    return Path.cwd() / f"{kind}.{safe_run}.{fmt}"


def _as_float(value: Any) -> float | None:
    """Best-effort numeric coercion; returns None when not a finite number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
    else:
        return None
    # Reject NaN/inf so matplotlib never plots a meaningless point.
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _require_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ChartRenderError("payload must be a mapping (decoded JSON object).")
    return payload


def _resolve_axis_metric_key(measures: Any, axis: str) -> str:
    """Resolve a canonical axis name to the payload's actual metric key."""
    if not isinstance(measures, dict):
        return axis
    resolved = measures.get(axis)
    if isinstance(resolved, str) and resolved:
        return resolved
    return axis


def _render_pareto(plt: Any, payload: dict[str, Any]) -> Any:
    """Plot a cost/quality scatter from a ``run_pareto`` payload.

    Frontier points are emphasized (the knee is annotated); dominated points are
    drawn faded. Axes are read straight from each config's ``metrics`` — nothing
    is recomputed.
    """
    measures = payload.get("measures") or {}
    quality_key = _resolve_axis_metric_key(measures, "quality")
    cost_key = _resolve_axis_metric_key(measures, "cost")

    fig, ax = plt.subplots(figsize=(7, 5))

    frontier = payload.get("frontier") or []
    dominated = payload.get("dominated_points") or payload.get("dominated") or []

    plotted_any = False
    knee_xy: tuple[float, float] | None = None

    front_x: list[float] = []
    front_y: list[float] = []
    for point in frontier:
        if not isinstance(point, dict):
            continue
        metrics = point.get("metrics") or {}
        cost = _as_float(metrics.get(cost_key))
        quality = _as_float(metrics.get(quality_key))
        if cost is None or quality is None:
            continue
        front_x.append(cost)
        front_y.append(quality)
        plotted_any = True
        if point.get("is_knee"):
            knee_xy = (cost, quality)
        label = point.get("config_id")
        if label:
            ax.annotate(
                str(label),
                (cost, quality),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )

    dom_x: list[float] = []
    dom_y: list[float] = []
    for point in dominated:
        if not isinstance(point, dict):
            continue
        metrics = point.get("metrics") or {}
        cost = _as_float(metrics.get(cost_key))
        quality = _as_float(metrics.get(quality_key))
        if cost is None or quality is None:
            continue
        dom_x.append(cost)
        dom_y.append(quality)
        plotted_any = True

    if dom_x:
        ax.scatter(dom_x, dom_y, c="#bbbbbb", marker="x", label="dominated", zorder=2)
    if front_x:
        # Sort frontier by cost so the connecting line reads left-to-right.
        order = sorted(range(len(front_x)), key=lambda i: front_x[i])
        sx = [front_x[i] for i in order]
        sy = [front_y[i] for i in order]
        ax.plot(sx, sy, color="#1f77b4", linewidth=1.0, alpha=0.6, zorder=3)
        ax.scatter(
            front_x, front_y, c="#1f77b4", marker="o", label="frontier", zorder=4
        )
    if knee_xy is not None:
        ax.scatter(
            [knee_xy[0]],
            [knee_xy[1]],
            facecolors="none",
            edgecolors="#d62728",
            s=180,
            linewidths=2,
            label="knee",
            zorder=5,
        )

    if not plotted_any:
        raise ChartRenderError(
            "run_pareto payload had no plottable points "
            f"(need frontier/dominated entries with metrics.{cost_key} "
            f"and metrics.{quality_key})."
        )

    quality_meta = measures.get(quality_key) if isinstance(measures, dict) else None
    cost_meta = measures.get(cost_key) if isinstance(measures, dict) else None
    ax.set_xlabel(_axis_label("cost", cost_meta))
    ax.set_ylabel(_axis_label("quality", quality_meta))
    run_id = payload.get("run_id") or "run"
    shape = payload.get("shape")
    title = f"Pareto frontier — {run_id}"
    if shape:
        title = f"{title} ({shape})"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


def _axis_label(default: str, meta: Any) -> str:
    if isinstance(meta, dict):
        name = meta.get("label") or meta.get("name")
        unit = meta.get("unit")
        if name and unit:
            return f"{name} ({unit})"
        if name:
            return str(name)
    return default


def _render_correlations(plt: Any, payload: dict[str, Any]) -> Any:
    """Plot a bar chart of measure correlations from a ``run_correlations`` payload.

    Each bar is one ``measure_correlations`` entry's ``r`` value. Values are read
    directly from the payload; nothing is recomputed.
    """
    correlations = payload.get("measure_correlations") or []
    labels: list[str] = []
    values: list[float] = []
    for entry in correlations:
        if not isinstance(entry, dict):
            continue
        r = _as_float(entry.get("r"))
        if r is None:
            continue
        x = entry.get("x")
        y = entry.get("y")
        labels.append(f"{x}↔{y}" if x and y else str(entry.get("strength", "?")))
        values.append(r)

    if not values:
        raise ChartRenderError(
            "run_correlations payload had no plottable measure_correlations "
            "entries with a numeric 'r'."
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    positions = list(range(len(values)))
    ax.bar(positions, values, color=colors)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    method = payload.get("method")
    run_id = payload.get("run_id") or "run"
    sample = payload.get("sample_size")
    title = f"Measure correlations — {run_id}"
    if method or sample is not None:
        suffix = ", ".join(
            part
            for part in (
                str(method) if method else None,
                f"n={sample}" if sample is not None else None,
            )
            if part
        )
        title = f"{title} ({suffix})"
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


def render_chart(
    payload: dict[str, Any],
    kind: ChartKind,
    output_path: str | Path | None = None,
    *,
    fmt: str | None = None,
) -> str:
    """Render a canonical analytics payload to an image file and return its path.

    Args:
        payload: A backend-produced ``run_pareto`` or ``run_correlations``
            document (the frozen v0 contract). Values are plotted as-is; this
            function never recomputes analytics.
        kind: ``"run_pareto"`` or ``"run_correlations"``.
        output_path: Destination file. When ``None`` a name is derived from the
            kind and ``run_id`` and written under the current working directory.
            The extension (``.png`` / ``.svg``) determines the format.
        fmt: Optional explicit format (``"png"`` or ``"svg"``) used only when
            ``output_path`` is ``None``.

    Returns:
        The absolute path to the written image, as a string.

    Raises:
        ChartRenderError: If ``kind`` is unsupported, ``matplotlib`` is missing,
            the format is unsupported, or the payload has no plottable data.
    """
    if kind not in _SUPPORTED_KINDS:
        raise ChartRenderError(
            f"Unsupported chart kind {kind!r}. "
            f"Supported kinds: {sorted(_SUPPORTED_KINDS)}."
        )
    payload = _require_mapping(payload)

    chosen_fmt = (fmt or _DEFAULT_FORMAT).lower().lstrip(".")
    if chosen_fmt not in _SUPPORTED_FORMATS:
        raise ChartRenderError(
            f"Unsupported format {chosen_fmt!r}. "
            f"Supported formats: {sorted(_SUPPORTED_FORMATS)}."
        )

    resolved = _resolve_output_path(
        output_path,
        kind=kind,
        run_id=(
            payload.get("run_id") if isinstance(payload.get("run_id"), str) else None
        ),
        fmt=chosen_fmt,
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)

    plt = _import_pyplot()
    if kind == "run_pareto":
        fig = _render_pareto(plt, payload)
    else:
        fig = _render_correlations(plt, payload)

    try:
        fig.savefig(str(resolved))
    finally:
        plt.close(fig)

    logger.debug("Rendered %s chart to %s", kind, resolved)
    return str(resolved.resolve())
