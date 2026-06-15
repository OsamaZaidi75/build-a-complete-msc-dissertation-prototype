"""Reusable Streamlit dashboard visualization helpers."""

from __future__ import annotations

from typing import Iterable

from feedback.haptics import HapticSignal
from navigation.virtual_270 import SectorReading
from vision.detections import Detection


SessionHistory = list[dict[str, object]]


def detection_table_rows(detections: Iterable[Detection]) -> list[dict[str, object]]:
    """Convert detections into dashboard table rows."""

    rows = []
    for detection in detections:
        rows.append(
            {
                "object": detection.label,
                "zone": detection.zone,
                "distance_m": detection.estimated_distance,
                "confidence": round(detection.confidence, 3),
                "warning": detection.warning_level,
                "message": detection.message,
            }
        )
    return rows


def haptic_bar(signal: HapticSignal) -> str:
    """Return a compact text bar for left/right haptic intensities."""

    left = "#" * int(signal.left_intensity * 10)
    right = "#" * int(signal.right_intensity * 10)
    return f"L [{left:<10}] R [{right:<10}]"


def build_sector_figure(readings: list[SectorReading]):
    """Build a Plotly polar chart for 270-degree environmental awareness."""

    import plotly.graph_objects as go

    colors = []
    labels = []
    intensities = []
    theta = []
    widths = []

    for reading in readings:
        theta.append(reading.centre_angle)
        widths.append(reading.end_angle - reading.start_angle)
        intensities.append(max(0.08, reading.intensity))
        label = reading.object_label or "clear"
        labels.append(
            f"{reading.start_angle:.0f} to {reading.end_angle:.0f} deg<br>"
            f"{label}<br>{reading.warning_level}"
        )
        colors.append(_sector_color(reading.warning_level, reading.intensity))

    fig = go.Figure(
        data=[
            go.Barpolar(
                r=intensities,
                theta=theta,
                width=widths,
                marker_color=colors,
                marker_line_color="white",
                marker_line_width=1,
                opacity=0.9,
                hovertext=labels,
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                range=[-135, 135],
                tickmode="array",
                tickvals=[-135, -90, -45, 0, 45, 90, 135],
            ),
        ),
        showlegend=False,
    )
    return fig


def _sector_color(level: str, intensity: float) -> str:
    if intensity <= 0.0:
        return "#4f5b66"
    return {
        "critical": "#d62828",
        "high": "#f77f00",
        "medium": "#fcbf49",
        "low": "#2a9d8f",
    }.get(level, "#607d8b")


def build_session_chart(history: SessionHistory):
    """Build a Plotly line chart of per-frame detection count and nearest distance."""

    import plotly.graph_objects as go

    frames = [row["frame"] for row in history]
    counts = [row["count"] for row in history]
    nearest = [row["nearest"] for row in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=counts,
            mode="lines",
            name="Detections",
            line=dict(color="#4e9af1", width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=nearest,
            mode="lines",
            name="Nearest (m)",
            line=dict(color="#f77f00", width=2, dash="dot"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12),
        yaxis=dict(title="Detections", rangemode="nonnegative"),
        yaxis2=dict(
            title="Distance (m)",
            overlaying="y",
            side="right",
            rangemode="nonnegative",
        ),
        xaxis=dict(title="Frame"),
        plot_bgcolor="#1e2128",
        paper_bgcolor="#1e2128",
        font=dict(color="#cdd6f4"),
    )
    return fig


def build_warning_level_chart(history: SessionHistory):
    """Build a stacked bar chart of warning levels per frame."""

    import plotly.graph_objects as go

    frames = [row["frame"] for row in history]
    levels = ("critical", "high", "medium", "low")
    level_colors = {"critical": "#d62828", "high": "#f77f00", "medium": "#fcbf49", "low": "#2a9d8f"}
    traces = []
    for level in levels:
        counts = [row.get(f"level_{level}", 0) for row in history]
        if any(c > 0 for c in counts):
            traces.append(
                go.Bar(
                    x=frames,
                    y=counts,
                    name=level.capitalize(),
                    marker_color=level_colors[level],
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="stack",
        height=200,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(title="Frame"),
        yaxis=dict(title="Detections"),
        plot_bgcolor="#1e2128",
        paper_bgcolor="#1e2128",
        font=dict(color="#cdd6f4"),
    )
    return fig
