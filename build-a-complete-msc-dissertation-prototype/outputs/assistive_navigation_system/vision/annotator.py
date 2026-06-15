"""Frame annotation utilities for dashboard and CLI display."""

from __future__ import annotations

from typing import Any

from navigation.zones import ZoneAssigner
from vision.camera import require_cv2
from vision.detections import Detection


ZONE_COLORS = {
    "left": (64, 180, 255),
    "centre": (50, 220, 100),
    "right": (255, 170, 60),
    "unknown": (190, 190, 190),
}

WARNING_COLORS = {
    "critical": (0, 0, 255),
    "high": (0, 120, 255),
    "medium": (0, 220, 255),
    "low": (80, 220, 80),
    "none": (180, 180, 180),
}


def annotate_frame(frame: Any, detections: list[Detection], draw_zones: bool = True) -> Any:
    """Draw zones, bounding boxes, labels, distance, and warning level."""

    cv2 = require_cv2()
    output = frame.copy()
    height, width = output.shape[:2]
    if draw_zones:
        _draw_zone_overlay(cv2, output, width, height)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox.as_int_tuple()
        color = WARNING_COLORS.get(detection.warning_level, WARNING_COLORS["none"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        distance_text = (
            f"{detection.estimated_distance:.2f}m" if detection.estimated_distance else "n/a"
        )
        label = (
            f"{detection.label} {detection.confidence:.2f} | "
            f"{detection.zone or '?'} | {distance_text}"
        )
        _draw_label(cv2, output, label, x1, max(18, y1 - 8), color)
    return output


def bgr_to_rgb(frame: Any) -> Any:
    """Convert an OpenCV BGR frame to RGB for Streamlit."""

    cv2 = require_cv2()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _draw_zone_overlay(cv2: Any, frame: Any, width: int, height: int) -> None:
    assigner = ZoneAssigner()
    left_boundary, right_boundary = assigner.boundaries_px(width)
    cv2.line(frame, (left_boundary, 0), (left_boundary, height), (255, 255, 255), 1)
    cv2.line(frame, (right_boundary, 0), (right_boundary, height), (255, 255, 255), 1)
    cv2.putText(frame, "LEFT", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ZONE_COLORS["left"], 2)
    cv2.putText(
        frame,
        "CENTRE",
        (left_boundary + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        ZONE_COLORS["centre"],
        2,
    )
    cv2.putText(
        frame,
        "RIGHT",
        (right_boundary + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        ZONE_COLORS["right"],
        2,
    )


def _draw_label(cv2: Any, frame: Any, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(frame, (x, y - text_height - 8), (x + text_width + 6, y + 4), color, -1)
    cv2.putText(frame, label, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
