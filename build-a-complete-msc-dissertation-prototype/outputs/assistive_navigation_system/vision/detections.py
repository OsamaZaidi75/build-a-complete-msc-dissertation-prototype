"""Shared detection data structures used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class BBox:
    """Bounding box in absolute pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass(frozen=True)
class Detection:
    """Obstacle detection enriched by navigation and feedback modules."""

    label: str
    confidence: float
    bbox: BBox
    zone: str | None = None
    estimated_distance: float | None = None
    warning_level: str = "none"
    message: str | None = None
    raw_label: str | None = None

    def with_updates(self, **changes: Any) -> "Detection":
        """Return a copy with selected fields replaced."""

        return replace(self, **changes)

    def to_log_row(self, timestamp: str) -> dict[str, str | float]:
        """Serialize the detection to the required CSV log schema."""

        return {
            "timestamp": timestamp,
            "object": self.label,
            "zone": self.zone or "unknown",
            "estimated_distance": round(self.estimated_distance or 0.0, 3),
            "warning_level": self.warning_level,
        }
