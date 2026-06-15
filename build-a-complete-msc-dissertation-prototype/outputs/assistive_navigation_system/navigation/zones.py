"""Left, centre, and right camera-zone assignment."""

from __future__ import annotations

from dataclasses import dataclass

from vision.detections import BBox


@dataclass(frozen=True)
class ZoneAssigner:
    """Assign detections to three vertical camera zones."""

    left_boundary: float = 1.0 / 3.0
    right_boundary: float = 2.0 / 3.0

    def zone_for_bbox(self, bbox: BBox, frame_width: int) -> str:
        """Return `left`, `centre`, or `right` based on the box centre x-coordinate."""

        if frame_width <= 0:
            return "unknown"
        centre_x, _ = bbox.center
        normalized_x = centre_x / frame_width
        if normalized_x < self.left_boundary:
            return "left"
        if normalized_x > self.right_boundary:
            return "right"
        return "centre"

    def boundaries_px(self, frame_width: int) -> tuple[int, int]:
        """Return pixel x-positions for left/centre and centre/right boundaries."""

        return (int(frame_width * self.left_boundary), int(frame_width * self.right_boundary))
