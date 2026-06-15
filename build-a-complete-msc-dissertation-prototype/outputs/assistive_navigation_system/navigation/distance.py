"""Relative monocular distance estimation for detected obstacles."""

from __future__ import annotations

from dataclasses import dataclass

from vision.config import KNOWN_WIDTH_METERS
from vision.detections import BBox


@dataclass(frozen=True)
class DistanceEstimator:
    """Estimate relative object distance from apparent bounding-box size.

    This is intentionally lightweight for a single RGB camera. It is not a
    calibrated depth sensor. Values are useful for ranking hazard urgency and
    dissertation demonstration, especially after empirical calibration.
    """

    focal_length_px: float = 700.0
    min_distance_m: float = 0.35
    max_distance_m: float = 8.0

    def estimate(self, label: str, bbox: BBox, frame_shape: tuple[int, int, int] | tuple[int, int]) -> float:
        """Return an approximate distance in meters for an obstacle."""

        frame_height = frame_shape[0]
        frame_width = frame_shape[1]
        known_width = KNOWN_WIDTH_METERS.get(label, 0.8)
        apparent_width = max(1.0, bbox.width)
        pinhole_distance = (known_width * self.focal_length_px) / apparent_width

        # Objects lower in the image are usually nearer on a forward-facing camera.
        bottom_ratio = max(0.0, min(1.0, bbox.y2 / max(1, frame_height)))
        floor_proximity = self.max_distance_m - (bottom_ratio * (self.max_distance_m - self.min_distance_m))

        # Blend scale and floor-position cues to reduce jitter from wide objects.
        distance = (0.72 * pinhole_distance) + (0.28 * floor_proximity)
        return round(max(self.min_distance_m, min(self.max_distance_m, distance)), 2)
