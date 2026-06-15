"""Navigation pipeline that enriches detections with spatial context."""

from __future__ import annotations

from dataclasses import dataclass

from navigation.distance import DistanceEstimator
from navigation.risk import warning_level_for_distance, warning_message
from navigation.zones import ZoneAssigner
from vision.detections import Detection


@dataclass
class NavigationEngine:
    """Combine zone assignment, distance estimation, and warning rules."""

    zone_assigner: ZoneAssigner = ZoneAssigner()
    distance_estimator: DistanceEstimator = DistanceEstimator()

    def enrich(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int, int] | tuple[int, int],
    ) -> list[Detection]:
        """Return detections enriched with zone, distance, level, and message."""

        frame_width = frame_shape[1]
        enriched: list[Detection] = []
        for detection in detections:
            zone = self.zone_assigner.zone_for_bbox(detection.bbox, frame_width)
            distance = self.distance_estimator.estimate(detection.label, detection.bbox, frame_shape)
            level = warning_level_for_distance(distance, zone)
            message = warning_message(detection.label, zone, level)
            enriched.append(
                detection.with_updates(
                    zone=zone,
                    estimated_distance=distance,
                    warning_level=level,
                    message=message,
                )
            )
        return enriched
