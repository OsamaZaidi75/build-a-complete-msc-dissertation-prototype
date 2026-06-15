"""Virtual 270-degree environmental awareness simulation."""

from __future__ import annotations

from dataclasses import dataclass

from vision.detections import Detection


@dataclass(frozen=True)
class SectorReading:
    """State of one virtual angular sector."""

    index: int
    start_angle: float
    end_angle: float
    centre_angle: float
    object_label: str | None = None
    distance_m: float | None = None
    warning_level: str = "none"
    intensity: float = 0.0


class VirtualNavigationSimulator:
    """Project a forward camera into a 270-degree sector awareness display.

    A normal webcam only sees the forward field of view. This simulator keeps
    the central sectors tied to live detections and shows peripheral sectors as
    virtual awareness placeholders for dissertation demonstration and UI design.
    """

    def __init__(self, sector_count: int = 9, total_fov_degrees: float = 270.0) -> None:
        if sector_count < 3:
            raise ValueError("sector_count must be at least 3")
        self.sector_count = sector_count
        self.total_fov_degrees = total_fov_degrees
        self.start_angle = -total_fov_degrees / 2.0
        self.end_angle = total_fov_degrees / 2.0
        self.sector_width = total_fov_degrees / sector_count

    def sectors(self) -> list[SectorReading]:
        """Return empty sector readings across the simulated field."""

        readings = []
        for idx in range(self.sector_count):
            start = self.start_angle + (idx * self.sector_width)
            end = start + self.sector_width
            readings.append(
                SectorReading(
                    index=idx,
                    start_angle=start,
                    end_angle=end,
                    centre_angle=(start + end) / 2.0,
                )
            )
        return readings

    def build_awareness(
        self,
        detections: list[Detection],
        frame_width: int,
        max_distance_m: float = 8.0,
    ) -> list[SectorReading]:
        """Return sector readings updated with the nearest detection per sector."""

        readings = self.sectors()
        mutable = [reading.__dict__.copy() for reading in readings]

        for detection in detections:
            angle = self._angle_from_bbox(detection, frame_width)
            idx = self._sector_index(angle)
            distance = detection.estimated_distance or max_distance_m
            current_distance = mutable[idx].get("distance_m")
            if current_distance is None or distance < current_distance:
                intensity = max(0.05, min(1.0, (max_distance_m - distance) / max_distance_m))
                mutable[idx].update(
                    {
                        "object_label": detection.label,
                        "distance_m": distance,
                        "warning_level": detection.warning_level,
                        "intensity": round(intensity, 2),
                    }
                )

        return [SectorReading(**reading) for reading in mutable]

    def suggested_direction(self, readings: list[SectorReading]) -> str:
        """Suggest a coarse avoidance direction from sector risk."""

        centre_idx = self.sector_count // 2
        centre_risk = readings[centre_idx].intensity
        if centre_risk < 0.35:
            return "path ahead is relatively clear"

        left_risk = sum(reading.intensity for reading in readings[:centre_idx])
        right_risk = sum(reading.intensity for reading in readings[centre_idx + 1 :])
        if left_risk < right_risk:
            return "prefer slight left"
        if right_risk < left_risk:
            return "prefer slight right"
        return "slow down"

    def _angle_from_bbox(self, detection: Detection, frame_width: int) -> float:
        """Map a detection centre into the live camera's central 90-degree span."""

        centre_x, _ = detection.bbox.center
        normalized_x = max(0.0, min(1.0, centre_x / max(1, frame_width)))
        return -45.0 + (normalized_x * 90.0)

    def _sector_index(self, angle: float) -> int:
        shifted = angle - self.start_angle
        idx = int(shifted // self.sector_width)
        return max(0, min(self.sector_count - 1, idx))
