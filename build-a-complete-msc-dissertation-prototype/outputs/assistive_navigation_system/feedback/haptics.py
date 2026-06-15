"""Haptic feedback simulation for left and right vibration motors."""

from __future__ import annotations

from dataclasses import dataclass

from vision.detections import Detection


@dataclass(frozen=True)
class HapticSignal:
    """Simulated haptic motor state."""

    left_intensity: float = 0.0
    right_intensity: float = 0.0
    pattern: str = "idle"

    @property
    def active(self) -> bool:
        return self.left_intensity > 0.0 or self.right_intensity > 0.0

    def as_text(self) -> str:
        """Return a human-readable haptic state for the dashboard."""

        if not self.active:
            return "No vibration"
        left_pct = int(self.left_intensity * 100)
        right_pct = int(self.right_intensity * 100)
        return f"Left {left_pct}% | Right {right_pct}% ({self.pattern})"


class HapticController:
    """Create left/right vibration signals from enriched detections."""

    def __init__(self, max_distance_m: float = 5.0) -> None:
        self.max_distance_m = max_distance_m

    def generate(self, detections: list[Detection]) -> HapticSignal:
        """Return the strongest haptic signal implied by current obstacles."""

        left = 0.0
        right = 0.0
        strongest_level = "idle"

        for detection in detections:
            if detection.estimated_distance is None or detection.zone is None:
                continue
            intensity = self._intensity(detection.estimated_distance)
            if detection.zone == "left":
                left = max(left, intensity)
            elif detection.zone == "right":
                right = max(right, intensity)
            elif detection.zone == "centre":
                left = max(left, intensity)
                right = max(right, intensity)
            strongest_level = self._more_urgent(strongest_level, detection.warning_level)

        return HapticSignal(round(left, 2), round(right, 2), strongest_level)

    def _intensity(self, distance_m: float) -> float:
        normalized = 1.0 - min(self.max_distance_m, max(0.0, distance_m)) / self.max_distance_m
        return max(0.1, min(1.0, normalized))

    @staticmethod
    def _more_urgent(current: str, candidate: str) -> str:
        order = {"idle": 0, "none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        return candidate if order.get(candidate, 0) > order.get(current, 0) else current
