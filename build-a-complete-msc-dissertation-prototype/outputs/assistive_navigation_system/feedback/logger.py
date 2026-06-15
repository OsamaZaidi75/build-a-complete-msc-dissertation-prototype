"""CSV logging for navigation detections."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from vision.detections import Detection


class DetectionLogger:
    """Append detection events to the required dissertation CSV schema."""

    fieldnames = ["timestamp", "object", "zone", "estimated_distance", "warning_level"]

    def __init__(self, path: str | Path = "logs/navigation_events.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def log(self, detections: list[Detection]) -> None:
        """Append one CSV row for each enriched detection."""

        if not detections:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            for detection in detections:
                writer.writerow(detection.to_log_row(timestamp))

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
