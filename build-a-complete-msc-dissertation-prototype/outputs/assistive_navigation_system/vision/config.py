"""Configuration and class vocabulary for YOLO11 obstacle detection."""

from __future__ import annotations

import os
from dataclasses import dataclass


SUPPORTED_TARGETS = {"person", "chair", "table", "wall", "door", "vehicle", "stairs"}

# Ultralytics COCO labels plus common custom labels mapped into dissertation labels.
CLASS_ALIASES = {
    "person": "person",
    "people": "person",
    "chair": "chair",
    "couch": "chair",
    "sofa": "chair",
    "bench": "chair",
    "dining table": "table",
    "table": "table",
    "desk": "table",
    "wall": "wall",
    "door": "door",
    "stairs": "stairs",
    "stair": "stairs",
    "staircase": "stairs",
    "steps": "stairs",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "motorcycle": "vehicle",
    "motorbike": "vehicle",
    "bicycle": "vehicle",
    "train": "vehicle",
    "van": "vehicle",
}

# Approximate real-world widths used only for relative monocular distance estimates.
KNOWN_WIDTH_METERS = {
    "person": 0.45,
    "chair": 0.55,
    "table": 1.20,
    "wall": 2.00,
    "door": 0.90,
    "vehicle": 1.80,
    "stairs": 1.20,
}


@dataclass(frozen=True)
class ModelConfig:
    """Runtime options for detector construction."""

    model_path: str = os.getenv("YOLO_MODEL", "yolo11n.pt")
    confidence: float = 0.35
    iou: float = 0.45
    image_size: int = 640
    device: str | None = None
    use_mock: bool = False


def canonicalize_label(raw_label: str) -> str | None:
    """Map a raw detector label into the supported obstacle vocabulary."""

    normalized = raw_label.strip().lower().replace("_", " ")
    return CLASS_ALIASES.get(normalized)
