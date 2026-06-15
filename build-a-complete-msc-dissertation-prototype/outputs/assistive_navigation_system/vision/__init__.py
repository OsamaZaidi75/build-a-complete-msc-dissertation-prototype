"""Vision package for detection, video input, and frame annotation."""

from .config import ModelConfig, canonicalize_label
from .detections import BBox, Detection
from .detector import MockObstacleDetector, YOLOv11ObstacleDetector, create_detector

__all__ = [
    "BBox",
    "Detection",
    "MockObstacleDetector",
    "ModelConfig",
    "YOLOv11ObstacleDetector",
    "canonicalize_label",
    "create_detector",
]
