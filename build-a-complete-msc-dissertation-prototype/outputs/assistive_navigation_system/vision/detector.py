"""YOLO11 and simulation detectors."""

from __future__ import annotations

import math
from typing import Any

from vision.config import ModelConfig, SUPPORTED_TARGETS, canonicalize_label
from vision.detections import BBox, Detection


class YOLOv11ObstacleDetector:
    """Ultralytics YOLO11 detector wrapper.

    The wrapper accepts any YOLO11 detection checkpoint. `yolo11n.pt` is the
    default for lightweight real-time demonstration. Custom checkpoints can add
    labels such as wall, door, and stairs.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run `pip install -r requirements.txt` "
                "or enable mock mode with `--mock`."
            ) from exc

        self.model = YOLO(self.config.model_path)
        self.names = getattr(self.model, "names", {})

    def detect(self, frame: Any) -> list[Detection]:
        """Run YOLO11 inference and return normalized obstacle detections."""

        results = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []
        return self._parse_result(results[0])

    def _parse_result(self, result: Any) -> list[Detection]:
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        xyxy = _tensor_to_list(getattr(boxes, "xyxy", []))
        classes = _tensor_to_list(getattr(boxes, "cls", []))
        confidences = _tensor_to_list(getattr(boxes, "conf", []))

        for box, cls_id, confidence in zip(xyxy, classes, confidences):
            raw_label = self._name_for_class(int(cls_id))
            label = canonicalize_label(raw_label)
            if label not in SUPPORTED_TARGETS:
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    bbox=BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    raw_label=raw_label,
                )
            )
        return detections

    def _name_for_class(self, class_id: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, class_id))
        if isinstance(self.names, list) and 0 <= class_id < len(self.names):
            return str(self.names[class_id])
        return str(class_id)


class MockObstacleDetector:
    """Deterministic detector for demos, tests, and machines without YOLO weights."""

    def __init__(self) -> None:
        self.frame_index = 0

    def detect(self, frame: Any) -> list[Detection]:
        """Return synthetic obstacles that move across the frame."""

        height, width = frame.shape[:2]
        t = self.frame_index
        self.frame_index += 1
        phase = math.sin(t / 18.0)

        person_x = int(width * (0.14 + 0.04 * phase))
        vehicle_x = int(width * (0.68 + 0.06 * math.cos(t / 22.0)))
        centre_width = int(width * (0.20 + 0.04 * math.sin(t / 25.0)))
        centre_x1 = int((width - centre_width) / 2)

        return [
            Detection(
                label="person",
                confidence=0.91,
                bbox=BBox(person_x, height * 0.30, person_x + width * 0.16, height * 0.86),
                raw_label="person",
            ),
            Detection(
                label="vehicle",
                confidence=0.86,
                bbox=BBox(vehicle_x, height * 0.42, vehicle_x + width * 0.23, height * 0.82),
                raw_label="car",
            ),
            Detection(
                label="stairs" if (t // 80) % 2 == 0 else "door",
                confidence=0.79,
                bbox=BBox(centre_x1, height * 0.48, centre_x1 + centre_width, height * 0.94),
                raw_label="stairs",
            ),
        ]


def create_detector(config: ModelConfig | None = None, fallback_to_mock: bool = True) -> Any:
    """Create a YOLO11 detector, optionally falling back to deterministic simulation."""

    resolved = config or ModelConfig()
    if resolved.use_mock:
        return MockObstacleDetector()
    try:
        return YOLOv11ObstacleDetector(resolved)
    except RuntimeError:
        if not fallback_to_mock:
            raise
        return MockObstacleDetector()


def _tensor_to_list(value: Any) -> list[Any]:
    """Convert a torch/numpy/list-like object to a plain Python list."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
