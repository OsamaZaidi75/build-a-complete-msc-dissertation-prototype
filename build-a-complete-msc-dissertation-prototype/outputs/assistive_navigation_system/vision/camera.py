"""OpenCV video-source helpers for webcam and file inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def require_cv2():
    """Import OpenCV with a helpful installation error."""

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required. Run `pip install -r requirements.txt`.") from exc
    return cv2


@dataclass
class VideoSource:
    """Iterator wrapper around `cv2.VideoCapture`."""

    source: str | int = 0

    def __post_init__(self) -> None:
        cv2 = require_cv2()
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

    @property
    def fps(self) -> float:
        fps = float(self.capture.get(self.cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0.0 else 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        width = int(self.capture.get(self.cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return width, height

    def frames(self, max_frames: int | None = None) -> Iterator[tuple[int, object]]:
        """Yield `(frame_index, frame)` until the source ends or `max_frames` is reached."""

        idx = 0
        while max_frames is None or idx < max_frames:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def release(self) -> None:
        self.capture.release()


def parse_source(source: str) -> str | int:
    """Parse CLI/dashboard source values into OpenCV-compatible inputs."""

    if source.lower() in {"webcam", "camera"}:
        return 0
    if source.isdigit():
        return int(source)
    return str(Path(source))
