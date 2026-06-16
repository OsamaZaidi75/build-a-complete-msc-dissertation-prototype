"""WebRTC adapter — bridges streamlit-webrtc with the detection pipeline.

The NavigationVideoProcessor runs YOLO detection, navigation enrichment, and
frame annotation inside the WebRTC callback thread.  Results are exposed via a
thread-safe dict for the Streamlit main thread to poll.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import av
import numpy as np

from feedback.haptics import HapticController
from navigation.engine import NavigationEngine
from navigation.virtual_270 import VirtualNavigationSimulator
from vision.annotator import annotate_frame
from vision.config import ModelConfig
from vision.detector import create_detector


@dataclass
class _SharedState:
    """Thread-safe container for the latest processing results."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    detections: list = field(default_factory=list)
    signal: object = None
    readings: list = field(default_factory=list)
    direction: str = ""
    frame_count: int = 0
    fps: float = 0.0


class NavigationVideoProcessor:
    """streamlit-webrtc VideoProcessorBase implementation.

    Receives av.VideoFrame in recv(), runs the full detection → navigation →
    annotation pipeline, and stores results for the main thread to read.
    """

    def __init__(self) -> None:
        self._state = _SharedState()
        self._detector = None
        self._nav_engine = NavigationEngine()
        self._haptics = HapticController()
        self._simulator = VirtualNavigationSimulator()
        self._start_time: float | None = None

    def configure(self, model_config: ModelConfig) -> None:
        """Initialise the detector.  Called from the main thread before streaming starts."""
        self._detector = create_detector(model_config, fallback_to_mock=True)

    # ------------------------------------------------------------------
    # WebRTC callback (runs in a worker thread)
    # ------------------------------------------------------------------

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process one video frame and return the annotated result."""

        if self._start_time is None:
            self._start_time = time.perf_counter()

        # Convert av.VideoFrame → numpy BGR array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")

        # Run the detection pipeline
        if self._detector is None:
            # Not yet configured — return frame unchanged
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        detections = self._detector.detect(img)
        enriched = self._nav_engine.enrich(detections, img.shape)
        signal = self._haptics.generate(enriched)
        readings = self._simulator.build_awareness(enriched, img.shape[1])
        direction = self._simulator.suggested_direction(readings)

        # Annotate the frame with bounding boxes / zones / labels
        annotated = annotate_frame(img, enriched)

        # Store results thread-safely for main-thread polling
        with self._state.lock:
            self._state.detections = enriched
            self._state.signal = signal
            self._state.readings = readings
            self._state.direction = direction
            self._state.frame_count += 1
            elapsed = time.perf_counter() - self._start_time
            self._state.fps = self._state.frame_count / max(elapsed, 1e-6)

        # Return annotated frame back to the browser
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def get_results(self) -> dict:
        """Return the latest detection results (called from Streamlit thread)."""

        with self._state.lock:
            return {
                "detections": list(self._state.detections),
                "signal": self._state.signal,
                "readings": list(self._state.readings),
                "direction": self._state.direction,
                "frame_count": self._state.frame_count,
                "fps": self._state.fps,
            }
