"""Synthetic frame helpers for simulation mode."""

from __future__ import annotations

import math
from typing import Any

from vision.camera import require_cv2


def make_simulation_frame(frame_index: int, width: int = 960, height: int = 540) -> Any:
    """Create a simple synthetic navigation scene for mock detections."""

    cv2 = require_cv2()
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy is required for simulation frames.") from exc

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (34, 39, 46)
    horizon = int(height * 0.42)
    cv2.rectangle(frame, (0, horizon), (width, height), (45, 52, 58), -1)

    # Perspective lane lines give the UI a navigation context.
    centre_x = width // 2
    for offset in (-260, -120, 120, 260):
        end_x = centre_x + offset
        cv2.line(frame, (centre_x, horizon), (end_x, height), (90, 100, 108), 2)

    pulse = int(10 * math.sin(frame_index / 12.0))
    cv2.circle(frame, (centre_x, horizon + 20 + pulse), 5, (120, 180, 255), -1)
    cv2.putText(
        frame,
        "SIMULATED CAMERA FEED",
        (24, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (210, 220, 230),
        2,
        cv2.LINE_AA,
    )
    return frame
