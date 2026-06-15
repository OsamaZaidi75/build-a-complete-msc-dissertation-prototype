"""CLI runner for webcam/video assistive navigation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feedback.audio import AudioFeedback
from feedback.haptics import HapticController
from feedback.logger import DetectionLogger
from navigation.engine import NavigationEngine
from vision.annotator import annotate_frame
from vision.camera import VideoSource, parse_source, require_cv2
from vision.config import ModelConfig
from vision.detector import create_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the assistive navigation pipeline.")
    parser.add_argument("--source", default="0", help="Webcam index or video path.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO11 checkpoint path.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic simulation detections.")
    parser.add_argument("--display", action="store_true", help="Display an OpenCV preview window.")
    parser.add_argument("--audio", action="store_true", help="Enable text-to-speech warnings.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames; 0 means no limit.")
    parser.add_argument("--log", default=str(ROOT / "logs" / "navigation_events.csv"), help="CSV log path.")
    args = parser.parse_args()

    cv2 = require_cv2()
    source = VideoSource(parse_source(args.source))
    detector = create_detector(
        ModelConfig(model_path=args.model, confidence=args.confidence, use_mock=args.mock),
        fallback_to_mock=True,
    )
    nav_engine = NavigationEngine()
    haptics = HapticController()
    audio = AudioFeedback(enabled=args.audio)
    logger = DetectionLogger(args.log)

    start = time.perf_counter()
    frame_total = 0
    max_frames = args.max_frames or None

    try:
        for frame_index, frame in source.frames(max_frames=max_frames):
            detections = detector.detect(frame)
            enriched = nav_engine.enrich(detections, frame.shape)
            signal = haptics.generate(enriched)
            logger.log(enriched)
            audio.speak_detections(enriched)
            frame_total = frame_index + 1

            if args.display:
                annotated = annotate_frame(frame, enriched)
                cv2.imshow("Assistive Navigation Prototype", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            warning = ", ".join(d.message or d.label for d in enriched[:3]) or "clear"
            print(f"frame={frame_index} detections={len(enriched)} haptic='{signal.as_text()}' {warning}")
    finally:
        source.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = max(1e-6, time.perf_counter() - start)
    print(f"Processed {frame_total} frames in {elapsed:.2f}s ({frame_total / elapsed:.2f} FPS)")


if __name__ == "__main__":
    main()
