"""Benchmark inference latency and FPS for webcam or video sources."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation.engine import NavigationEngine
from vision.camera import VideoSource, parse_source
from vision.config import ModelConfig
from vision.detector import create_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure detector latency and FPS.")
    parser.add_argument("--source", default=str(ROOT / "samples" / "left_person_right_vehicle.avi"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", default=str(ROOT / "logs" / "benchmark_results.csv"))
    args = parser.parse_args()

    detector = create_detector(
        ModelConfig(model_path=args.model, confidence=args.confidence, use_mock=args.mock),
        fallback_to_mock=True,
    )
    nav_engine = NavigationEngine()
    source = VideoSource(parse_source(args.source))

    latencies_ms: list[float] = []
    frame_count = 0
    start = time.perf_counter()

    try:
        for _, frame in source.frames(max_frames=args.frames):
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            nav_engine.enrich(detections, frame.shape)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            frame_count += 1
    finally:
        source.release()

    elapsed = max(1e-6, time.perf_counter() - start)
    fps = frame_count / elapsed
    summary = {
        "frames": frame_count,
        "mean_latency_ms": round(statistics.mean(latencies_ms), 3) if latencies_ms else 0.0,
        "p95_latency_ms": round(_percentile(latencies_ms, 95), 3) if latencies_ms else 0.0,
        "fps": round(fps, 3),
        "source": args.source,
        "model": "mock" if args.mock else args.model,
    }
    _write_summary(Path(args.output), summary)
    for key, value in summary.items():
        print(f"{key}: {value}")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
