"""Streamlit dashboard for the assistive navigation prototype."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from feedback.audio import AudioFeedback
from feedback.haptics import HapticController
from feedback.logger import DetectionLogger
from navigation.engine import NavigationEngine
from navigation.virtual_270 import VirtualNavigationSimulator
from ui.components import (
    build_sector_figure,
    build_session_chart,
    build_warning_level_chart,
    detection_table_rows,
    haptic_bar,
)
from vision.annotator import annotate_frame, bgr_to_rgb
from vision.camera import VideoSource, parse_source, require_cv2
from vision.config import ModelConfig
from vision.detector import create_detector
from vision.synthetic import make_simulation_frame


def main() -> None:
    st.set_page_config(page_title="Assistive Navigation YOLO11", layout="wide")

    tab_live, tab_eval, tab_about = st.tabs(["Live Navigation", "Evaluation", "About"])

    with tab_live:
        _render_live_tab()

    with tab_eval:
        _render_evaluation_tab()

    with tab_about:
        _render_about_tab()


# ---------------------------------------------------------------------------
# Live Navigation tab
# ---------------------------------------------------------------------------

def _render_live_tab() -> None:
    st.header("AI-Powered Assistive Navigation System")
    st.caption("YOLO11 obstacle awareness prototype for visually impaired users")

    with st.sidebar:
        st.header("Configuration")
        source_mode = st.selectbox("Source", ["Simulation", "Webcam", "Video file", "Upload video"])
        model_path = st.text_input("YOLO11 model", value="yolo11n.pt")
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
        use_mock = st.checkbox("Use simulation detector", value=source_mode == "Simulation")
        audio_enabled = st.checkbox("Enable text-to-speech", value=False)
        max_frames = st.slider("Frames per run", 30, 900, 180, 30)
        video_path = None
        uploaded_path = None

        if source_mode == "Video file":
            video_path = st.text_input(
                "Video path",
                value=str(ROOT / "samples" / "left_person_right_vehicle.avi"),
            )
        elif source_mode == "Upload video":
            upload = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
            if upload is not None:
                suffix = Path(upload.name).suffix or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(upload.read())
                tmp.close()
                uploaded_path = tmp.name

        run = st.button("Run navigation", type="primary")

    camera_placeholder = st.empty()
    status_cols = st.columns(5)
    table_placeholder = st.empty()
    warning_placeholder = st.empty()

    col_sector, col_chart = st.columns([1, 1])
    sector_placeholder = col_sector.empty()
    chart_placeholder = col_chart.empty()
    warning_chart_placeholder = col_chart.empty()

    if not run:
        st.info("Choose an input source and press **Run navigation** to start.")
        return

    # Build source iterator
    try:
        source, frame_iterator = _build_frame_iterator(
            source_mode, video_path, uploaded_path, max_frames
        )
    except RuntimeError as exc:
        st.error(f"Could not open video source: {exc}")
        return

    try:
        cv2 = require_cv2()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    detector = create_detector(
        ModelConfig(model_path=model_path, confidence=confidence, use_mock=use_mock),
        fallback_to_mock=True,
    )
    nav_engine = NavigationEngine()
    haptics = HapticController()
    simulator = VirtualNavigationSimulator()
    audio = AudioFeedback(enabled=audio_enabled)
    logger = DetectionLogger(ROOT / "logs" / "navigation_events.csv")

    session_history: list[dict[str, object]] = []
    start = time.perf_counter()
    frame_count = 0
    total_detections = 0

    try:
        for frame_count, frame in frame_iterator:
            detections = detector.detect(frame)
            enriched = nav_engine.enrich(detections, frame.shape)
            signal = haptics.generate(enriched)
            logger.log(enriched)
            spoken = audio.speak_detections(enriched)
            readings = simulator.build_awareness(enriched, frame.shape[1])
            direction = simulator.suggested_direction(readings)

            annotated = annotate_frame(frame, enriched)
            camera_placeholder.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

            elapsed = max(1e-6, time.perf_counter() - start)
            fps = (frame_count + 1) / elapsed
            total_detections += len(enriched)
            nearest = min(
                (d.estimated_distance for d in enriched if d.estimated_distance is not None),
                default=None,
            )
            active_warnings = [d.message for d in enriched if d.message]

            status_cols[0].metric("FPS", f"{fps:.1f}")
            status_cols[1].metric("Detections", len(enriched))
            status_cols[2].metric("Nearest", f"{nearest:.2f} m" if nearest is not None else "clear")
            status_cols[3].metric("Haptic", signal.as_text())
            status_cols[4].metric("Direction", direction.capitalize())

            table_placeholder.dataframe(detection_table_rows(enriched), use_container_width=True)

            warning_text = " | ".join(spoken or active_warnings[:3]) or "No active warning"
            warning_placeholder.warning(f"{warning_text}\n\n{haptic_bar(signal)}")

            sector_placeholder.plotly_chart(
                build_sector_figure(readings), use_container_width=True, key=f"sector_{frame_count}"
            )

            # Accumulate session history for charting
            level_counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for d in enriched:
                if d.warning_level in level_counts:
                    level_counts[d.warning_level] += 1
            session_history.append(
                {
                    "frame": frame_count,
                    "count": len(enriched),
                    "nearest": nearest if nearest is not None else 0.0,
                    **{f"level_{k}": v for k, v in level_counts.items()},
                }
            )

            if len(session_history) > 1:
                chart_placeholder.plotly_chart(
                    build_session_chart(session_history),
                    use_container_width=True,
                    key=f"chart_{frame_count}",
                )
                warning_chart_placeholder.plotly_chart(
                    build_warning_level_chart(session_history),
                    use_container_width=True,
                    key=f"warnlevel_{frame_count}",
                )

            if source_mode == "Simulation":
                time.sleep(0.03)

    finally:
        if source is not None:
            source.release()

    # Session summary
    processed = frame_count + 1
    elapsed_total = max(1e-6, time.perf_counter() - start)
    avg_fps = processed / elapsed_total
    st.success(
        f"Processed **{processed}** frames at **{avg_fps:.1f} FPS**. "
        f"Total detections: **{total_detections}**. "
        f"CSV log saved to `logs/navigation_events.csv`."
    )


def _build_frame_iterator(
    source_mode: str,
    video_path: str | None,
    uploaded_path: str | None,
    max_frames: int,
) -> tuple[VideoSource | None, object]:
    if source_mode == "Webcam":
        source = VideoSource(0)
        return source, source.frames(max_frames=max_frames)
    if source_mode == "Video file":
        source = VideoSource(parse_source(video_path or "0"))
        return source, source.frames(max_frames=max_frames)
    if source_mode == "Upload video" and uploaded_path:
        source = VideoSource(uploaded_path)
        return source, source.frames(max_frames=max_frames)
    # Simulation
    return None, _simulation_frames(max_frames=max_frames)


def _simulation_frames(max_frames: int):
    for idx in range(max_frames):
        yield idx, make_simulation_frame(idx)


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------

def _render_evaluation_tab() -> None:
    st.header("Detection Evaluation")
    st.markdown(
        "Upload a ground-truth CSV and a predictions CSV to compute "
        "precision, recall, and F1 for this run."
    )

    col_gt, col_pred = st.columns(2)
    with col_gt:
        gt_file = st.file_uploader(
            "Ground-truth CSV",
            type=["csv"],
            key="gt_upload",
            help="Columns: frame_id, object, xmin, ymin, xmax, ymax",
        )
        st.caption("Or use the bundled sample:")
        use_sample_gt = st.checkbox("Use sample ground-truth", value=False)

    with col_pred:
        pred_file = st.file_uploader(
            "Predictions CSV",
            type=["csv"],
            key="pred_upload",
            help="Columns: frame_id, object, xmin, ymin, xmax, ymax, confidence",
        )
        st.caption("Or use the bundled sample:")
        use_sample_pred = st.checkbox("Use sample predictions", value=False)

    iou_threshold = st.slider("IoU threshold", 0.1, 0.9, 0.5, 0.05)
    evaluate_btn = st.button("Evaluate", type="primary")

    if not evaluate_btn:
        return

    try:
        from evaluation.metrics import evaluate_precision_recall, load_box_csv
    except Exception as exc:
        st.error(f"Evaluation module error: {exc}")
        return

    gt_path = ROOT / "samples" / "sample_ground_truth.csv" if use_sample_gt else None
    pred_path = ROOT / "samples" / "sample_predictions.csv" if use_sample_pred else None

    try:
        if gt_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(gt_file.read())
                gt_path = Path(tmp.name)
        if pred_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(pred_file.read())
                pred_path = Path(tmp.name)

        if gt_path is None or pred_path is None:
            st.warning("Please supply both a ground-truth and a predictions CSV (or tick 'Use sample').")
            return

        gt = load_box_csv(gt_path)
        preds = load_box_csv(pred_path)
        metrics = evaluate_precision_recall(gt, preds, iou_threshold=iou_threshold)

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return

    res_cols = st.columns(5)
    res_cols[0].metric("Precision", f"{metrics.precision:.3f}")
    res_cols[1].metric("Recall", f"{metrics.recall:.3f}")
    res_cols[2].metric("F1", f"{metrics.f1:.3f}")
    res_cols[3].metric("True Positives", metrics.true_positives)
    res_cols[4].metric("False Positives / FN", f"{metrics.false_positives} / {metrics.false_negatives}")

    st.markdown("---")
    st.subheader("Metric Bar Chart")
    _render_metrics_bar(metrics)

    st.subheader("Ground-truth boxes loaded")
    st.metric("Count", len(gt))
    st.subheader("Prediction boxes loaded")
    st.metric("Count", len(preds))


def _render_metrics_bar(metrics) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Precision", "Recall", "F1"],
                y=[metrics.precision, metrics.recall, metrics.f1],
                marker_color=["#4e9af1", "#f77f00", "#2a9d8f"],
                text=[f"{v:.3f}" for v in [metrics.precision, metrics.recall, metrics.f1]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 1.1]),
        plot_bgcolor="#1e2128",
        paper_bgcolor="#1e2128",
        font=dict(color="#cdd6f4"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------

def _render_about_tab() -> None:
    st.header("About this Prototype")
    st.markdown(
        """
This is an MSc dissertation prototype demonstrating **AI-powered assistive navigation** for
visually impaired users. The system uses YOLO11 real-time object detection and a spatial reasoning
pipeline to generate directional warnings, simulated haptic signals, and text-to-speech alerts.

### System Components

| Module | Purpose |
|---|---|
| `vision/` | YOLO11 detector, webcam/video input, frame annotation |
| `navigation/` | Zone assignment, monocular distance estimation, warning rules, 270° awareness |
| `feedback/` | Left/right haptic simulation, audio (pyttsx3), CSV event logging |
| `ui/` | Streamlit dashboard (this app) |
| `evaluation/` | Precision/recall IoU matching, latency/FPS benchmark |
| `scripts/` | CLI runner, synthetic sample video generator |

### Warning Levels

| Level | Distance threshold |
|---|---|
| **Critical** | ≤ 1.2 m (≤ 0.95 m in centre zone) |
| **High** | ≤ 2.0 m (≤ 1.75 m in centre zone) |
| **Medium** | ≤ 3.5 m |
| **Low** | > 3.5 m |

### Haptic Mapping

- **Left zone** → left vibration motor
- **Right zone** → right vibration motor
- **Centre zone** → both motors
- Closer obstacle → higher intensity (0–100 %)

### 270° Environmental Awareness

The polar sector chart extends the 90° forward camera field to a simulated 270° arc.
Central sectors are driven by live detections; peripheral sectors show virtual awareness
for dissertation UI design purposes.

### Evaluation

Use the **Evaluation** tab to compute precision, recall, and F1-score from bounding-box
CSV files. Sample files are in `samples/` and were generated by
`scripts/generate_sample_videos.py`.

### Dissertation Scope

Distance values are **relative monocular estimates** from bounding-box scale and vertical
position — not calibrated metric depth. This is intentional for a single RGB camera
without LiDAR or stereo. Values are useful for hazard ranking and comparative warnings.
        """
    )
    st.markdown("---")
    st.caption(
        "Prototype built for MSc research purposes. "
        "YOLO11 weights © Ultralytics. "
        "Not for clinical or safety-critical deployment."
    )


if __name__ == "__main__":
    main()
