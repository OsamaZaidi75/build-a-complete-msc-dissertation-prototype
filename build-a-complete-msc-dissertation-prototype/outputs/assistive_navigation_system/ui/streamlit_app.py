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

# use_column_width=True works on every Streamlit version ever released.
# use_container_width was added later and is not present in all builds.
# We always use use_column_width so the app runs on any version >= 1.0.

def _img(placeholder, frame, **kwargs):
    """Display an image full-width, compatible with all Streamlit versions."""
    placeholder.image(frame, use_column_width=True, **kwargs)


def _chart(container, fig, key=None):
    """Display a Plotly chart full-width, compatible with all Streamlit versions."""
    container.plotly_chart(fig, use_column_width=True)


def _df(container, data):
    """Display a dataframe, compatible with all Streamlit versions."""
    container.dataframe(data)


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
)
from vision.annotator import annotate_frame, bgr_to_rgb
from vision.camera import VideoSource, parse_source, require_cv2
from vision.config import ModelConfig
from vision.detector import create_detector
from vision.synthetic import make_simulation_frame

_LEVEL_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

_SHORT_DIRECTION = {
    "path ahead is relatively clear": "Clear",
    "prefer slight left": "Go left",
    "prefer slight right": "Go right",
    "slow down": "Slow down",
}


def main() -> None:
    st.set_page_config(
        page_title="Assistive Navigation",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Minimal top-bar custom CSS — tighten padding, clean font
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
        h1 { font-size: 1.4rem !important; margin-bottom: 0 !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_live, tab_eval, tab_about = st.tabs(["Live Navigation", "Evaluation", "About"])

    with tab_live:
        _render_live_tab()
    with tab_eval:
        _render_evaluation_tab()
    with tab_about:
        _render_about_tab()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _scan_cameras() -> list[tuple[int, str]]:
    """Try indices 0–9 and return (index, label) for every camera that opens.
    Falls back to [(0,'Camera 0')] if cv2 is not yet installed.
    """
    try:
        import cv2
    except ImportError:
        return [(0, "Camera 0 (default)")]

    found = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if cap is not None and cap.isOpened():
            found.append((idx, f"Camera {idx}"))
            cap.release()
    return found if found else [(0, "Camera 0 (default)")]


def _build_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## Configuration")
        source_mode = st.selectbox(
            "Input source",
            ["Simulation", "Webcam", "Video file", "Upload video"],
        )
        st.markdown("---")
        model_path  = st.text_input("YOLO11 model", value="yolo11n.pt")
        confidence  = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
        use_mock    = st.checkbox("Simulation detector", value=(source_mode == "Simulation"))
        audio_on    = st.checkbox("Text-to-speech", value=False)
        max_frames  = st.slider("Max frames", 30, 900, 180, 30)

        video_path    = None
        uploaded_path = None
        webcam_index  = 0

        if source_mode == "Webcam":
            cameras = _scan_cameras()
            if len(cameras) == 1:
                webcam_index = cameras[0][0]
                st.info(f"Found: {cameras[0][1]}")
            else:
                labels = [label for _, label in cameras]
                choice = st.selectbox("Select camera", labels)
                webcam_index = cameras[labels.index(choice)][0]
            st.caption(
                "DroidCam: make sure the DroidCam desktop app is running "
                "and connected **before** pressing Run."
            )
        elif source_mode == "Video file":
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

        st.markdown("---")
        run = st.button("Run navigation", type="primary")

    return dict(
        source_mode=source_mode,
        model_path=model_path,
        confidence=confidence,
        use_mock=use_mock,
        audio_on=audio_on,
        max_frames=max_frames,
        video_path=video_path,
        uploaded_path=uploaded_path,
        webcam_index=webcam_index,
        run=run,
    )


# ---------------------------------------------------------------------------
# Live Navigation tab
# ---------------------------------------------------------------------------

def _render_live_tab() -> None:
    st.markdown("# Assistive Navigation System")
    cfg = _build_sidebar()

    if not cfg["run"]:
        _show_idle_screen()
        return

    # ── Build layout placeholders ────────────────────────────────────────
    camera_ph = st.empty()

    col_info, col_sector = st.columns([5, 6])

    with col_info:
        mc = st.columns(2)
        fps_ph  = mc[0].empty()
        near_ph = mc[1].empty()
        det_ph  = mc[0].empty()
        dir_ph  = mc[1].empty()

        st.markdown("---")
        warn_ph = st.empty()
        st.markdown("---")
        st.caption("Haptic feedback")
        hap_l_ph = st.empty()
        hap_r_ph = st.empty()

    with col_sector:
        sector_ph = st.empty()

    chart_ph = st.empty()

    with st.expander("Detection log", expanded=False):
        table_ph = st.empty()

    # ── Initialise pipeline ──────────────────────────────────────────────
    try:
        source, frame_iter = _build_frame_iterator(
            cfg["source_mode"], cfg["video_path"], cfg["uploaded_path"],
            cfg["max_frames"], cfg["webcam_index"],
        )
    except RuntimeError as exc:
        st.error(f"Could not open video source: {exc}")
        return

    try:
        require_cv2()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    detector   = create_detector(
        ModelConfig(model_path=cfg["model_path"], confidence=cfg["confidence"], use_mock=cfg["use_mock"]),
        fallback_to_mock=True,
    )
    nav_engine = NavigationEngine()
    haptics    = HapticController()
    simulator  = VirtualNavigationSimulator()
    audio      = AudioFeedback(enabled=cfg["audio_on"])
    logger     = DetectionLogger(ROOT / "logs" / "navigation_events.csv")

    history: list[dict] = []
    start        = time.perf_counter()
    frame_count  = 0
    total_dets   = 0

    # ── Main loop ────────────────────────────────────────────────────────
    try:
        for frame_count, frame in frame_iter:
            detections = detector.detect(frame)
            enriched   = nav_engine.enrich(detections, frame.shape)
            signal     = haptics.generate(enriched)
            logger.log(enriched)
            spoken     = audio.speak_detections(enriched)
            readings   = simulator.build_awareness(enriched, frame.shape[1])
            direction  = simulator.suggested_direction(readings)

            # Camera feed
            annotated = annotate_frame(frame, enriched)
            _img(camera_ph, bgr_to_rgb(annotated), channels="RGB")

            # Derived values
            elapsed      = max(1e-6, time.perf_counter() - start)
            fps          = (frame_count + 1) / elapsed
            total_dets  += len(enriched)
            nearest      = min(
                (d.estimated_distance for d in enriched if d.estimated_distance is not None),
                default=None,
            )
            active_msgs  = [d.message for d in enriched if d.message]
            top_level    = max(
                (d.warning_level for d in enriched),
                key=lambda lv: _LEVEL_ORDER.get(lv, 0),
                default="none",
            )

            # 2x2 metrics
            fps_ph.metric("FPS",        f"{fps:.0f}")
            near_ph.metric("Nearest",   f"{nearest:.1f} m" if nearest is not None else "clear")
            det_ph.metric("Detections", len(enriched))
            dir_ph.metric("Direction",  _SHORT_DIRECTION.get(direction, direction.capitalize()))

            # Warning banner — colour-coded, only shows when relevant
            msg = " | ".join(spoken or active_msgs[:2])
            if msg and top_level == "critical":
                warn_ph.error(f"STOP — {msg}")
            elif msg and top_level == "high":
                warn_ph.warning(f"Warning — {msg}")
            elif msg and top_level == "medium":
                warn_ph.info(f"Caution — {msg}")
            else:
                warn_ph.success("Path clear")

            # Haptic progress bars
            l_pct = int(signal.left_intensity * 100)
            r_pct = int(signal.right_intensity * 100)
            hap_l_ph.progress(signal.left_intensity, text=f"Left motor — {l_pct}%")
            hap_r_ph.progress(signal.right_intensity, text=f"Right motor — {r_pct}%")

            # 270° sector chart
            _chart(sector_ph, build_sector_figure(readings), key=f"sector_{frame_count}")

            # Session trend chart (appears after enough data)
            level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for d in enriched:
                if d.warning_level in level_counts:
                    level_counts[d.warning_level] += 1
            history.append({
                "frame":   frame_count,
                "count":   len(enriched),
                "nearest": nearest if nearest is not None else 0.0,
                **{f"level_{k}": v for k, v in level_counts.items()},
            })
            if len(history) >= 5:
                _chart(chart_ph, build_session_chart(history), key=f"chart_{frame_count}")

            # Hidden detection table
            _df(table_ph, detection_table_rows(enriched))

            if cfg["source_mode"] == "Simulation":
                time.sleep(0.03)

    finally:
        if source is not None:
            source.release()

    # Session summary
    processed = frame_count + 1
    avg_fps   = processed / max(1e-6, time.perf_counter() - start)
    st.success(
        f"Done — **{processed}** frames at **{avg_fps:.1f} FPS** · "
        f"**{total_dets}** total detections · log saved to `logs/navigation_events.csv`"
    )


def _show_idle_screen() -> None:
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input sources", "4",  "Webcam / Video / Upload / Simulation")
    c2.metric("AI model",      "YOLO11n", "Real-time detection")
    c3.metric("Object classes", "7", "Person, chair, stairs…")
    c4.metric("Feedback modes", "3", "Visual · Haptic · Audio")
    st.info("Select an input source in the sidebar, then press **Run navigation**.")


# ---------------------------------------------------------------------------
# Frame iterator helpers
# ---------------------------------------------------------------------------

def _build_frame_iterator(source_mode, video_path, uploaded_path, max_frames, webcam_index=0):
    if source_mode == "Webcam":
        src = VideoSource(webcam_index)
        return src, src.frames(max_frames=max_frames)
    if source_mode == "Video file":
        src = VideoSource(parse_source(video_path or "0"))
        return src, src.frames(max_frames=max_frames)
    if source_mode == "Upload video" and uploaded_path:
        src = VideoSource(uploaded_path)
        return src, src.frames(max_frames=max_frames)
    return None, _sim_frames(max_frames)


def _sim_frames(max_frames: int):
    for idx in range(max_frames):
        yield idx, make_simulation_frame(idx)


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------

def _render_evaluation_tab() -> None:
    st.markdown("# Detection Evaluation")
    st.caption("Measure precision, recall, and F1 against ground-truth bounding-box CSVs.")

    col_gt, col_pred = st.columns(2)
    with col_gt:
        gt_file = st.file_uploader(
            "Ground-truth CSV",
            type=["csv"],
            key="gt_upload",
            help="Columns: frame_id, object, xmin, ymin, xmax, ymax",
        )
        use_sample_gt = st.checkbox("Use bundled sample", key="use_sample_gt")
    with col_pred:
        pred_file = st.file_uploader(
            "Predictions CSV",
            type=["csv"],
            key="pred_upload",
            help="Columns: frame_id, object, xmin, ymin, xmax, ymax, confidence",
        )
        use_sample_pred = st.checkbox("Use bundled sample", key="use_sample_pred")

    iou_threshold = st.slider("IoU threshold", 0.1, 0.9, 0.5, 0.05)

    if not st.button("Evaluate", type="primary"):
        return

    try:
        from evaluation.metrics import evaluate_precision_recall, load_box_csv
    except Exception as exc:
        st.error(f"Evaluation module error: {exc}")
        return

    gt_path   = ROOT / "samples" / "sample_ground_truth.csv" if use_sample_gt else None
    pred_path = ROOT / "samples" / "sample_predictions.csv"  if use_sample_pred else None

    try:
        if gt_file is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(gt_file.read()); tmp.close()
            gt_path = Path(tmp.name)
        if pred_file is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(pred_file.read()); tmp.close()
            pred_path = Path(tmp.name)
        if gt_path is None or pred_path is None:
            st.warning("Supply both CSVs or tick 'Use bundled sample'.")
            return
        gt      = load_box_csv(gt_path)
        preds   = load_box_csv(pred_path)
        metrics = evaluate_precision_recall(gt, preds, iou_threshold=iou_threshold)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return

    st.markdown("---")
    rc = st.columns(5)
    rc[0].metric("Precision",        f"{metrics.precision:.3f}")
    rc[1].metric("Recall",           f"{metrics.recall:.3f}")
    rc[2].metric("F1",               f"{metrics.f1:.3f}")
    rc[3].metric("True Positives",   metrics.true_positives)
    rc[4].metric("FP / FN",          f"{metrics.false_positives} / {metrics.false_negatives}")

    _chart(st, _metrics_bar(metrics))

    with st.expander("Box counts"):
        st.metric("Ground-truth boxes", len(gt))
        st.metric("Prediction boxes",   len(preds))


def _metrics_bar(metrics):
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=["Precision", "Recall", "F1"],
        y=[metrics.precision, metrics.recall, metrics.f1],
        marker_color=["#4e9af1", "#f77f00", "#2a9d8f"],
        text=[f"{v:.3f}" for v in [metrics.precision, metrics.recall, metrics.f1]],
        textposition="outside",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 1.15], showgrid=False),
        xaxis=dict(showgrid=False),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=14),
    )
    return fig


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------

def _render_about_tab() -> None:
    st.markdown("# About this Prototype")
    st.markdown("""
An MSc dissertation prototype for **AI-powered assistive navigation** for visually impaired users.
YOLO11 detects obstacles in real time; a spatial pipeline converts detections into
directional warnings, haptic signals, and spoken alerts.

---

### System modules

| Module | Purpose |
|---|---|
| `vision/` | YOLO11 detector, webcam/video input, frame annotation |
| `navigation/` | Zone assignment (L/C/R), distance estimation, warning rules, 270° awareness |
| `feedback/` | Haptic simulation, pyttsx3 audio, CSV logging |
| `ui/` | This Streamlit dashboard |
| `evaluation/` | IoU precision/recall, latency/FPS benchmark |
| `scripts/` | CLI runner, synthetic video generator |

---

### Warning levels

| Level | Distance | Colour | Audio |
|---|---|---|---|
| Critical | ≤ 1.2 m | Red | Yes |
| High | ≤ 2.0 m | Orange | Yes |
| Medium | ≤ 3.5 m | Yellow | Yes |
| Low | > 3.5 m | Green | No |

Centre-zone obstacles escalate one level earlier (−0.25 m bias).

---

### Haptic mapping

- Left zone obstacle → left motor
- Right zone obstacle → right motor
- Centre zone obstacle → both motors
- Intensity = `1 − distance / 5 m` clamped to 10–100 %

---

### Dissertation scope

Distance values are **relative monocular estimates** from bounding-box scale and vertical
position — not calibrated metric depth. No LiDAR or stereo camera is used.
Values are appropriate for hazard ranking in a dissertation demonstration context.
    """)
    st.caption(
        "YOLO11 weights by Ultralytics. "
        "Not for clinical or safety-critical deployment."
    )


if __name__ == "__main__":
    main()
