"""Streamlit dashboard for the assistive navigation prototype.

Supports browser camera (WebRTC), video upload, video file, and simulation
modes.  Designed for remote hosting (Streamlit Cloud, VPS).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st


def _img(placeholder, frame, **kwargs):
    """Display an image full-width, compatible with all Streamlit versions."""
    placeholder.image(frame, use_column_width=True, **kwargs)


def _chart(container, fig, key=None):
    """Display a Plotly chart full-width, compatible with all Streamlit versions."""
    container.plotly_chart(fig, use_column_width=True)


def _df(container, data):
    """Display a dataframe, compatible with all Streamlit versions."""
    container.dataframe(data)


from feedback.browser_audio import BrowserAudioFeedback, inject_browser_tts
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
from vision.config import ModelConfig
from vision.detector import create_detector
from vision.synthetic import make_simulation_frame

# Optional WebRTC import
_WEBRTC_AVAILABLE = False
try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
    from vision.webrtc_adapter import NavigationVideoProcessor
    _WEBRTC_AVAILABLE = True
except ImportError:
    pass

_LEVEL_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

_SHORT_DIRECTION = {
    "path ahead is relatively clear": "Clear",
    "prefer slight left": "Go left",
    "prefer slight right": "Go right",
    "slow down": "Slow down",
}

# WebRTC ICE configuration — STUN servers for NAT traversal.
# For deployment behind symmetric NAT (Streamlit Cloud), set TURN_URL,
# TURN_USERNAME, and TURN_CREDENTIAL environment variables.
_ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
]
if os.getenv("TURN_URL"):
    _ICE_SERVERS.append({
        "urls": os.getenv("TURN_URL"),
        "username": os.getenv("TURN_USERNAME", ""),
        "credential": os.getenv("TURN_CREDENTIAL", ""),
    })


def main() -> None:
    st.set_page_config(
        page_title="Assistive Navigation",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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

def _build_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## Configuration")

        # Build input source options based on available packages
        source_options = []
        if _WEBRTC_AVAILABLE:
            source_options.append("Browser Camera")
        source_options.extend(["Simulation", "Video file", "Upload video"])

        source_mode = st.selectbox("Input source", source_options)
        st.markdown("---")
        model_path = st.text_input("YOLO11 model", value="yolo11n.pt")
        confidence = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
        use_mock = st.checkbox("Simulation detector", value=(source_mode == "Simulation"))
        audio_on = st.checkbox("Text-to-speech (browser)", value=False)
        max_frames = st.slider("Max frames", 30, 900, 180, 30)

        video_path = None
        uploaded_path = None

        if source_mode == "Browser Camera":
            st.caption(
                "📹 Uses your device camera via WebRTC. "
                "Grant camera permission when prompted."
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
        run=run,
    )


# ---------------------------------------------------------------------------
# Live Navigation tab
# ---------------------------------------------------------------------------

def _render_live_tab() -> None:
    st.markdown("# Assistive Navigation System")
    cfg = _build_sidebar()

    if cfg["source_mode"] == "Browser Camera":
        _render_live_webrtc(cfg)
    elif cfg["run"]:
        _render_live_synchronous(cfg)
    else:
        _show_idle_screen()


# ---------------------------------------------------------------------------
# WebRTC browser camera mode (LIVE STREAMING)
# ---------------------------------------------------------------------------

def _render_live_webrtc(cfg: dict) -> None:
    """Live navigation using browser camera via WebRTC."""

    if not _WEBRTC_AVAILABLE:
        st.error(
            "⚠️ `streamlit-webrtc` is not installed. "
            "Add it to requirements.txt: `streamlit-webrtc>=0.47.0`"
        )
        return

    st.info("👁️ Click **START** to begin real-time obstacle detection from your camera.")

    # Create the WebRTC streamer
    rtc_config = RTCConfiguration(iceServers=_ICE_SERVERS)

    def _processor_factory():
        processor = NavigationVideoProcessor()
        processor.configure(ModelConfig(
            model_path=cfg["model_path"],
            confidence=cfg["confidence"],
            use_mock=cfg["use_mock"],
        ))
        return processor

    ctx = webrtc_streamer(
        key="assistive-nav",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=_processor_factory,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    # Dashboard below the video stream
    if ctx.state.playing and ctx.video_processor:
        _render_webrtc_dashboard(ctx, cfg)
    elif not ctx.state.playing:
        st.markdown("---")
        st.markdown(
            "**How it works:** Your browser camera streams video frames to the server. "
            "YOLO11 detects obstacles in real time, and the annotated feed is returned "
            "back to you with zone markers, distance estimates, and warnings."
        )


def _render_webrtc_dashboard(ctx, cfg: dict) -> None:
    """Display metrics and charts by polling the WebRTC processor's results."""

    col_info, col_sector = st.columns([5, 6])

    with col_info:
        mc = st.columns(2)
        fps_ph = mc[0].empty()
        near_ph = mc[1].empty()
        det_ph = mc[0].empty()
        dir_ph = mc[1].empty()

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

    # Audio feedback handler
    audio = BrowserAudioFeedback(enabled=cfg["audio_on"])

    # Initialise session history
    if "webrtc_history" not in st.session_state:
        st.session_state.webrtc_history = []

    # Read latest results from the processor
    results = ctx.video_processor.get_results()

    if results["frame_count"] > 0:
        enriched = results["detections"]
        signal = results["signal"]
        readings = results["readings"]
        direction = results["direction"]
        fps = results["fps"]

        # Derived values
        nearest = min(
            (d.estimated_distance for d in enriched if d.estimated_distance is not None),
            default=None,
        )
        active_msgs = [d.message for d in enriched if d.message]
        top_level = max(
            (d.warning_level for d in enriched),
            key=lambda lv: _LEVEL_ORDER.get(lv, 0),
            default="none",
        )

        # 2x2 metrics
        fps_ph.metric("FPS", f"{fps:.0f}")
        near_ph.metric("Nearest", f"{nearest:.1f} m" if nearest is not None else "clear")
        det_ph.metric("Detections", len(enriched))
        dir_ph.metric("Direction", _SHORT_DIRECTION.get(direction, direction.capitalize()))

        # Warning banner
        msg = " | ".join(active_msgs[:2])
        if msg and top_level == "critical":
            warn_ph.error(f"🛑 STOP — {msg}")
        elif msg and top_level == "high":
            warn_ph.warning(f"⚠️ Warning — {msg}")
        elif msg and top_level == "medium":
            warn_ph.info(f"ℹ️ Caution — {msg}")
        else:
            warn_ph.success("✅ Path clear")

        # Haptic progress bars
        if signal is not None:
            l_pct = int(signal.left_intensity * 100)
            r_pct = int(signal.right_intensity * 100)
            hap_l_ph.progress(signal.left_intensity, text=f"Left motor — {l_pct}%")
            hap_r_ph.progress(signal.right_intensity, text=f"Right motor — {r_pct}%")

        # 270° sector chart
        if readings:
            _chart(sector_ph, build_sector_figure(readings))

        # Session history
        level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for d in enriched:
            if d.warning_level in level_counts:
                level_counts[d.warning_level] += 1
        st.session_state.webrtc_history.append({
            "frame": results["frame_count"],
            "count": len(enriched),
            "nearest": nearest if nearest is not None else 0.0,
            **{f"level_{k}": v for k, v in level_counts.items()},
        })
        if len(st.session_state.webrtc_history) >= 5:
            _chart(chart_ph, build_session_chart(st.session_state.webrtc_history))

        # Detection table
        _df(table_ph, detection_table_rows(enriched))

        # Browser TTS
        spoken = audio.speak_detections(enriched)
        pending = audio.drain_pending()
        inject_browser_tts(pending)

    # Auto-refresh to keep metrics updating while camera is streaming
    time.sleep(0.3)
    st.rerun()


# ---------------------------------------------------------------------------
# Synchronous mode (Simulation / Video file / Upload)
# ---------------------------------------------------------------------------

def _render_live_synchronous(cfg: dict) -> None:
    """Synchronous frame-loop for simulation, file, and upload modes."""

    # Build layout placeholders
    camera_ph = st.empty()

    col_info, col_sector = st.columns([5, 6])

    with col_info:
        mc = st.columns(2)
        fps_ph = mc[0].empty()
        near_ph = mc[1].empty()
        det_ph = mc[0].empty()
        dir_ph = mc[1].empty()

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

    # Initialise pipeline
    try:
        source, frame_iter = _build_frame_iterator(
            cfg["source_mode"], cfg["video_path"], cfg["uploaded_path"],
            cfg["max_frames"],
        )
    except RuntimeError as exc:
        st.error(f"Could not open video source: {exc}")
        return

    detector = create_detector(
        ModelConfig(model_path=cfg["model_path"], confidence=cfg["confidence"], use_mock=cfg["use_mock"]),
        fallback_to_mock=True,
    )
    nav_engine = NavigationEngine()
    haptics = HapticController()
    simulator = VirtualNavigationSimulator()
    audio = BrowserAudioFeedback(enabled=cfg["audio_on"])
    logger = DetectionLogger(ROOT / "logs" / "navigation_events.csv")

    history: list[dict] = []
    start = time.perf_counter()
    frame_count = 0
    total_dets = 0

    # Main loop
    try:
        for frame_count, frame in frame_iter:
            detections = detector.detect(frame)
            enriched = nav_engine.enrich(detections, frame.shape)
            signal = haptics.generate(enriched)
            logger.log(enriched)
            readings = simulator.build_awareness(enriched, frame.shape[1])
            direction = simulator.suggested_direction(readings)

            # Camera feed
            annotated = annotate_frame(frame, enriched)
            _img(camera_ph, bgr_to_rgb(annotated), channels="RGB")

            # Derived values
            elapsed = max(1e-6, time.perf_counter() - start)
            fps = (frame_count + 1) / elapsed
            total_dets += len(enriched)
            nearest = min(
                (d.estimated_distance for d in enriched if d.estimated_distance is not None),
                default=None,
            )
            active_msgs = [d.message for d in enriched if d.message]
            top_level = max(
                (d.warning_level for d in enriched),
                key=lambda lv: _LEVEL_ORDER.get(lv, 0),
                default="none",
            )

            # 2x2 metrics
            fps_ph.metric("FPS", f"{fps:.0f}")
            near_ph.metric("Nearest", f"{nearest:.1f} m" if nearest is not None else "clear")
            det_ph.metric("Detections", len(enriched))
            dir_ph.metric("Direction", _SHORT_DIRECTION.get(direction, direction.capitalize()))

            # Warning banner
            msg = " | ".join(active_msgs[:2])
            if msg and top_level == "critical":
                warn_ph.error(f"🛑 STOP — {msg}")
            elif msg and top_level == "high":
                warn_ph.warning(f"⚠️ Warning — {msg}")
            elif msg and top_level == "medium":
                warn_ph.info(f"ℹ️ Caution — {msg}")
            else:
                warn_ph.success("✅ Path clear")

            # Haptic progress bars
            l_pct = int(signal.left_intensity * 100)
            r_pct = int(signal.right_intensity * 100)
            hap_l_ph.progress(signal.left_intensity, text=f"Left motor — {l_pct}%")
            hap_r_ph.progress(signal.right_intensity, text=f"Right motor — {r_pct}%")

            # 270° sector chart
            _chart(sector_ph, build_sector_figure(readings), key=f"sector_{frame_count}")

            # Session trend chart
            level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for d in enriched:
                if d.warning_level in level_counts:
                    level_counts[d.warning_level] += 1
            history.append({
                "frame": frame_count,
                "count": len(enriched),
                "nearest": nearest if nearest is not None else 0.0,
                **{f"level_{k}": v for k, v in level_counts.items()},
            })
            if len(history) >= 5:
                _chart(chart_ph, build_session_chart(history), key=f"chart_{frame_count}")

            # Detection table
            _df(table_ph, detection_table_rows(enriched))

            # Browser TTS
            spoken = audio.speak_detections(enriched)
            pending = audio.drain_pending()
            inject_browser_tts(pending)

            if cfg["source_mode"] == "Simulation":
                time.sleep(0.03)

    finally:
        if source is not None:
            source.release()

    # Session summary
    processed = frame_count + 1
    avg_fps = processed / max(1e-6, time.perf_counter() - start)
    st.success(
        f"Done — **{processed}** frames at **{avg_fps:.1f} FPS** · "
        f"**{total_dets}** total detections · log saved to `logs/navigation_events.csv`"
    )


def _show_idle_screen() -> None:
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input sources", "4", "Camera / Video / Upload / Simulation")
    c2.metric("AI model", "YOLO11n", "Real-time detection")
    c3.metric("Object classes", "7", "Person, chair, stairs…")
    c4.metric("Feedback modes", "3", "Visual · Haptic · Audio")
    st.info("Select an input source in the sidebar, then press **Run navigation** or click **START** for camera.")


# ---------------------------------------------------------------------------
# Frame iterator helpers
# ---------------------------------------------------------------------------

def _build_frame_iterator(source_mode, video_path, uploaded_path, max_frames):
    """Build a frame iterator for non-WebRTC modes."""
    if source_mode in ("Video file", "Upload video"):
        try:
            from vision.camera import VideoSource, parse_source
        except RuntimeError:
            st.error("OpenCV is not available. Use Browser Camera or Simulation mode.")
            return None, _sim_frames(max_frames)

        if source_mode == "Video file":
            src = VideoSource(parse_source(video_path or "0"))
            return src, src.frames(max_frames=max_frames)
        if source_mode == "Upload video" and uploaded_path:
            src = VideoSource(uploaded_path)
            return src, src.frames(max_frames=max_frames)

    # Default: simulation
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

    gt_path = ROOT / "samples" / "sample_ground_truth.csv" if use_sample_gt else None
    pred_path = ROOT / "samples" / "sample_predictions.csv" if use_sample_pred else None

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
        gt = load_box_csv(gt_path)
        preds = load_box_csv(pred_path)
        metrics = evaluate_precision_recall(gt, preds, iou_threshold=iou_threshold)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return

    st.markdown("---")
    rc = st.columns(5)
    rc[0].metric("Precision", f"{metrics.precision:.3f}")
    rc[1].metric("Recall", f"{metrics.recall:.3f}")
    rc[2].metric("F1", f"{metrics.f1:.3f}")
    rc[3].metric("True Positives", metrics.true_positives)
    rc[4].metric("FP / FN", f"{metrics.false_positives} / {metrics.false_negatives}")

    _chart(st, _metrics_bar(metrics))

    with st.expander("Box counts"):
        st.metric("Ground-truth boxes", len(gt))
        st.metric("Prediction boxes", len(preds))


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
| `vision/` | YOLO11 detector, browser camera (WebRTC), frame annotation |
| `navigation/` | Zone assignment (L/C/R), distance estimation, warning rules, 270° awareness |
| `feedback/` | Haptic simulation, browser TTS audio, CSV logging |
| `ui/` | This Streamlit dashboard |
| `evaluation/` | IoU precision/recall, latency/FPS benchmark |
| `scripts/` | CLI runner, synthetic video generator |

---

### Input modes

| Mode | Description |
|---|---|
| Browser Camera | Real-time WebRTC stream from client device camera |
| Simulation | Synthetic frames with mock detections |
| Video file | Process a local video file |
| Upload video | Upload and process a video |

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

### Deployment

This app is hosted on **Streamlit Cloud** with browser-based camera access via WebRTC.
For TURN server configuration (required behind strict NAT), set these secrets:
- `TURN_URL` — e.g. `turn:your-server.com:3478`
- `TURN_USERNAME`
- `TURN_CREDENTIAL`

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
