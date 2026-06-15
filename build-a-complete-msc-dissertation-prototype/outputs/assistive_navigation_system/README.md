# AI-Powered Assistive Navigation System for Visually Impaired Users

An MSc dissertation prototype for real-time obstacle awareness using YOLO11 object detection, monocular distance estimation, spatial zone reasoning, a virtual 270-degree awareness display, simulated haptic feedback, text-to-speech alerts, CSV logging, and an in-app evaluation suite — all in a single Streamlit dashboard.

---

## Quick Start — One Command, Every Device

**All you need is Python 3.10+.** The launcher handles everything else automatically
(virtual environment, dependencies, sample videos).

### Windows — double-click or run in terminal
```powershell
cd outputs\assistive_navigation_system
python start.py
```
Or just **double-click `start.bat`** in File Explorer.

### macOS / Linux
```bash
cd outputs/assistive_navigation_system
python3 start.py
```
Or run `bash start.sh`.

### What the launcher does (automatically, in order)
1. Checks Python version (3.10+ required)
2. Creates a `.venv` virtual environment if not already present
3. Installs all dependencies from `requirements.txt` (first run only, ~2 min)
4. Generates the synthetic sample videos (first run only, instant)
5. Opens the Streamlit dashboard at **http://localhost:8501**

> **No internet needed after first run.** YOLO11 weights (~6 MB) download once automatically.
> Enable **"Use simulation detector"** in the sidebar to run 100% offline from the start.

---

## Key Features

| Feature | Details |
|---|---|
| YOLO11 detection | Ultralytics `yolo11n.pt` by default; custom checkpoints supported |
| Input sources | Simulation, webcam, local video file, browser-uploaded video |
| Target classes | person, chair, table, wall, door, vehicle, stairs |
| Zone reasoning | Camera split into **left / centre / right** — drives haptic direction |
| Distance estimation | Monocular relative estimate from bounding-box scale + vertical position |
| Warning levels | critical (≤ 1.2 m) · high (≤ 2.0 m) · medium (≤ 3.5 m) · low |
| Haptic simulation | Left obstacle → left motor · right → right · centre → both · closer → stronger |
| Audio | Text-to-speech via `pyttsx3`, with per-message cooldown throttle |
| 270° awareness | Polar sector chart projecting forward detections across a simulated arc |
| Live session charts | Per-frame detection count, nearest distance, stacked warning-level history |
| Avoidance direction | Automatic path suggestion (prefer left / prefer right / slow down / clear) |
| Evaluation tab | In-app precision / recall / F1 with adjustable IoU threshold and bar chart |
| CSV logging | `logs/navigation_events.csv` — timestamp, object, zone, distance, warning level |
| Benchmark CLI | Latency (mean + p95) and FPS measurement script |
| Docker support | One-command containerised deployment |

---

## Dashboard Tabs

### Live Navigation
The main operating view. Configure source and model in the **sidebar**, then press **Run navigation**.

Real-time output:
- Annotated camera feed with bounding boxes, zone overlays, distance labels, and warning colours
- Five status metrics: **FPS · Detections · Nearest obstacle · Haptic state · Avoidance direction**
- Detection table with object, zone, distance, confidence, warning level, and message
- Warning banner with audio-style message and haptic bar visualization
- 270-degree polar sector chart
- Live session charts: detection count + nearest distance per frame, and stacked warning-level history

Session summary is printed on completion with total frames, average FPS, and total detections.

### Evaluation
Upload your own ground-truth and prediction CSVs, or tick **"Use sample"** to load the bundled files from `samples/`. Adjust the IoU threshold with the slider and press **Evaluate** to see:

- Precision · Recall · F1 · True Positives · False Positives / False Negatives
- Metric bar chart

### About
System reference: component table, warning level thresholds, haptic mapping rules, 270° field explanation, and dissertation scope notes.

---

## CLI Usage

### Run on a video file (display window)
```powershell
python scripts\run_pipeline.py --source samples\left_person_right_vehicle.avi --mock --display
```

### Run on webcam with audio
```powershell
python scripts\run_pipeline.py --source 0 --audio
```

### Run with a custom YOLO11 model
```powershell
python scripts\run_pipeline.py --source samples\centre_stairs_door.avi --model models\best.pt
```

### Precision / Recall evaluation
```powershell
python evaluation\evaluate_detections.py `
  --ground-truth samples\sample_ground_truth.csv `
  --predictions  samples\sample_predictions.csv `
  --iou 0.5
```

### Latency and FPS benchmark
```powershell
python evaluation\benchmark.py --source samples\centre_stairs_door.avi --mock --frames 120
```

---

## Project Structure

```text
assistive_navigation_system/
├── ui/
│   ├── streamlit_app.py       Streamlit dashboard (3 tabs: Live / Evaluation / About)
│   └── components.py          Plotly figures, detection table, haptic bar, session charts
├── vision/
│   ├── detector.py            YOLO11 detector + deterministic mock detector
│   ├── detections.py          BBox and Detection dataclasses
│   ├── annotator.py           OpenCV frame annotation (boxes, zones, labels)
│   ├── camera.py              VideoSource iterator (webcam / file)
│   ├── config.py              ModelConfig, class aliases, known widths
│   └── synthetic.py           Synthetic simulation frame generator
├── navigation/
│   ├── engine.py              NavigationEngine: enrich detections with zone + distance + level
│   ├── zones.py               Left / centre / right zone assigner
│   ├── distance.py            Monocular distance estimator
│   ├── risk.py                Warning level rules and message generator
│   └── virtual_270.py         270-degree sector awareness simulator
├── feedback/
│   ├── haptics.py             HapticController: left/right intensity signals
│   ├── audio.py               AudioFeedback: pyttsx3 TTS with cooldown
│   └── logger.py              DetectionLogger: CSV event log
├── evaluation/
│   ├── metrics.py             IoU matching, precision / recall / F1 computation
│   ├── evaluate_detections.py CLI precision/recall script
│   └── benchmark.py           CLI latency/FPS benchmark
├── scripts/
│   ├── run_pipeline.py        CLI runner (webcam / video, optional display + audio)
│   └── generate_sample_videos.py Generates AVI samples and scenario CSVs (stdlib only)
├── samples/
│   ├── left_person_right_vehicle.avi
│   ├── centre_stairs_door.avi
│   ├── simulation_scenarios.csv
│   ├── sample_ground_truth.csv
│   └── sample_predictions.csv
├── logs/
│   └── navigation_events.csv  Runtime detection log (auto-created)
├── docs/
│   ├── INSTALLATION.md        Full installation guide
│   ├── USER_MANUAL.md         Dashboard and CLI usage manual
│   ├── dissertation_notes.md  Research aims, evaluation design, limitations
│   ├── architecture.md        Architecture overview
│   └── uml.md                 UML diagrams
├── tests/
│   └── test_navigation_feedback_metrics.py  Unit tests (pytest)
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## Running Tests

```powershell
# With the venv active:
python -m pytest tests/ -v

# Without venv (system Python):
python -m pytest tests/ -v --rootdir .
```

All 4 tests cover: zone assignment, distance ordering, haptic centre-obstacle logic, and precision/recall metrics.

---

## Docker

```powershell
docker build -t assistive-nav-yolo11 .
docker run --rm -p 8501:8501 assistive-nav-yolo11
```

Open `http://localhost:8501`. For webcam on Linux add `--device=/dev/video0`.

---

## Warning Level Reference

| Level | Centre zone | Other zones | Behaviour |
|---|---|---|---|
| **Critical** | ≤ 0.95 m | ≤ 1.2 m | Red box · urgent audio · strong haptic |
| **High** | ≤ 1.75 m | ≤ 2.0 m | Orange box · audio · haptic |
| **Medium** | ≤ 3.5 m | ≤ 3.5 m | Yellow box · audio · haptic |
| **Low** | > 3.5 m | > 3.5 m | Green box · no audio · no haptic |

Centre-zone obstacles get a 0.25 m bias applied — they trigger one level higher than the same distance in a side zone.

---

## CSV Log Schema

`logs/navigation_events.csv`

```
timestamp,object,zone,estimated_distance,warning_level
2025-06-15T10:23:01.123456+00:00,person,left,1.84,high
2025-06-15T10:23:01.123456+00:00,vehicle,right,2.41,medium
```

---

## Evaluation CSV Schema

**Ground-truth and predictions CSVs** (used by the Evaluation tab and CLI):

```
frame_id,object,xmin,ymin,xmax,ymax,confidence
left_person_right_vehicle.avi:0,person,35,64,69,166,1.0
```

`confidence` is optional in ground-truth files (defaults to 1.0).

---

## Dissertation Scope

Distance values are **relative monocular estimates** from bounding-box apparent size and vertical position — not calibrated metric depth. No LiDAR, stereo camera, or depth sensor is used. Values are appropriate for hazard ranking and comparative warnings in a dissertation demonstration context.

YOLO11 COCO pretrained weights detect `person`, `chair`, `dining table`, and common vehicles natively. `door`, `wall`, and `stairs` are supported in the system vocabulary and require custom YOLO11 weights trained on those classes, or the built-in simulation/mock detector.

---

## Model Note

Official Ultralytics YOLO11 reference: [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)
