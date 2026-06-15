# User Manual

## Purpose

This prototype demonstrates a software-only assistive navigation system for visually impaired users. It uses YOLO11 object detection to identify obstacles in a camera feed, estimates their relative distance, assigns each to a left / centre / right spatial zone, and converts the hazard state into:

- **Visual** — annotated camera feed with coloured bounding boxes and labels
- **Haptic (simulated)** — left/right vibration intensity displayed in the dashboard
- **Audio** — spoken warnings via text-to-speech (pyttsx3)
- **Dashboard metrics** — FPS, detection count, nearest obstacle, avoidance direction

---

## Starting the Dashboard

### Easiest way — one command, works on every device

**Windows:**
```powershell
python start.py
```
Or double-click **`start.bat`**.

**macOS / Linux:**
```bash
python3 start.py
```
Or run `bash start.sh`.

The launcher auto-installs everything and opens **http://localhost:8501** in your browser.

| Tab | Purpose |
|---|---|
| **Live Navigation** | Real-time detection and feedback |
| **Evaluation** | Precision / recall / F1 against ground-truth CSVs |
| **About** | System reference and dissertation context |

---

## Live Navigation Tab

### Step-by-step

1. In the **sidebar**, choose an input source:

   | Source | Use case |
   |---|---|
   | `Simulation` | No camera needed — deterministic synthetic detections, best for demos |
   | `Webcam` | Live camera input (index 0 by default) |
   | `Video file` | Enter a local file path (samples are in `samples/`) |
   | `Upload video` | Drag and drop an `.mp4`, `.avi`, `.mov`, or `.mkv` |

2. Set the **YOLO11 model** path (default `yolo11n.pt` — auto-downloaded on first run).
3. Adjust **confidence threshold** (0.35 is a good starting point).
4. Tick **"Use simulation detector"** if you want deterministic mock detections regardless of source.
5. Optionally enable **text-to-speech** for spoken warnings.
6. Set **Frames per run** (30–900).
7. Press **Run navigation**.

### Reading the output

**Camera feed** — the annotated frame with:
- Zone overlay lines dividing LEFT / CENTRE / RIGHT
- Bounding boxes coloured by warning level (red = critical, orange = high, yellow = medium, green = low)
- Labels: `object confidence | zone | distance`

**Five status metrics** (top row):
- **FPS** — frames processed per second
- **Detections** — number of obstacles in the current frame
- **Nearest** — distance to the closest detected obstacle in metres
- **Haptic** — left/right motor intensities and pattern (e.g. `Left 72% | Right 45% (high)`)
- **Direction** — avoidance suggestion (prefer left · prefer right · slow down · path ahead is relatively clear)

**Detection table** — per-detection breakdown: object, zone, distance, confidence, warning level, message.

**Warning banner** — active audio-style message (e.g. `Person on your left`) and haptic bar showing motor intensities.

**270° sector chart** — polar chart simulating a 270-degree environmental awareness arc. The central 3 sectors correspond to the live camera field of view. Colours follow the warning level palette.

**Session charts** (appear after the first few frames):
- Line chart: detection count and nearest distance per frame
- Stacked bar chart: warning-level distribution per frame

**Session summary** — printed when the run ends: total frames, average FPS, total detections.

---

## Evaluation Tab

Use this tab to measure detection accuracy with bounding-box CSV files.

1. Upload a **ground-truth CSV** (columns: `frame_id, object, xmin, ymin, xmax, ymax`) or tick **"Use sample ground-truth"**.
2. Upload a **predictions CSV** (same columns plus `confidence`) or tick **"Use sample predictions"**.
3. Adjust the **IoU threshold** (default 0.50 — standard COCO metric).
4. Press **Evaluate**.

Results displayed:
- **Precision** — fraction of predictions that matched a ground-truth box
- **Recall** — fraction of ground-truth boxes that were detected
- **F1** — harmonic mean of precision and recall
- **True Positives / False Positives / False Negatives**
- **Metric bar chart**

The bundled sample CSVs in `samples/` were generated from the two synthetic AVI videos and give near-perfect scores by design (verifying the pipeline end-to-end).

---

## CLI Usage

### Run on a sample video (simulation detector, show OpenCV window)
```powershell
python scripts\run_pipeline.py --source samples\left_person_right_vehicle.avi --mock --display
```

### Run on webcam with audio
```powershell
python scripts\run_pipeline.py --source 0 --audio
```

### Run with YOLO11 (no display, log only)
```powershell
python scripts\run_pipeline.py --source samples\centre_stairs_door.avi
```

### Run with a custom model
```powershell
python scripts\run_pipeline.py --source samples\centre_stairs_door.avi --model models\best.pt
```

### Limit frames
```powershell
python scripts\run_pipeline.py --source samples\left_person_right_vehicle.avi --mock --max-frames 60
```

---

## Evaluation CLI

### Precision / recall
```powershell
python evaluation\evaluate_detections.py `
  --ground-truth samples\sample_ground_truth.csv `
  --predictions  samples\sample_predictions.csv `
  --iou 0.5
```

Output (JSON):
```json
{
  "true_positives": 576,
  "false_positives": 0,
  "false_negatives": 0,
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0
}
```

### Latency and FPS benchmark
```powershell
python evaluation\benchmark.py --source samples\centre_stairs_door.avi --mock --frames 120
```

Output:
```
frames: 120
mean_latency_ms: 1.234
p95_latency_ms: 2.100
fps: 94.5
source: samples\centre_stairs_door.avi
model: mock
```

Results are saved to `logs/benchmark_results.csv`.

---

## Haptic Simulation Rules

| Obstacle zone | Motor(s) activated |
|---|---|
| Left | Left motor |
| Right | Right motor |
| Centre | Both motors |

Intensity formula: `1 - (distance / 5.0)` — clamped to `[0.1, 1.0]`.
The haptic bar in the warning panel shows intensity as `#` characters on a 0–10 scale.

---

## Warning Level Rules

| Level | Adjusted distance | Colour | Audio | Haptic |
|---|---|---|---|---|
| **Critical** | ≤ 1.2 m | Red | Yes | Yes |
| **High** | ≤ 2.0 m | Orange | Yes | Yes |
| **Medium** | ≤ 3.5 m | Yellow | Yes | Yes |
| **Low** | > 3.5 m | Green | No | No |

Centre-zone obstacles receive a −0.25 m bias on the adjusted distance (escalate one level earlier).

---

## Avoidance Direction Logic

After each frame the 270° simulator computes a **suggested direction**:

| Condition | Suggestion |
|---|---|
| Centre sector intensity < 0.35 | `path ahead is relatively clear` |
| Left sectors riskier than right | `prefer slight right` |
| Right sectors riskier than left | `prefer slight left` |
| Balanced risk | `slow down` |

---

## CSV Log Reference

Runtime detections are appended to `logs/navigation_events.csv`:

```
timestamp,object,zone,estimated_distance,warning_level
2025-06-15T10:23:01.123456+00:00,person,left,1.84,high
2025-06-15T10:23:01.123456+00:00,vehicle,right,2.41,medium
```

The file is created automatically on first run. Delete it to reset the log between experiments.

---

## Supported Obstacle Classes

| Dashboard label | COCO labels mapped | Notes |
|---|---|---|
| `person` | person, people | Detected by default COCO weights |
| `chair` | chair, couch, sofa, bench | Detected by default COCO weights |
| `table` | dining table, table, desk | Detected by default COCO weights |
| `vehicle` | car, bus, truck, motorcycle, bicycle, train, van | Detected by default COCO weights |
| `door` | door | Requires custom weights or simulation mode |
| `wall` | wall | Requires custom weights or simulation mode |
| `stairs` | stairs, stair, staircase, steps | Requires custom weights or simulation mode |

---

## Dissertation Scope

This prototype does not claim calibrated metric depth. Distance values are **relative monocular estimates** from bounding-box scale and vertical screen position. They are appropriate for hazard urgency ranking and comparative feedback in a dissertation demonstration — not for safety-critical navigation decisions.
