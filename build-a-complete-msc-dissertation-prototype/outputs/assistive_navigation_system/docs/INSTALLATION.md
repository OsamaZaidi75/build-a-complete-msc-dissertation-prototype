# Installation Guide

## Easiest Way — One Command on Any Device

All you need is **Python 3.10+** installed. Everything else is automatic.

### Windows
```powershell
cd outputs\assistive_navigation_system
python start.py
```
Or **double-click `start.bat`** in File Explorer — no terminal needed.

### macOS / Linux
```bash
cd outputs/assistive_navigation_system
python3 start.py
```
Or run `bash start.sh`.

The launcher will:
1. Check your Python version
2. Create a `.venv` virtual environment (once)
3. Install all packages from `requirements.txt` (once, ~2 min)
4. Generate synthetic sample videos (once, instant)
5. Open the dashboard at **http://localhost:8501**

> On first YOLO run the `yolo11n.pt` weights (~6 MB) are downloaded automatically.
> Tick **"Use simulation detector"** in the sidebar to skip this and run fully offline.

---

## System Requirements

- Python **3.10 or later** (3.12 recommended — the bundled venv uses 3.12)
- Windows 10/11, macOS, or Linux
- Webcam or video files for live input (optional — simulation mode works without hardware)
- NVIDIA GPU optional — YOLO11 runs on CPU with acceptable performance at 640px

---

## Local Python Setup (Windows)

```powershell
# 1. Open a terminal in the project root
cd "outputs\assistive_navigation_system"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt

# 6. Generate synthetic sample videos (one-time, stdlib only — no OpenCV needed)
python scripts\generate_sample_videos.py

# 7. Launch the Streamlit dashboard
streamlit run ui\streamlit_app.py
```

The dashboard opens automatically at **http://localhost:8501**.

---

## Local Python Setup (macOS / Linux)

```bash
cd outputs/assistive_navigation_system
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_sample_videos.py
streamlit run ui/streamlit_app.py
```

---

## YOLO11 Weights

`yolo11n.pt` (~6 MB) is downloaded automatically by Ultralytics on first use. An internet connection is required for the initial download only. The file is cached locally in `~/.config/Ultralytics/` (Linux/macOS) or `C:\Users\<user>\AppData\Roaming\Ultralytics\` (Windows).

To skip the download entirely, enable **"Use simulation detector"** in the dashboard sidebar.

To use a custom dissertation checkpoint trained on doors, walls, and stairs:

```powershell
# In the dashboard sidebar, enter the model path:
models\best.pt

# Or via CLI:
python scripts\run_pipeline.py --source samples\centre_stairs_door.avi --model models\best.pt
```

---

## Docker Setup

```powershell
# Build the image
docker build -t assistive-nav-yolo11 .

# Run (CPU only, no webcam)
docker run --rm -p 8501:8501 assistive-nav-yolo11
```

Open **http://localhost:8501**.

For webcam access on Linux:

```bash
docker run --rm -p 8501:8501 --device=/dev/video0 assistive-nav-yolo11
```

For GPU acceleration in Docker:

```bash
docker run --rm -p 8501:8501 --gpus all assistive-nav-yolo11
```

---

## Installed Packages

| Package | Purpose |
|---|---|
| `ultralytics` | YOLO11 inference |
| `opencv-python` | Video capture and frame annotation |
| `streamlit` | Dashboard UI |
| `plotly` | Polar sector chart and session charts |
| `pandas` | Dataframe display in dashboard |
| `numpy` | Array operations for synthetic frames |
| `pyttsx3` | Offline text-to-speech audio alerts |
| `pytest` | Unit test runner |

---

## Running Tests

```powershell
# With venv active:
python -m pytest tests/ -v

# Without activating the venv:
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected output — all 4 tests pass:

```
tests/test_navigation_feedback_metrics.py::test_zone_assignment_three_regions PASSED
tests/test_navigation_feedback_metrics.py::test_distance_decreases_for_larger_box PASSED
tests/test_navigation_feedback_metrics.py::test_navigation_and_haptics_centre_obstacle_uses_both_motors PASSED
tests/test_navigation_feedback_metrics.py::test_precision_recall_perfect_match PASSED
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cv2` not found | `pip install opencv-python` |
| `pyttsx3` audio not working | Audio falls back to console print — the app still runs |
| YOLO weights fail to download | Enable **"Use simulation detector"** in the sidebar |
| Webcam not found | Try changing the webcam index to `1` or `2` in the sidebar |
| Streamlit port already in use | `streamlit run ui\streamlit_app.py --server.port 8502` |
| PowerShell execution policy error | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Windows venv activation fails | Use `cmd.exe` and run `.venv\Scripts\activate.bat` |
| Docker webcam on Windows | Use local Python instead — Docker on Windows cannot access USB cameras |
