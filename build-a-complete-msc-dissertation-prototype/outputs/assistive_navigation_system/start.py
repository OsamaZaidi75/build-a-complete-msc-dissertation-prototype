"""
One-command launcher for the Assistive Navigation System.
Works on Windows, macOS, and Linux.

Usage:  python start.py
        (or double-click start.bat on Windows / bash start.sh on Mac/Linux)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQ  = ROOT / "requirements.txt"
SAMPLES_MARKER = ROOT / "samples" / "left_person_right_vehicle.avi"
APP  = ROOT / "ui" / "streamlit_app.py"

if sys.platform == "win32":
    VENV_PYTHON = VENV / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV / "bin" / "python"


def banner(msg: str) -> None:
    print(f"\n  >>> {msg}")


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {' '.join(str(c) for c in cmd)}")
        sys.exit(result.returncode)
    return result


# ── Step 1: Python version ────────────────────────────────────────────────
def check_python_version() -> None:
    if sys.version_info < (3, 10):
        print(
            f"\n[ERROR] Python 3.10+ required. You have "
            f"{sys.version_info.major}.{sys.version_info.minor}.\n"
            "Download Python 3.12 from https://www.python.org/downloads/"
        )
        sys.exit(1)
    print(f"  [OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


# ── Step 2: Venv — create or recreate if broken ──────────────────────────
def venv_python_works() -> bool:
    """Return True only if the venv Python actually executes."""
    if not VENV_PYTHON.exists():
        return False
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import sys; print(sys.version)"],
        capture_output=True,
    )
    return result.returncode == 0


def ensure_venv() -> None:
    if venv_python_works():
        print("  [OK] Virtual environment is healthy.")
        return

    if VENV.exists():
        banner("Virtual environment is broken — rebuilding it…")
        shutil.rmtree(VENV)
    else:
        banner("Creating virtual environment…")

    run([sys.executable, "-m", "venv", str(VENV)])

    if not venv_python_works():
        print("\n[ERROR] Could not create a working virtual environment.")
        print("  Make sure Python 3.10+ is properly installed.")
        sys.exit(1)

    print("  [OK] Virtual environment ready.")


# ── Step 3: Install / upgrade packages ───────────────────────────────────
def packages_installed() -> bool:
    """Quick check: are the three heaviest packages present?"""
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import streamlit, cv2, ultralytics"],
        capture_output=True,
    )
    return result.returncode == 0


def install_requirements() -> None:
    if packages_installed():
        print("  [OK] All packages already installed.")
        return
    banner("Installing packages — this takes 2-4 minutes on first run…")
    print("  (installing: streamlit, opencv, ultralytics/YOLO11, plotly, pyttsx3 …)\n")
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "-q"])
    run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQ)])
    print("\n  [OK] All packages installed.")


# ── Step 4: Sample videos ─────────────────────────────────────────────────
def ensure_samples() -> None:
    if SAMPLES_MARKER.exists():
        print("  [OK] Sample videos ready.")
        return
    banner("Generating sample videos…")
    run([str(VENV_PYTHON), str(ROOT / "scripts" / "generate_sample_videos.py")])
    print("  [OK] Sample videos generated.")


# ── Step 5: Launch ────────────────────────────────────────────────────────
def launch_app() -> None:
    print()
    print("=" * 52)
    print("  Dashboard ready at:  http://localhost:8501")
    print("  Stop with:           Ctrl+C")
    print("=" * 52)
    print()

    cmd = [
        str(VENV_PYTHON), "-m", "streamlit", "run", str(APP),
        "--server.headless", "false",
    ]

    if sys.platform == "win32":
        subprocess.run(cmd)
    else:
        os.execv(str(VENV_PYTHON), cmd)


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(ROOT)

    print()
    print("=" * 52)
    print("  AI-Powered Assistive Navigation System")
    print("  MSc Dissertation Prototype")
    print("=" * 52)
    print()
    print("  Webcam | Video files | Simulation | Real YOLO11")
    print()

    check_python_version()
    ensure_venv()
    install_requirements()
    ensure_samples()
    launch_app()
