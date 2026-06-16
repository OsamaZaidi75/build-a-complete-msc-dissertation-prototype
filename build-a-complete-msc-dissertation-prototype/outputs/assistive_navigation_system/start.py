"""
One-command launcher for the Assistive Navigation System.
Works on Windows, macOS, and Linux.

Usage:  python start.py
        (or double-click start.bat on Windows / bash start.sh on Mac/Linux)

Windows MAX_PATH note
---------------------
Windows has a 260-character path limit that breaks pip when the project lives
inside a deeply nested folder.  This launcher detects that condition and places
the virtual environment in a short path  (~/.assistive_nav_venv)  so pip never
hits the limit, regardless of where the project folder is stored.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent
REQ            = ROOT / "requirements.txt"
SAMPLES_MARKER = ROOT / "samples" / "left_person_right_vehicle.avi"
APP            = ROOT / "ui" / "streamlit_app.py"

# Windows MAX_PATH limit is 260 characters.
# A .venv sitting ~10 levels deep in a long project path will cause pip to
# fail when writing paths like:
#   <project>\.venv\Lib\site-packages\pkg_resources\tests\data\...egg
# We measure the worst-case depth and redirect the venv to a short home path.
_WORST_CASE_TAIL = r"\.venv\Lib\site-packages\pkg_resources\tests\data\my-test-package_unpacked-egg\my_test_package-1.0-py3.7.egg"
_MAX_PATH        = 255   # leave a few chars headroom below 260

def _pick_venv_dir() -> Path:
    """Return a venv path that is guaranteed to stay under MAX_PATH."""
    local_venv     = ROOT / ".venv"
    worst_case_len = len(str(local_venv)) + len(_WORST_CASE_TAIL)
    if sys.platform != "win32" or worst_case_len <= _MAX_PATH:
        return local_venv
    # Path too long — use a short fixed location in the user home directory.
    safe = Path.home() / ".assistive_nav_venv"
    print(f"  [INFO] Project path is long ({len(str(ROOT))} chars).")
    print(f"         Virtual environment placed at short path: {safe}")
    return safe

VENV = _pick_venv_dir()

if sys.platform == "win32":
    VENV_PYTHON = VENV / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV / "bin" / "python"


# ── Helpers ───────────────────────────────────────────────────────────────
def banner(msg: str) -> None:
    print(f"\n  >>> {msg}")


def run(cmd: list, **kwargs) -> None:
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {' '.join(str(c) for c in cmd)}")
        sys.exit(result.returncode)


# ── Step 1: Python version ─────────────────────────────────────────────────
def check_python_version() -> None:
    if sys.version_info < (3, 10):
        print(
            f"\n[ERROR] Python 3.10+ required. You have "
            f"{sys.version_info.major}.{sys.version_info.minor}.\n"
            "Download Python 3.12 from https://www.python.org/downloads/"
        )
        sys.exit(1)
    print(f"  [OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


# ── Step 2: Venv — create or recreate if broken ───────────────────────────
def _venv_works() -> bool:
    if not VENV_PYTHON.exists():
        return False
    r = subprocess.run(
        [str(VENV_PYTHON), "-c", "import sys; print(sys.version)"],
        capture_output=True,
    )
    return r.returncode == 0


def ensure_venv() -> None:
    if _venv_works():
        print(f"  [OK] Virtual environment ready at {VENV}")
        return

    if VENV.exists():
        banner("Virtual environment is broken — rebuilding…")
        shutil.rmtree(VENV, ignore_errors=True)
    else:
        banner(f"Creating virtual environment at {VENV} …")

    run([sys.executable, "-m", "venv", str(VENV)])

    if not _venv_works():
        print("\n[ERROR] Could not create a working virtual environment.")
        print("  Make sure Python 3.10+ is properly installed.")
        sys.exit(1)

    print("  [OK] Virtual environment created.")


# ── Step 3: Install packages ───────────────────────────────────────────────
def _packages_ready() -> bool:
    r = subprocess.run(
        [str(VENV_PYTHON), "-c", "import streamlit, cv2, ultralytics, streamlit_webrtc, av"],
        capture_output=True,
    )
    return r.returncode == 0


def install_requirements() -> None:
    if _packages_ready():
        print("  [OK] All packages already installed.")
        return

    banner("Installing packages — takes 2-4 minutes on first run…")
    print("  (streamlit, opencv, ultralytics/YOLO11, plotly, pyttsx3 …)\n")

    # Upgrade pip first
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # Install with --no-build-isolation to avoid long temp paths on Windows
    pip_cmd = [
        str(VENV_PYTHON), "-m", "pip", "install",
        "-r", str(REQ),
        "--no-build-isolation",
    ]
    # On Windows also pass a short temp dir to keep build paths under the limit
    if sys.platform == "win32":
        short_tmp = Path.home() / ".pip_tmp"
        short_tmp.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["TMPDIR"]   = str(short_tmp)
        env["TEMP"]     = str(short_tmp)
        env["TMP"]      = str(short_tmp)
        result = subprocess.run(pip_cmd, env=env)
    else:
        result = subprocess.run(pip_cmd)

    if result.returncode != 0:
        print("\n[ERROR] pip install failed.")
        print("  If you see 'filename too long', enable Windows long paths:")
        print("  Run this command in an Admin PowerShell, then retry:\n")
        print("  Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem'")
        print("                   -Name 'LongPathsEnabled' -Value 1\n")
        sys.exit(1)

    print("\n  [OK] All packages installed.")


# ── Step 4: Sample videos ──────────────────────────────────────────────────
def ensure_samples() -> None:
    if SAMPLES_MARKER.exists():
        print("  [OK] Sample videos ready.")
        return
    banner("Generating sample videos…")
    run([str(VENV_PYTHON), str(ROOT / "scripts" / "generate_sample_videos.py")])
    print("  [OK] Sample videos generated.")


# ── Step 5: Launch ─────────────────────────────────────────────────────────
def launch_app() -> None:
    print()
    print("=" * 52)
    print("  Dashboard:  http://localhost:8501")
    print("  Stop:       Ctrl+C")
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


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(ROOT)

    print()
    print("=" * 52)
    print("  AI-Powered Assistive Navigation System")
    print("  MSc Dissertation Prototype")
    print("=" * 52)
    print()

    check_python_version()
    ensure_venv()
    install_requirements()
    ensure_samples()
    launch_app()
