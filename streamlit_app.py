"""Streamlit Cloud entry point — delegates to the nested app."""
import sys
import os
from pathlib import Path

# Point Python to the project's module root
APP_DIR = Path(__file__).parent / "build-a-complete-msc-dissertation-prototype" / "outputs" / "assistive_navigation_system"
sys.path.insert(0, str(APP_DIR))

# Set working directory so relative paths (logs/, samples/) resolve correctly
os.chdir(APP_DIR)

from ui.streamlit_app import main

main()
