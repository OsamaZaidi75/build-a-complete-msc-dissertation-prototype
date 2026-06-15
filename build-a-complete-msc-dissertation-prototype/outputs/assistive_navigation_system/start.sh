#!/usr/bin/env bash
# One-click launcher for macOS and Linux.
# Double-click in Finder/Files, or run:  bash start.sh

set -e
cd "$(dirname "$0")"

echo ""
echo "===================================================="
echo "  AI-Powered Assistive Navigation System"
echo "  MSc Dissertation Prototype"
echo "===================================================="
echo ""
echo "  Starting... (first run installs dependencies)"
echo ""

python3 start.py
