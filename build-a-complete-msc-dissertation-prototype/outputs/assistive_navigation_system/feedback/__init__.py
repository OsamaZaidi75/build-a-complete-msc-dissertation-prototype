"""Feedback modules for haptic simulation, audio alerts, and logging."""

from .audio import AudioFeedback
from .haptics import HapticController, HapticSignal
from .logger import DetectionLogger

__all__ = ["AudioFeedback", "DetectionLogger", "HapticController", "HapticSignal"]
