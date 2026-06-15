"""Text-to-speech audio feedback with throttling."""

from __future__ import annotations

import time
from dataclasses import dataclass

from vision.detections import Detection


@dataclass
class AudioFeedback:
    """Speak warning messages using pyttsx3 when available."""

    enabled: bool = True
    cooldown_seconds: float = 2.5

    def __post_init__(self) -> None:
        self._last_spoken: dict[str, float] = {}
        self._engine = None
        if self.enabled:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 165)
            except Exception:
                self._engine = None

    def speak_detections(self, detections: list[Detection]) -> list[str]:
        """Speak urgent detection messages and return the messages emitted."""

        if not self.enabled:
            return []
        messages: list[str] = []
        for detection in sorted(detections, key=lambda item: item.estimated_distance or 999):
            if detection.warning_level not in {"critical", "high", "medium"}:
                continue
            message = detection.message or f"{detection.label} detected"
            if self._should_speak(message):
                self._speak(message)
                messages.append(message)
        return messages

    def _should_speak(self, message: str) -> bool:
        now = time.monotonic()
        last = self._last_spoken.get(message, 0.0)
        if now - last < self.cooldown_seconds:
            return False
        self._last_spoken[message] = now
        return True

    def _speak(self, message: str) -> None:
        if self._engine is None:
            print(f"[audio] {message}")
            return
        self._engine.say(message)
        self._engine.runAndWait()
