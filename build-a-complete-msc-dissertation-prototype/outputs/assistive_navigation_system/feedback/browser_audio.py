"""Browser-based audio feedback using the Web Speech API.

Instead of server-side pyttsx3, this module queues warning messages and
provides a helper that injects a <script> block into the Streamlit page.
The browser's speechSynthesis API handles playback on the client device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vision.detections import Detection


@dataclass
class BrowserAudioFeedback:
    """Queue warning messages for browser-side TTS playback."""

    enabled: bool = True
    cooldown_seconds: float = 2.5

    def __post_init__(self) -> None:
        self._last_spoken: dict[str, float] = {}
        self._pending: list[str] = []

    def speak_detections(self, detections: list[Detection]) -> list[str]:
        """Determine which messages should be spoken and queue them.

        Returns the list of message strings queued this call.
        """
        if not self.enabled:
            return []

        messages: list[str] = []
        for det in sorted(detections, key=lambda d: d.estimated_distance or 999):
            if det.warning_level not in {"critical", "high", "medium"}:
                continue
            message = det.message or f"{det.label} detected"
            if self._should_speak(message):
                self._pending.append(message)
                messages.append(message)
        return messages

    def drain_pending(self) -> list[str]:
        """Return and clear pending messages (called by the UI layer)."""
        msgs = self._pending[:]
        self._pending.clear()
        return msgs

    def _should_speak(self, message: str) -> bool:
        now = time.monotonic()
        last = self._last_spoken.get(message, 0.0)
        if now - last < self.cooldown_seconds:
            return False
        self._last_spoken[message] = now
        return True


def inject_browser_tts(messages: list[str]) -> None:
    """Inject JavaScript to speak messages via the Web Speech API.

    Call this from the Streamlit main thread.  It renders an invisible
    HTML component that triggers speechSynthesis.speak() in the browser.
    """
    if not messages:
        return

    import streamlit.components.v1 as components

    # Escape for safe JS string embedding
    escaped = [m.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"') for m in messages]
    js_array = ", ".join(f'"{m}"' for m in escaped)

    js_code = f"""
    <script>
    (function() {{
        const msgs = [{js_array}];
        if ('speechSynthesis' in window) {{
            msgs.forEach(function(text) {{
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.1;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                window.speechSynthesis.speak(utterance);
            }});
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)
