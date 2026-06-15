"""Warning-level rules for obstacle avoidance."""

from __future__ import annotations


def warning_level_for_distance(distance_m: float, zone: str) -> str:
    """Convert a relative distance estimate and zone into a warning level."""

    centre_bias = 0.25 if zone == "centre" else 0.0
    adjusted = max(0.0, distance_m - centre_bias)
    if adjusted <= 1.2:
        return "critical"
    if adjusted <= 2.0:
        return "high"
    if adjusted <= 3.5:
        return "medium"
    return "low"


def warning_message(label: str, zone: str, level: str) -> str | None:
    """Create a concise audio/dashboard warning message, or None when urgency is too low."""

    if level not in {"critical", "high", "medium"}:
        return None
    if zone == "centre" and level in {"critical", "high"}:
        if label == "stairs":
            return "Stairs ahead — stop"
        return "Obstacle ahead"
    if label == "person":
        return f"Person on your {zone}"
    return f"{label.capitalize()} on your {zone}"
