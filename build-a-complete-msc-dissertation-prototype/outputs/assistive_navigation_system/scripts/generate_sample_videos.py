"""Generate synthetic sample videos and scenario CSV files.

The script uses only the Python standard library so the repository can ship
with runnable samples even before OpenCV is installed. Videos are uncompressed
AVI files that OpenCV can read after installing `opencv-python`.
"""

from __future__ import annotations

import csv
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@dataclass(frozen=True)
class ObjectSpec:
    label: str
    bbox: tuple[int, int, int, int]
    color: tuple[int, int, int]
    zone: str
    approx_distance: float


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    width, height, fps, frames = 320, 180, 12, 96

    scenarios = [
        ("left_person_right_vehicle.avi", scene_left_person_right_vehicle),
        ("centre_stairs_door.avi", scene_centre_stairs_door),
    ]

    scenario_rows: list[dict[str, object]] = []
    gt_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []

    for video_name, scene_fn in scenarios:
        video_frames: list[bytearray] = []
        for frame_idx in range(frames):
            objects = scene_fn(frame_idx, width, height)
            frame = draw_scene(width, height, objects)
            video_frames.append(frame)
            for obj in objects:
                x1, y1, x2, y2 = obj.bbox
                scenario_rows.append(
                    {
                        "video": video_name,
                        "frame": frame_idx,
                        "object": obj.label,
                        "zone": obj.zone,
                        "approx_distance_m": obj.approx_distance,
                    }
                )
                frame_id = f"{video_name}:{frame_idx}"
                gt_rows.append(
                    {
                        "frame_id": frame_id,
                        "object": obj.label,
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                        "confidence": 1.0,
                    }
                )
                pred_rows.append(
                    {
                        "frame_id": frame_id,
                        "object": obj.label,
                        "xmin": max(0, x1 + 2),
                        "ymin": max(0, y1 + 1),
                        "xmax": min(width, x2 - 2),
                        "ymax": min(height, y2 - 1),
                        "confidence": 0.82,
                    }
                )
        write_uncompressed_avi(SAMPLES / video_name, video_frames, width, height, fps)

    write_csv(SAMPLES / "simulation_scenarios.csv", scenario_rows)
    write_csv(SAMPLES / "sample_ground_truth.csv", gt_rows)
    write_csv(SAMPLES / "sample_predictions.csv", pred_rows)
    print(f"Generated {len(scenarios)} videos and CSV scenarios in {SAMPLES}")


def scene_left_person_right_vehicle(frame_idx: int, width: int, height: int) -> list[ObjectSpec]:
    wave = math.sin(frame_idx / 13.0)
    person_w, person_h = 34, 102
    person_x = int(35 + 8 * wave)
    person_y = height - person_h - 14
    vehicle_w, vehicle_h = 72, 58
    vehicle_x = int(width - vehicle_w - 34 - 10 * math.cos(frame_idx / 16.0))
    vehicle_y = height - vehicle_h - 20
    chair_w, chair_h = 44, 48
    chair_x = int(width * 0.44)
    chair_y = height - chair_h - 18
    return [
        ObjectSpec("person", (person_x, person_y, person_x + person_w, person_y + person_h), (60, 170, 255), "left", 1.8),
        ObjectSpec("vehicle", (vehicle_x, vehicle_y, vehicle_x + vehicle_w, vehicle_y + vehicle_h), (240, 105, 60), "right", 2.4),
        ObjectSpec("chair", (chair_x, chair_y, chair_x + chair_w, chair_y + chair_h), (115, 220, 125), "centre", 2.1),
    ]


def scene_centre_stairs_door(frame_idx: int, width: int, height: int) -> list[ObjectSpec]:
    stairs_w = 90 + int(8 * math.sin(frame_idx / 12.0))
    stairs_h = 70
    stairs_x = (width - stairs_w) // 2
    stairs_y = height - stairs_h - 8
    door_w, door_h = 46, 100
    door_x = width - door_w - 45
    door_y = height - door_h - 20
    wall_w, wall_h = 52, 116
    wall_x = 20
    wall_y = height - wall_h - 16
    table_w, table_h = 58, 42
    table_x = 64
    table_y = height - table_h - 18
    return [
        ObjectSpec("stairs", (stairs_x, stairs_y, stairs_x + stairs_w, stairs_y + stairs_h), (230, 190, 80), "centre", 1.3),
        ObjectSpec("door", (door_x, door_y, door_x + door_w, door_y + door_h), (150, 100, 210), "right", 2.8),
        ObjectSpec("wall", (wall_x, wall_y, wall_x + wall_w, wall_y + wall_h), (155, 160, 166), "left", 2.2),
        ObjectSpec("table", (table_x, table_y, table_x + table_w, table_y + table_h), (90, 210, 190), "left", 1.9),
    ]


def draw_scene(width: int, height: int, objects: list[ObjectSpec]) -> bytearray:
    canvas = new_canvas(width, height, (35, 40, 47))
    horizon = int(height * 0.45)
    fill_rect(canvas, width, height, 0, horizon, width, height, (50, 57, 63))
    draw_line(canvas, width, height, width // 2, horizon, 26, height, (85, 96, 106))
    draw_line(canvas, width, height, width // 2, horizon, width - 26, height, (85, 96, 106))
    draw_line(canvas, width, height, width // 3, 0, width // 3, height, (115, 124, 132))
    draw_line(canvas, width, height, (2 * width) // 3, 0, (2 * width) // 3, height, (115, 124, 132))

    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        fill_rect(canvas, width, height, x1, y1, x2, y2, obj.color)
        draw_rect(canvas, width, height, x1, y1, x2, y2, (245, 245, 245))
        if obj.label == "stairs":
            step_h = max(5, (y2 - y1) // 6)
            for y in range(y1 + step_h, y2, step_h):
                draw_line(canvas, width, height, x1, y, x2, y, (95, 80, 55))
        elif obj.label == "door":
            fill_rect(canvas, width, height, x2 - 9, (y1 + y2) // 2, x2 - 5, (y1 + y2) // 2 + 4, (30, 25, 35))
        elif obj.label == "vehicle":
            fill_rect(canvas, width, height, x1 + 8, y1 + 8, x2 - 8, y1 + 22, (60, 90, 120))
        elif obj.label == "person":
            head = max(5, (x2 - x1) // 4)
            fill_rect(canvas, width, height, x1 + head, y1 + 5, x2 - head, y1 + 5 + head, (245, 210, 170))
    return canvas


def new_canvas(width: int, height: int, color: tuple[int, int, int]) -> bytearray:
    r, g, b = color
    return bytearray([r, g, b] * width * height)


def set_pixel(canvas: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if not (0 <= x < width and 0 <= y < height):
        return
    idx = (y * width + x) * 3
    canvas[idx : idx + 3] = bytes(color)


def fill_rect(
    canvas: bytearray,
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    row = bytes(color) * max(0, x2 - x1)
    for y in range(y1, y2):
        start = (y * width + x1) * 3
        canvas[start : start + len(row)] = row


def draw_rect(
    canvas: bytearray,
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    draw_line(canvas, width, height, x1, y1, x2, y1, color)
    draw_line(canvas, width, height, x2, y1, x2, y2, color)
    draw_line(canvas, width, height, x2, y2, x1, y2, color)
    draw_line(canvas, width, height, x1, y2, x1, y1, color)


def draw_line(
    canvas: bytearray,
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    while True:
        set_pixel(canvas, width, height, x1, y1, color)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy


def write_uncompressed_avi(path: Path, rgb_frames: list[bytearray], width: int, height: int, fps: int) -> None:
    if not rgb_frames:
        raise ValueError("No frames supplied")
    row_size = ((width * 3 + 3) // 4) * 4
    frame_size = row_size * height
    frame_count = len(rgb_frames)
    avi_frames = [rgb_to_bgr_bottom_up(frame, width, height, row_size) for frame in rgb_frames]

    avih = struct.pack(
        "<IIIIIIIIII4I",
        int(1_000_000 / fps),
        frame_size * fps,
        0,
        0x10,
        frame_count,
        0,
        1,
        frame_size,
        width,
        height,
        0,
        0,
        0,
        0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        fps,
        0,
        frame_count,
        frame_size,
        0xFFFFFFFF,
        0,
        0,
        0,
        width,
        height,
    )
    strf = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, frame_size, 0, 0, 0, 0)

    hdrl = make_list(
        b"hdrl",
        make_chunk(b"avih", avih)
        + make_list(b"strl", make_chunk(b"strh", strh) + make_chunk(b"strf", strf)),
    )

    movi_data = bytearray()
    index_data = bytearray()
    offset = 4
    for frame in avi_frames:
        chunk = make_chunk(b"00db", frame)
        movi_data.extend(chunk)
        index_data.extend(struct.pack("<4sIII", b"00db", 0x10, offset, len(frame)))
        offset += len(chunk)

    body = hdrl + make_list(b"movi", bytes(movi_data)) + make_chunk(b"idx1", bytes(index_data))
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"AVI " + body)


def rgb_to_bgr_bottom_up(frame: bytearray, width: int, height: int, row_size: int) -> bytes:
    output = bytearray()
    pad = b"\x00" * (row_size - width * 3)
    for y in range(height - 1, -1, -1):
        row_start = y * width * 3
        for x in range(width):
            idx = row_start + x * 3
            r, g, b = frame[idx], frame[idx + 1], frame[idx + 2]
            output.extend((b, g, r))
        output.extend(pad)
    return bytes(output)


def make_chunk(chunk_id: bytes, data: bytes) -> bytes:
    padding = b"\x00" if len(data) % 2 else b""
    return chunk_id + struct.pack("<I", len(data)) + data + padding


def make_list(list_type: bytes, data: bytes) -> bytes:
    payload = list_type + data
    padding = b"\x00" if len(payload) % 2 else b""
    return b"LIST" + struct.pack("<I", len(payload)) + payload + padding


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to generate samples: {exc}", file=sys.stderr)
        raise
