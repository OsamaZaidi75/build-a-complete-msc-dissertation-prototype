"""Detection precision and recall metrics for bounding-box CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoxRecord:
    """One labeled bounding box from a ground-truth or prediction CSV."""

    frame_id: str
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float = 1.0


@dataclass(frozen=True)
class DetectionMetrics:
    """Aggregate detection metrics."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def load_box_csv(path: str | Path) -> list[BoxRecord]:
    """Load boxes from CSV columns: frame_id, object, xmin, ymin, xmax, ymax, confidence."""

    records: list[BoxRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                BoxRecord(
                    frame_id=str(row["frame_id"]),
                    label=str(row.get("object") or row.get("label")).strip().lower(),
                    xmin=float(row["xmin"]),
                    ymin=float(row["ymin"]),
                    xmax=float(row["xmax"]),
                    ymax=float(row["ymax"]),
                    confidence=float(row.get("confidence") or 1.0),
                )
            )
    return records


def evaluate_precision_recall(
    ground_truth: list[BoxRecord],
    predictions: list[BoxRecord],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Evaluate greedy one-to-one IoU matching by frame and class label."""

    matched_gt: set[int] = set()
    true_positives = 0
    false_positives = 0

    sorted_predictions = sorted(predictions, key=lambda record: record.confidence, reverse=True)
    for pred in sorted_predictions:
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue
            if pred.frame_id != gt.frame_id or pred.label != gt.label:
                continue
            overlap = iou(pred, gt)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truth) - len(matched_gt)
    precision = _safe_div(true_positives, true_positives + false_positives)
    recall = _safe_div(true_positives, true_positives + false_negatives)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return DetectionMetrics(true_positives, false_positives, false_negatives, precision, recall, f1)


def iou(a: BoxRecord, b: BoxRecord) -> float:
    """Return intersection-over-union for two box records."""

    inter_x1 = max(a.xmin, b.xmin)
    inter_y1 = max(a.ymin, b.ymin)
    inter_x2 = min(a.xmax, b.xmax)
    inter_y2 = min(a.ymax, b.ymax)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = _area(a) + _area(b) - intersection
    return _safe_div(intersection, union)


def _area(record: BoxRecord) -> float:
    return max(0.0, record.xmax - record.xmin) * max(0.0, record.ymax - record.ymin)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
