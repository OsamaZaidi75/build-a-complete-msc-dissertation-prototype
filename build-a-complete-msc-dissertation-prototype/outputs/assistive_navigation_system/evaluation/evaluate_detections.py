"""Command-line precision and recall evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import evaluate_precision_recall, load_box_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection precision and recall.")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth boxes CSV.")
    parser.add_argument("--predictions", required=True, help="Predicted boxes CSV.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a true positive.")
    args = parser.parse_args()

    gt = load_box_csv(args.ground_truth)
    preds = load_box_csv(args.predictions)
    metrics = evaluate_precision_recall(gt, preds, iou_threshold=args.iou)
    print(json.dumps(metrics.__dict__, indent=2))


if __name__ == "__main__":
    main()
