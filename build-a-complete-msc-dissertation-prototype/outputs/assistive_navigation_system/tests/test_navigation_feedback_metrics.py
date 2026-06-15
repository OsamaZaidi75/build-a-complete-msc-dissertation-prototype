from feedback.haptics import HapticController
from navigation.distance import DistanceEstimator
from navigation.engine import NavigationEngine
from navigation.zones import ZoneAssigner
from evaluation.metrics import BoxRecord, evaluate_precision_recall
from vision.detections import BBox, Detection


def test_zone_assignment_three_regions():
    assigner = ZoneAssigner()
    assert assigner.zone_for_bbox(BBox(0, 0, 90, 100), 300) == "left"
    assert assigner.zone_for_bbox(BBox(110, 0, 190, 100), 300) == "centre"
    assert assigner.zone_for_bbox(BBox(230, 0, 290, 100), 300) == "right"


def test_distance_decreases_for_larger_box():
    estimator = DistanceEstimator()
    small = estimator.estimate("person", BBox(10, 50, 50, 220), (480, 640, 3))
    large = estimator.estimate("person", BBox(10, 50, 220, 460), (480, 640, 3))
    assert large < small


def test_navigation_and_haptics_centre_obstacle_uses_both_motors():
    engine = NavigationEngine()
    detection = Detection("person", 0.9, BBox(250, 100, 390, 470))
    enriched = engine.enrich([detection], (480, 640, 3))
    signal = HapticController().generate(enriched)
    assert enriched[0].zone == "centre"
    assert signal.left_intensity > 0
    assert signal.right_intensity > 0


def test_precision_recall_perfect_match():
    gt = [BoxRecord("0", "person", 0, 0, 100, 100)]
    pred = [BoxRecord("0", "person", 5, 5, 95, 95, 0.9)]
    metrics = evaluate_precision_recall(gt, pred, iou_threshold=0.5)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
