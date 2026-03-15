from __future__ import annotations

import pytest

from metrics.ap_map import compute_ap_for_class, compute_map
from metrics.precision_recall import compute_precision_recall_f1


def _box(category: str, x: float, y: float, score: float | None = None) -> dict:
    box = {
        "category_name": category,
        "translation": [x, y, 0.0],
        "size": [2.0, 4.0, 1.5],
        "yaw": 0.0,
    }
    if score is not None:
        box["score"] = score
    return box


def test_precision_recall_f1_counts() -> None:
    metrics = compute_precision_recall_f1(tp=8, fp=2, fn=4)
    assert metrics["precision"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["f1"] == pytest.approx(8.0 / 11.0)


def test_ap_for_class_perfect_detection() -> None:
    gt = [_box("car", 0.0, 0.0), _box("car", 10.0, 0.0)]
    pred = [_box("car", 0.0, 0.0, score=0.95), _box("car", 10.0, 0.0, score=0.90)]

    ap_result = compute_ap_for_class(gt_boxes=gt, pred_boxes=pred, class_name="car", iou_threshold=0.5)
    assert ap_result["ap"] == pytest.approx(1.0)


def test_map_multiclass_expected_value() -> None:
    gt = [
        _box("car", 0.0, 0.0),
        _box("car", 10.0, 0.0),
        _box("pedestrian", 30.0, 0.0),
    ]
    pred = [
        _box("car", 0.1, 0.0, score=0.95),
        _box("car", 10.1, 0.0, score=0.85),
        _box("car", 30.1, 0.0, score=0.80),
        _box("pedestrian", 100.0, 100.0, score=0.40),
    ]

    result = compute_map(gt_boxes=gt, pred_boxes=pred, classes=["car", "pedestrian"], iou_threshold=0.5)

    assert result["per_class"]["car"]["ap"] == pytest.approx(1.0)
    assert result["per_class"]["pedestrian"]["ap"] == pytest.approx(0.0)
    assert result["map"] == pytest.approx(0.5)
