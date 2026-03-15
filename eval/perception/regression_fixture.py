from __future__ import annotations

from typing import Any

from eval.perception.detection_eval import evaluate_detection_frames
from eval.perception.tracking_eval import evaluate_tracking_frames


def _frame(scene_description: str, gt_boxes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scene_name": "scene-001",
        "scene_description": scene_description,
        "location": "singapore-hollandvillage",
        "gt_boxes": gt_boxes,
    }


def _gt(track_id: str, category: str, x: float, y: float, distance: float, visibility: str) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "category_name": category,
        "translation": [x, y, 0.0],
        "size": [2.0, 4.0, 1.5],
        "yaw": 0.0,
        "distance_to_ego": distance,
        "visibility_token": visibility,
    }


def _pred(track_id: str, category: str, x: float, y: float, score: float, distance: float, visibility: str) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "category_name": category,
        "translation": [x, y, 0.0],
        "size": [2.0, 4.0, 1.5],
        "yaw": 0.0,
        "score": score,
        "distance_to_ego": distance,
        "visibility_token": visibility,
    }


def build_perception_regression_case() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = [
        _frame(
            "night urban",
            [
                _gt("car-1", "car", 0.0, 0.0, 4.0, "4"),
                _gt("ped-1", "pedestrian", 10.0, 0.0, 10.0, "2"),
            ],
        ),
        _frame(
            "day urban",
            [
                _gt("car-1", "car", 1.0, 0.0, 4.5, "4"),
            ],
        ),
    ]
    predictions = [
        {
            "pred_boxes": [
                _pred("p-car-1", "car", 0.1, 0.0, 0.95, 4.0, "4"),
                _pred("p-car-wrong", "car", 10.1, 0.0, 0.80, 10.1, "2"),
                _pred("p-ped-fp", "pedestrian", 100.0, 100.0, 0.40, 140.0, "1"),
            ]
        },
        {
            "pred_boxes": [
                _pred("p-car-sw", "car", 1.0, 0.0, 0.85, 4.5, "4"),
            ]
        },
    ]
    return frames, predictions


def summarize_perception_regression_case() -> dict[str, Any]:
    frames, predictions = build_perception_regression_case()

    detection_result = evaluate_detection_frames(
        frame_records=frames,
        prediction_records=predictions,
        iou_threshold=0.5,
        matcher="greedy",
        class_aware=True,
    )
    tracking_result = evaluate_tracking_frames(
        frame_records=frames,
        prediction_records=predictions,
        iou_threshold=0.5,
        matcher="greedy",
        class_aware=True,
        metrics_level="standard",
    )

    det_overall = detection_result["overall"]
    trk_overall = tracking_result["overall"]
    return {
        "detection": {
            "num_frames": detection_result["num_frames"],
            "map": detection_result["map"]["map"],
            "overall": {
                "tp": det_overall["tp"],
                "fp": det_overall["fp"],
                "fn": det_overall["fn"],
                "precision": det_overall["precision"],
                "recall": det_overall["recall"],
                "f1": det_overall["f1"],
            },
        },
        "tracking": {
            "num_frames": tracking_result["num_frames"],
            "overall": {
                "gt": trk_overall["gt"],
                "tp": trk_overall["tp"],
                "fp": trk_overall["fp"],
                "fn": trk_overall["fn"],
                "idsw": trk_overall["idsw"],
                "mota": trk_overall["mota"],
                "motp": trk_overall["motp"],
                "mt": trk_overall["mt"],
                "ml": trk_overall["ml"],
            },
        },
    }
