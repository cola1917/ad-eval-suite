from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
	from matching.iou_matching import bev_iou
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from matching.iou_matching import bev_iou


def compute_precision_recall_curve(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	iou_threshold: float = 0.5,
	class_name: str | None = None,
) -> Dict[str, Any]:
	filtered_gt = [box for box in gt_boxes if class_name is None or box.get("category_name") == class_name]
	filtered_pred = [box for box in pred_boxes if class_name is None or box.get("category_name") == class_name]
	sorted_pred = sorted(filtered_pred, key=lambda box: box.get("score", 0.0), reverse=True)

	precisions: List[float] = []
	recalls: List[float] = []
	scores: List[float] = []
	matched_gt_indices = set()
	tp = 0
	fp = 0
	total_gt = len(filtered_gt)

	for pred_box in sorted_pred:
		best_gt_index = None
		best_iou = 0.0
		for gt_index, gt_box in enumerate(filtered_gt):
			if gt_index in matched_gt_indices:
				continue
			iou_score = bev_iou(gt_box, pred_box)
			if iou_score >= iou_threshold and iou_score > best_iou:
				best_iou = iou_score
				best_gt_index = gt_index

		if best_gt_index is not None:
			matched_gt_indices.add(best_gt_index)
			tp += 1
		else:
			fp += 1

		precision = tp / (tp + fp) if tp + fp > 0 else 0.0
		recall = tp / total_gt if total_gt > 0 else 0.0
		precisions.append(precision)
		recalls.append(recall)
		scores.append(float(pred_box.get("score", 0.0)))

	return {
		"class_name": class_name,
		"num_gt": total_gt,
		"scores": scores,
		"precisions": precisions,
		"recalls": recalls,
	}


def compute_average_precision(precisions: Sequence[float], recalls: Sequence[float]) -> float:
	if not precisions or not recalls:
		return 0.0

	mrec = np.array([0.0, *recalls, 1.0], dtype=float)
	mpre = np.array([0.0, *precisions, 0.0], dtype=float)

	for index in range(len(mpre) - 2, -1, -1):
		mpre[index] = max(mpre[index], mpre[index + 1])

	changing_points = np.where(mrec[1:] != mrec[:-1])[0]
	return float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))


def compute_ap_for_class(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	class_name: str,
	iou_threshold: float = 0.5,
) -> Dict[str, Any]:
	curve = compute_precision_recall_curve(gt_boxes, pred_boxes, iou_threshold=iou_threshold, class_name=class_name)
	ap = compute_average_precision(curve["precisions"], curve["recalls"])
	return {**curve, "ap": ap, "iou_threshold": iou_threshold}


def compute_map(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	classes: Sequence[str],
	iou_threshold: float = 0.5,
) -> Dict[str, Any]:
	per_class = {
		class_name: compute_ap_for_class(gt_boxes, pred_boxes, class_name, iou_threshold=iou_threshold)
		for class_name in classes
	}
	map_score = float(np.mean([result["ap"] for result in per_class.values()])) if per_class else 0.0
	return {
		"iou_threshold": iou_threshold,
		"classes": list(classes),
		"per_class": per_class,
		"map": map_score,
	}


if __name__ == "__main__":
	gt = [
		{"category_name": "car", "translation": [10.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5]},
		{"category_name": "car", "translation": [20.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5]},
	]
	pred = [
		{"category_name": "car", "translation": [10.1, 5.0, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.95},
		{"category_name": "car", "translation": [20.3, 5.2, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.85},
		{"category_name": "car", "translation": [40.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.30},
	]
	result = compute_ap_for_class(gt, pred, class_name="car")
	print(f"[self-test] ap={result['ap']:.4f} points={len(result['precisions'])}")
