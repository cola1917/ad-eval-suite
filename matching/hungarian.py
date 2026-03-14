from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from scipy.optimize import linear_sum_assignment

try:
	from .iou_matching import bev_iou, center_distance
except ImportError:  # pragma: no cover
	from iou_matching import bev_iou, center_distance


MatchResult = Dict[str, Any]


def hungarian_match_detections(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	iou_threshold: float = 0.5,
	class_aware: bool = True,
	center_distance_threshold: float | None = None,
) -> MatchResult:
	"""Optimal one-to-one matching using Hungarian assignment on IoU cost."""
	if not gt_boxes:
		return {
			"matches": [],
			"false_positives": [
				{"pred_index": index, "pred_box": pred_box, "score": pred_box.get("score", 0.0)}
				for index, pred_box in enumerate(pred_boxes)
			],
			"false_negatives": [],
			"num_gt": 0,
			"num_pred": len(pred_boxes),
			"num_matches": 0,
		}

	if not pred_boxes:
		return {
			"matches": [],
			"false_positives": [],
			"false_negatives": [
				{"gt_index": index, "gt_box": gt_box}
				for index, gt_box in enumerate(gt_boxes)
			],
			"num_gt": len(gt_boxes),
			"num_pred": 0,
			"num_matches": 0,
		}

	matched_gt_indices = set()
	matched_pred_indices = set()
	matches: List[Dict[str, Any]] = []

	invalid_cost = 1e6
	if class_aware:
		gt_by_class: Dict[str, List[int]] = defaultdict(list)
		pred_by_class: Dict[str, List[int]] = defaultdict(list)
		for gt_index, gt_box in enumerate(gt_boxes):
			gt_by_class[str(gt_box.get("category_name", ""))].append(gt_index)
		for pred_index, pred_box in enumerate(pred_boxes):
			pred_by_class[str(pred_box.get("category_name", ""))].append(pred_index)

		for class_name, pred_indices in pred_by_class.items():
			gt_indices = gt_by_class.get(class_name, [])
			if not gt_indices or not pred_indices:
				continue

			cost_matrix: List[List[float]] = []
			for gt_index in gt_indices:
				gt_box = gt_boxes[gt_index]
				row: List[float] = []
				for pred_index in pred_indices:
					pred_box = pred_boxes[pred_index]
					iou_score = bev_iou(gt_box, pred_box)
					if iou_score < iou_threshold:
						row.append(invalid_cost)
						continue
					if center_distance_threshold is not None and center_distance(gt_box, pred_box) > center_distance_threshold:
						row.append(invalid_cost)
						continue
					row.append(1.0 - iou_score)
				cost_matrix.append(row)

			row_indices, col_indices = linear_sum_assignment(cost_matrix)
			for row_index, col_index in zip(row_indices, col_indices):
				if cost_matrix[row_index][col_index] >= invalid_cost:
					continue
				gt_index = gt_indices[row_index]
				pred_index = pred_indices[col_index]
				gt_box = gt_boxes[gt_index]
				pred_box = pred_boxes[pred_index]
				iou_score = bev_iou(gt_box, pred_box)
				if iou_score < iou_threshold:
					continue

				matched_gt_indices.add(gt_index)
				matched_pred_indices.add(pred_index)
				matches.append(
					{
						"gt_index": gt_index,
						"pred_index": pred_index,
						"gt_box": gt_box,
						"pred_box": pred_box,
						"iou": iou_score,
						"center_distance": center_distance(gt_box, pred_box),
						"score": pred_box.get("score", 0.0),
					}
				)
	else:
		cost_matrix: List[List[float]] = []
		for gt_box in gt_boxes:
			row: List[float] = []
			for pred_box in pred_boxes:
				iou_score = bev_iou(gt_box, pred_box)
				if iou_score < iou_threshold:
					row.append(invalid_cost)
					continue
				if center_distance_threshold is not None and center_distance(gt_box, pred_box) > center_distance_threshold:
					row.append(invalid_cost)
					continue
				row.append(1.0 - iou_score)
			cost_matrix.append(row)

		row_indices, col_indices = linear_sum_assignment(cost_matrix)
		for gt_index, pred_index in zip(row_indices, col_indices):
			if cost_matrix[gt_index][pred_index] >= invalid_cost:
				continue
			gt_box = gt_boxes[gt_index]
			pred_box = pred_boxes[pred_index]
			iou_score = bev_iou(gt_box, pred_box)
			if iou_score < iou_threshold:
				continue

			matched_gt_indices.add(gt_index)
			matched_pred_indices.add(pred_index)
			matches.append(
				{
					"gt_index": gt_index,
					"pred_index": pred_index,
					"gt_box": gt_box,
					"pred_box": pred_box,
					"iou": iou_score,
					"center_distance": center_distance(gt_box, pred_box),
					"score": pred_box.get("score", 0.0),
				}
			)

	false_positives = [
		{"pred_index": pred_index, "pred_box": pred_box, "score": pred_box.get("score", 0.0)}
		for pred_index, pred_box in enumerate(pred_boxes)
		if pred_index not in matched_pred_indices
	]
	false_negatives = [
		{"gt_index": gt_index, "gt_box": gt_box}
		for gt_index, gt_box in enumerate(gt_boxes)
		if gt_index not in matched_gt_indices
	]

	return {
		"matches": matches,
		"false_positives": false_positives,
		"false_negatives": false_negatives,
		"num_gt": len(gt_boxes),
		"num_pred": len(pred_boxes),
		"num_matches": len(matches),
	}


if __name__ == "__main__":
	gt = [
		{"category_name": "car", "translation": [10.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5]},
		{"category_name": "car", "translation": [20.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5]},
	]
	pred = [
		{"category_name": "car", "translation": [10.2, 5.0, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.9},
		{"category_name": "car", "translation": [20.1, 5.2, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.8},
		{"category_name": "car", "translation": [40.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.3},
	]
	result = hungarian_match_detections(gt, pred, iou_threshold=0.3)
	print(
		f"[self-test] matches={result['num_matches']} fp={len(result['false_positives'])} "
		f"fn={len(result['false_negatives'])}"
	)
