from __future__ import annotations

from typing import Any, Dict, List, Sequence

try:
	from .iou_matching import bev_iou, center_distance
except ImportError:  # pragma: no cover
	from iou_matching import bev_iou, center_distance


MatchResult = Dict[str, Any]


def greedy_match_detections(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	iou_threshold: float = 0.5,
	class_aware: bool = True,
) -> MatchResult:
	"""Greedy one-to-one matching of predictions to GT ordered by score."""

	sorted_predictions = sorted(pred_boxes, key=lambda box: box.get("score", 0.0), reverse=True)
	matched_gt_indices = set()
	matches: List[Dict[str, Any]] = []
	false_positives: List[Dict[str, Any]] = []

	for pred_index, pred_box in enumerate(sorted_predictions):
		best_gt_index = None
		best_iou = 0.0
		for gt_index, gt_box in enumerate(gt_boxes):
			if gt_index in matched_gt_indices:
				continue
			if class_aware and pred_box.get("category_name") != gt_box.get("category_name"):
				continue

			iou_score = bev_iou(gt_box, pred_box)
			if iou_score >= iou_threshold and iou_score > best_iou:
				best_iou = iou_score
				best_gt_index = gt_index

		if best_gt_index is None:
			false_positives.append(
				{
					"pred_index": pred_index,
					"pred_box": pred_box,
					"score": pred_box.get("score", 0.0),
				}
			)
			continue

		matched_gt_indices.add(best_gt_index)
		gt_box = gt_boxes[best_gt_index]
		matches.append(
			{
				"gt_index": best_gt_index,
				"pred_index": pred_index,
				"gt_box": gt_box,
				"pred_box": pred_box,
				"iou": best_iou,
				"center_distance": center_distance(gt_box, pred_box),
				"score": pred_box.get("score", 0.0),
			}
		)

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
		"num_pred": len(sorted_predictions),
		"num_matches": len(matches),
	}


if __name__ == "__main__":
	gt = [
		{"category_name": "car", "translation": [10.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5]},
		{"category_name": "car", "translation": [20.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5]},
	]
	pred = [
		{"category_name": "car", "translation": [10.2, 5.1, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.9},
		{"category_name": "car", "translation": [30.0, 8.0, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.4},
	]
	result = greedy_match_detections(gt, pred)
	print(
		f"[self-test] matches={result['num_matches']} fp={len(result['false_positives'])} "
		f"fn={len(result['false_negatives'])}"
	)
