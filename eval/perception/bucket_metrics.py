"""Detection bucket metrics: FP breakdown and distance/occlusion sub-group evaluations."""

from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
	from eval.perception._common import MatcherFn
	from matching.iou_matching import bev_iou
	from metrics.precision_recall import summarize_detection_frame
	from utils.distance_bucket import DEFAULT_BUCKET_BOUNDARIES, assign_distance_bucket
	from utils.occlusion_bucket import assign_occlusion_bucket, VISIBILITY_LEVELS
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[2]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from eval.perception._common import MatcherFn
	from matching.iou_matching import bev_iou
	from metrics.precision_recall import summarize_detection_frame
	from utils.distance_bucket import DEFAULT_BUCKET_BOUNDARIES, assign_distance_bucket
	from utils.occlusion_bucket import assign_occlusion_bucket, VISIBILITY_LEVELS


def compute_fp_breakdown(
	gt_boxes: Sequence[Dict[str, Any]],
	match_result: Dict[str, Any],
	iou_threshold: float,
) -> Dict[str, int]:
	"""Classify false positives into localization/classification/duplicate/background."""

	breakdown = {
		"localization": 0,
		"classification": 0,
		"duplicate": 0,
		"background": 0,
	}

	matched_gt_indices = {entry["gt_index"] for entry in match_result.get("matches", [])}
	gt_by_class: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
	for gt_index, gt_box in enumerate(gt_boxes):
		gt_by_class[str(gt_box.get("category_name", ""))].append((gt_index, gt_box))
	all_classes = list(gt_by_class.keys())

	for fp_entry in match_result.get("false_positives", []):
		pred_box = fp_entry["pred_box"]
		pred_class = str(pred_box.get("category_name", ""))
		best_iou = 0.0
		best_gt_index = -1
		best_class_match = False

		for gt_index, gt_box in gt_by_class.get(pred_class, []):
			iou_score = bev_iou(gt_box, pred_box)
			if iou_score > best_iou:
				best_iou = iou_score
				best_gt_index = gt_index
				best_class_match = True

		for class_name in all_classes:
			if class_name == pred_class:
				continue
			for gt_index, gt_box in gt_by_class[class_name]:
				iou_score = bev_iou(gt_box, pred_box)
				if iou_score > best_iou:
					best_iou = iou_score
					best_gt_index = gt_index
					best_class_match = False

		if best_gt_index >= 0 and best_iou >= iou_threshold:
			if best_class_match:
				if best_gt_index in matched_gt_indices:
					breakdown["duplicate"] += 1
				else:
					breakdown["localization"] += 1
			else:
				breakdown["classification"] += 1
		elif best_iou >= 0.1:
			if best_class_match:
				breakdown["localization"] += 1
			else:
				breakdown["classification"] += 1
		else:
			breakdown["background"] += 1

	return breakdown


def aggregate_fp_breakdowns(frame_breakdowns: Sequence[Dict[str, int]]) -> Dict[str, int]:
	aggregated = {
		"localization": 0,
		"classification": 0,
		"duplicate": 0,
		"background": 0,
	}
	for breakdown in frame_breakdowns:
		for key in aggregated:
			aggregated[key] += int(breakdown.get(key, 0))
	return aggregated


def compute_distance_bucket_metrics(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	matcher_fn: MatcherFn,
	iou_threshold: float,
	class_aware: bool = True,
	boundaries: Tuple[float, float] = DEFAULT_BUCKET_BOUNDARIES,
) -> Dict[str, Dict[str, Any]]:
	bucket_names = ("near", "medium", "far")
	results: Dict[str, Dict[str, Any]] = {}

	for bucket_name in bucket_names:
		bucket_gt = [
			box
			for box in gt_boxes
			if assign_distance_bucket(float(box.get("distance_to_ego", 0.0)), boundaries) == bucket_name
		]
		bucket_pred = [
			box
			for box in pred_boxes
			if assign_distance_bucket(float(box.get("distance_to_ego", 0.0)), boundaries) == bucket_name
		]

		summary = summarize_detection_frame(
			gt_boxes=bucket_gt,
			pred_boxes=bucket_pred,
			iou_threshold=iou_threshold,
			class_aware=class_aware,
			matcher_fn=matcher_fn,
		)
		results[bucket_name] = summary

	return results


def compute_occlusion_bucket_metrics(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	matcher_fn: MatcherFn,
	iou_threshold: float,
	class_aware: bool = True,
	visibility_mapping: Dict[str, str] = VISIBILITY_LEVELS,
) -> Dict[str, Dict[str, Any]]:
	"""Compute metrics bucketed by occlusion level."""
	bucket_names = ("fully_visible", "mostly_visible", "partially_occluded", "mostly_occluded", "unknown")
	results: Dict[str, Dict[str, Any]] = {}

	for bucket_name in bucket_names:
		bucket_gt = [
			box
			for box in gt_boxes
			if assign_occlusion_bucket(box, visibility_mapping) == bucket_name
		]
		bucket_pred = [
			box
			for box in pred_boxes
			if assign_occlusion_bucket(box, visibility_mapping) == bucket_name
		]

		summary = summarize_detection_frame(
			gt_boxes=bucket_gt,
			pred_boxes=bucket_pred,
			iou_threshold=iou_threshold,
			class_aware=class_aware,
			matcher_fn=matcher_fn,
		)
		results[bucket_name] = summary

	return results


def compute_occlusion_distance_bucket_metrics(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	matcher_fn: MatcherFn,
	iou_threshold: float,
	class_aware: bool = True,
	distance_boundaries: Tuple[float, float] = DEFAULT_BUCKET_BOUNDARIES,
	visibility_mapping: Dict[str, str] = VISIBILITY_LEVELS,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
	"""
	Compute metrics bucketed by both occlusion level and distance.
	Returns nested dict: {occlusion_level: {distance_level: metrics}}
	"""
	occlusion_names = ("fully_visible", "mostly_visible", "partially_occluded", "mostly_occluded", "unknown")
	distance_names = ("near", "medium", "far")
	results: Dict[str, Dict[str, Dict[str, Any]]] = {}

	for occ_name in occlusion_names:
		results[occ_name] = {}
		for dist_name in distance_names:
			bucket_gt = [
				box
				for box in gt_boxes
				if assign_occlusion_bucket(box, visibility_mapping) == occ_name
				and assign_distance_bucket(float(box.get("distance_to_ego", 0.0)), distance_boundaries) == dist_name
			]
			bucket_pred = [
				box
				for box in pred_boxes
				if assign_occlusion_bucket(box, visibility_mapping) == occ_name
				and assign_distance_bucket(float(box.get("distance_to_ego", 0.0)), distance_boundaries) == dist_name
			]

			summary = summarize_detection_frame(
				gt_boxes=bucket_gt,
				pred_boxes=bucket_pred,
				iou_threshold=iou_threshold,
				class_aware=class_aware,
				matcher_fn=matcher_fn,
			)
			results[occ_name][dist_name] = summary

	return results


if __name__ == "__main__":
	gt = [
		{"category_name": "car", "translation": [5.0, 2.0, 0.0], "size": [2.0, 4.0, 1.5], "distance_to_ego": 8.0},
		{"category_name": "car", "translation": [28.0, 3.0, 0.0], "size": [2.0, 4.0, 1.5], "distance_to_ego": 28.2},
	]
	pred = [
		{"category_name": "car", "translation": [5.2, 2.1, 0.0], "size": [2.0, 4.0, 1.5], "distance_to_ego": 8.2, "score": 0.9},
		{"category_name": "car", "translation": [40.0, 3.0, 0.0], "size": [2.0, 4.0, 1.5], "distance_to_ego": 40.1, "score": 0.3},
	]
	from matching.greedy_match import greedy_match_detections

	result = compute_distance_bucket_metrics(gt, pred, greedy_match_detections, iou_threshold=0.3)
	print(f"[self-test] buckets={ {name: metrics['tp'] for name, metrics in result.items()} }")
