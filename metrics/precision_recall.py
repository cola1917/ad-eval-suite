from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
	from matching.greedy_match import greedy_match_detections
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from matching.greedy_match import greedy_match_detections


MetricSummary = Dict[str, Any]


def compute_detection_counts(match_result: Dict[str, Any]) -> Dict[str, int]:
	return {
		"tp": int(match_result.get("num_matches", 0)),
		"fp": len(match_result.get("false_positives", [])),
		"fn": len(match_result.get("false_negatives", [])),
	}


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
	precision = tp / (tp + fp) if tp + fp > 0 else 0.0
	recall = tp / (tp + fn) if tp + fn > 0 else 0.0
	f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
	return {
		"precision": precision,
		"recall": recall,
		"f1": f1,
	}


def summarize_detection_frame(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	iou_threshold: float = 0.5,
	class_aware: bool = True,
	matcher_fn=greedy_match_detections,
) -> MetricSummary:
	match_result = matcher_fn(
		gt_boxes=gt_boxes,
		pred_boxes=pred_boxes,
		iou_threshold=iou_threshold,
		class_aware=class_aware,
	)
	counts = compute_detection_counts(match_result)
	metrics = compute_precision_recall_f1(**counts)
	return {
		**counts,
		**metrics,
		"iou_threshold": iou_threshold,
		"match_result": match_result,
	}


def aggregate_frame_summaries(frame_summaries: Sequence[MetricSummary]) -> MetricSummary:
	total_tp = sum(summary.get("tp", 0) for summary in frame_summaries)
	total_fp = sum(summary.get("fp", 0) for summary in frame_summaries)
	total_fn = sum(summary.get("fn", 0) for summary in frame_summaries)
	aggregated_metrics = compute_precision_recall_f1(total_tp, total_fp, total_fn)
	return {
		"tp": total_tp,
		"fp": total_fp,
		"fn": total_fn,
		**aggregated_metrics,
		"num_frames": len(frame_summaries),
	}


def summarize_by_class(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	classes: Sequence[str],
	iou_threshold: float = 0.5,
	matcher_fn=greedy_match_detections,
) -> Dict[str, MetricSummary]:
	results: Dict[str, MetricSummary] = {}
	for class_name in classes:
		class_gt = [box for box in gt_boxes if box.get("category_name") == class_name]
		class_pred = [box for box in pred_boxes if box.get("category_name") == class_name]
		results[class_name] = summarize_detection_frame(
			class_gt,
			class_pred,
			iou_threshold=iou_threshold,
			matcher_fn=matcher_fn,
		)
	return results


if __name__ == "__main__":
	gt = [{"category_name": "car", "translation": [10.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5]}]
	pred = [
		{"category_name": "car", "translation": [10.1, 5.0, 0.0], "size": [2.0, 4.0, 1.5], "score": 0.9},
		{"category_name": "pedestrian", "translation": [3.0, 1.0, 0.0], "size": [0.5, 0.5, 1.7], "score": 0.3},
	]
	summary = summarize_detection_frame(gt, pred)
	print(
		f"[self-test] tp={summary['tp']} fp={summary['fp']} fn={summary['fn']} "
		f"precision={summary['precision']:.3f} recall={summary['recall']:.3f}"
	)
