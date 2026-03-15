from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from eval.perception._common import _resolve_matcher
from metrics.ap_map import compute_map
from metrics.precision_recall import summarize_detection_frame


@dataclass(frozen=True)
class FailureScoreWeights:
	w_fn: float = 1.0
	w_fp: float = 0.5
	w_map: float = 2.0
	w_idsw: float = 1.0


def _safe_track_id(box: Dict[str, Any], fallback: str) -> str:
	track_id = box.get("track_id") or box.get("source_gt_instance_token") or box.get("instance_token")
	return str(track_id) if track_id else fallback


def _compute_frame_id_switches(
	match_result: Dict[str, Any],
	last_assignment: Dict[str, str],
) -> int:
	frame_idsw = 0
	for match in match_result.get("matches", []):
		gt_box = match.get("gt_box", {})
		pred_box = match.get("pred_box", {})
		gt_track_id = _safe_track_id(gt_box, f"gt_{match.get('gt_index', -1)}")
		pred_track_id = _safe_track_id(pred_box, f"pred_{match.get('pred_index', -1)}")

		previous_pred_track_id = last_assignment.get(gt_track_id)
		if previous_pred_track_id is not None and previous_pred_track_id != pred_track_id:
			frame_idsw += 1
		last_assignment[gt_track_id] = pred_track_id

	return frame_idsw


def _frame_failure_score(
	*,
	fn_count: int,
	fp_count: int,
	map_term: float,
	idsw_count: int,
	weights: FailureScoreWeights,
) -> float:
	return (
		weights.w_fn * float(fn_count)
		+ weights.w_fp * float(fp_count)
		+ weights.w_map * float(map_term)
		+ weights.w_idsw * float(idsw_count)
	)


def rank_frames_and_scenes(
	*,
	frame_records: Sequence[Dict[str, Any]],
	prediction_records: Sequence[Dict[str, Any]],
	iou_threshold: float,
	matcher: str,
	class_aware: bool,
	weights: FailureScoreWeights,
) -> Dict[str, Any]:
	if len(frame_records) != len(prediction_records):
		raise ValueError("frame_records and prediction_records length mismatch")

	matcher_fn = _resolve_matcher(matcher)
	last_assignment: Dict[str, str] = {}
	frame_rows: List[Dict[str, Any]] = []
	scene_rows: Dict[str, Dict[str, Any]] = {}
	scene_gt_boxes: Dict[str, List[Dict[str, Any]]] = {}
	scene_pred_boxes: Dict[str, List[Dict[str, Any]]] = {}
	scene_frame_counters: Dict[str, int] = {}

	for index, (frame, pred) in enumerate(zip(frame_records, prediction_records), start=1):
		scene_name = str(frame.get("scene_name", "unknown_scene"))
		sample_token = str(frame.get("sample_token", ""))
		gt_boxes = frame.get("gt_boxes", [])
		pred_boxes = pred.get("pred_boxes", [])

		summary = summarize_detection_frame(
			gt_boxes=gt_boxes,
			pred_boxes=pred_boxes,
			iou_threshold=iou_threshold,
			class_aware=class_aware,
			matcher_fn=matcher_fn,
		)
		frame_idsw = _compute_frame_id_switches(summary.get("match_result", {}), last_assignment)
		frame_score = _frame_failure_score(
			fn_count=int(summary.get("fn", 0)),
			fp_count=int(summary.get("fp", 0)),
			map_term=1.0 - float(summary.get("f1", 0.0)),
			idsw_count=frame_idsw,
			weights=weights,
		)

		scene_frame_counters[scene_name] = scene_frame_counters.get(scene_name, 0) + 1
		scene_frame_index = scene_frame_counters[scene_name]

		frame_row = {
			"rank": 0,
			"failure_score": frame_score,
			"scene_id": scene_name,
			"record_index": index,
			"frame_index": scene_frame_index,
			"sample_token": sample_token,
			"tp": int(summary.get("tp", 0)),
			"fp": int(summary.get("fp", 0)),
			"fn": int(summary.get("fn", 0)),
			"idf_switches": frame_idsw,
			"precision": float(summary.get("precision", 0.0)),
			"recall": float(summary.get("recall", 0.0)),
			"f1": float(summary.get("f1", 0.0)),
			"match_result": summary.get("match_result", {}),
		}
		frame_rows.append(frame_row)

		scene_row = scene_rows.setdefault(
			scene_name,
			{
				"scene_id": scene_name,
				"num_frames": 0,
				"tp": 0,
				"fp": 0,
				"fn": 0,
				"idf_switches": 0,
				"precision": 0.0,
				"recall": 0.0,
				"f1": 0.0,
				"map": 0.0,
				"failure_score": 0.0,
			},
		)
		scene_row["num_frames"] += 1
		scene_row["tp"] += int(summary.get("tp", 0))
		scene_row["fp"] += int(summary.get("fp", 0))
		scene_row["fn"] += int(summary.get("fn", 0))
		scene_row["idf_switches"] += frame_idsw

		scene_gt_boxes.setdefault(scene_name, []).extend(gt_boxes)
		scene_pred_boxes.setdefault(scene_name, []).extend(pred_boxes)

	for scene_name, scene_row in scene_rows.items():
		tp = int(scene_row["tp"])
		fp = int(scene_row["fp"])
		fn = int(scene_row["fn"])
		scene_row["precision"] = tp / (tp + fp) if tp + fp > 0 else 0.0
		scene_row["recall"] = tp / (tp + fn) if tp + fn > 0 else 0.0
		scene_row["f1"] = (
			2.0 * scene_row["precision"] * scene_row["recall"] / (scene_row["precision"] + scene_row["recall"])
			if scene_row["precision"] + scene_row["recall"] > 0
			else 0.0
		)

		classes = sorted({str(box.get("category_name", "")) for box in scene_gt_boxes.get(scene_name, []) if box.get("category_name")})
		if classes:
			scene_map = compute_map(
				gt_boxes=scene_gt_boxes.get(scene_name, []),
				pred_boxes=scene_pred_boxes.get(scene_name, []),
				classes=classes,
				iou_threshold=iou_threshold,
			).get("map", 0.0)
		else:
			scene_map = 0.0
		scene_row["map"] = float(scene_map)
		scene_row["failure_score"] = _frame_failure_score(
			fn_count=fn,
			fp_count=fp,
			map_term=1.0 - float(scene_map),
			idsw_count=int(scene_row["idf_switches"]),
			weights=weights,
		)

	frame_rows = sorted(frame_rows, key=lambda row: row["failure_score"], reverse=True)
	for rank, row in enumerate(frame_rows, start=1):
		row["rank"] = rank

	scene_ranked = sorted(scene_rows.values(), key=lambda row: row["failure_score"], reverse=True)
	for rank, row in enumerate(scene_ranked, start=1):
		row["rank"] = rank

	return {
		"frames": frame_rows,
		"scenes": scene_ranked,
	}
