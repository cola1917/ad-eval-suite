from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

try:
	import motmetrics as mm
except Exception:  # pragma: no cover
	mm = None

try:
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception._common import MatcherFn, _infer_scenario_bucket, _resolve_matcher
	from generators.detection_generator import DetectionGenerator
	from matching.iou_matching import bev_iou, center_distance
	from metrics.precision_recall import summarize_detection_frame
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[2]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception._common import MatcherFn, _infer_scenario_bucket, _resolve_matcher
	from generators.detection_generator import DetectionGenerator
	from matching.iou_matching import bev_iou, center_distance
	from metrics.precision_recall import summarize_detection_frame


def _safe_track_id(box: Dict[str, Any], fallback: str) -> str:
	track_id = box.get("track_id") or box.get("source_gt_instance_token") or box.get("instance_token")
	return str(track_id) if track_id else fallback


def _aggregate_track_quality(track_totals: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
	num_tracks = len(track_totals)
	if num_tracks == 0:
		return {"num_tracks": 0, "mt": 0, "ml": 0, "mt_ratio": 0.0, "ml_ratio": 0.0}

	mt = 0
	ml = 0
	for stats in track_totals.values():
		ratio = stats["matched"] / max(1, stats["total"])
		if ratio >= 0.8:
			mt += 1
		if ratio <= 0.2:
			ml += 1

	return {
		"num_tracks": num_tracks,
		"mt": mt,
		"ml": ml,
		"mt_ratio": mt / num_tracks,
		"ml_ratio": ml / num_tracks,
	}


def _create_mot_accumulator() -> Any:
	if mm is None:
		return None
	return mm.MOTAccumulator(auto_id=True)


def _to_numeric_ids(ids: List[str], id_map: Dict[str, int], next_id: List[int]) -> List[int]:
	numeric_ids: List[int] = []
	for track_id in ids:
		if track_id not in id_map:
			id_map[track_id] = next_id[0]
			next_id[0] += 1
		numeric_ids.append(id_map[track_id])
	return numeric_ids


def _build_distance_matrix(
	gt_boxes: List[Dict[str, Any]],
	pred_boxes: List[Dict[str, Any]],
	iou_threshold: float,
	class_aware: bool,
) -> np.ndarray:
	if not gt_boxes or not pred_boxes:
		return np.empty((len(gt_boxes), len(pred_boxes)), dtype=float)

	distance_matrix = np.full((len(gt_boxes), len(pred_boxes)), np.nan, dtype=float)
	for gt_index, gt_box in enumerate(gt_boxes):
		for pred_index, pred_box in enumerate(pred_boxes):
			if class_aware and gt_box.get("category_name") != pred_box.get("category_name"):
				continue
			iou_score = bev_iou(gt_box, pred_box)
			if iou_score < iou_threshold:
				continue
			distance_matrix[gt_index, pred_index] = center_distance(gt_box, pred_box)
	return distance_matrix


def _compute_motmetrics_summary(accumulator: Any) -> Dict[str, float]:
	if mm is None or accumulator is None:
		return {
			"motmetrics_available": False,
			"mota": 0.0,
			"motp": 0.0,
			"idf1": 0.0,
			"num_switches": 0.0,
			"mostly_tracked": 0.0,
			"mostly_lost": 0.0,
		}

	mh = mm.metrics.create()
	metric_names = ["mota", "motp", "idf1", "num_switches", "mostly_tracked", "mostly_lost"]
	summary = mh.compute(accumulator, metrics=metric_names, name="overall")
	row = summary.loc["overall"]
	return {
		"motmetrics_available": True,
		"mota": float(row.get("mota", 0.0)),
		"motp": float(row.get("motp", 0.0)) if not np.isnan(float(row.get("motp", 0.0))) else 0.0,
		"idf1": float(row.get("idf1", 0.0)),
		"num_switches": float(row.get("num_switches", 0.0)),
		"mostly_tracked": float(row.get("mostly_tracked", 0.0)),
		"mostly_lost": float(row.get("mostly_lost", 0.0)),
	}


def evaluate_tracking_frames(
	frame_records: Iterable[Dict[str, Any]],
	prediction_records: Iterable[Dict[str, Any]],
	iou_threshold: float = 0.5,
	matcher: str | MatcherFn = "hungarian",
	class_aware: bool = True,
	center_distance_threshold: float | None = None,
) -> Dict[str, Any]:
	matcher_fn = _resolve_matcher(matcher)
	if center_distance_threshold is not None:
		matcher_fn = partial(matcher_fn, center_distance_threshold=center_distance_threshold)
	frame_records_list = list(frame_records)
	prediction_records_list = list(prediction_records)
	if len(frame_records_list) != len(prediction_records_list):
		raise ValueError(
			f"Mismatched frame/prediction counts: {len(frame_records_list)} vs {len(prediction_records_list)}"
		)

	gt_total = 0
	tp_total = 0
	fp_total = 0
	fn_total = 0
	idsw_total = 0
	distance_sum = 0.0

	last_assignment: Dict[str, str] = {}
	track_totals: Dict[str, Dict[str, int]] = {}

	per_scene_counts: Dict[str, Dict[str, float]] = {}
	per_scenario_counts: Dict[str, Dict[str, float]] = {}
	per_class_counts: Dict[str, Dict[str, float]] = {}

	overall_accumulator = _create_mot_accumulator()
	per_scene_accumulators: Dict[str, Any] = {}
	per_scenario_accumulators: Dict[str, Any] = {}
	gt_id_map: Dict[str, int] = {}
	pred_id_map: Dict[str, int] = {}
	next_gt_id = [1]
	next_pred_id = [1]

	for frame_record, pred_record in zip(frame_records_list, prediction_records_list):
		scene_name = str(frame_record.get("scene_name", "unknown_scene"))
		scenario_name = _infer_scenario_bucket(frame_record)
		gt_boxes = frame_record.get("gt_boxes", [])
		pred_boxes = pred_record.get("pred_boxes", [])

		gt_ids = [_safe_track_id(gt_box, f"gt_{index}") for index, gt_box in enumerate(gt_boxes)]
		pred_ids = [_safe_track_id(pred_box, f"pred_{index}") for index, pred_box in enumerate(pred_boxes)]
		numeric_gt_ids = _to_numeric_ids(gt_ids, gt_id_map, next_gt_id)
		numeric_pred_ids = _to_numeric_ids(pred_ids, pred_id_map, next_pred_id)
		distance_matrix = _build_distance_matrix(gt_boxes, pred_boxes, iou_threshold=iou_threshold, class_aware=class_aware)

		if overall_accumulator is not None:
			overall_accumulator.update(numeric_gt_ids, numeric_pred_ids, distance_matrix)
			scene_acc = per_scene_accumulators.setdefault(scene_name, _create_mot_accumulator())
			scenario_acc = per_scenario_accumulators.setdefault(scenario_name, _create_mot_accumulator())
			if scene_acc is not None:
				scene_acc.update(numeric_gt_ids, numeric_pred_ids, distance_matrix)
			if scenario_acc is not None:
				scenario_acc.update(numeric_gt_ids, numeric_pred_ids, distance_matrix)

		summary = summarize_detection_frame(
			gt_boxes=gt_boxes,
			pred_boxes=pred_boxes,
			iou_threshold=iou_threshold,
			class_aware=class_aware,
			matcher_fn=matcher_fn,
		)
		match_result = summary["match_result"]
		frame_idsw = 0
		frame_idsw_by_class: Dict[str, int] = {}

		tp = int(summary["tp"])
		fp = int(summary["fp"])
		fn = int(summary["fn"])

		gt_total += len(gt_boxes)
		tp_total += tp
		fp_total += fp
		fn_total += fn

		for gt_index, gt_box in enumerate(gt_boxes):
			gt_track_id = _safe_track_id(gt_box, f"gt_{gt_index}")
			stats = track_totals.setdefault(gt_track_id, {"total": 0, "matched": 0})
			stats["total"] += 1
			class_name = str(gt_box.get("category_name", "unknown"))
			class_counts = per_class_counts.setdefault(
				class_name,
				{"gt": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "idsw": 0.0, "distance_sum": 0.0},
			)
			class_counts["gt"] += 1

		for match in match_result.get("matches", []):
			gt_box = match["gt_box"]
			pred_box = match["pred_box"]
			class_name = str(gt_box.get("category_name", "unknown"))
			gt_track_id = _safe_track_id(gt_box, f"gt_{match['gt_index']}")
			pred_track_id = _safe_track_id(pred_box, f"pred_{match['pred_index']}")

			track_totals[gt_track_id]["matched"] += 1
			distance_sum += float(match.get("center_distance", 0.0))
			class_counts = per_class_counts.setdefault(
				class_name,
				{"gt": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "idsw": 0.0, "distance_sum": 0.0},
			)
			class_counts["tp"] += 1
			class_counts["distance_sum"] += float(match.get("center_distance", 0.0))

			previous_pred_track_id = last_assignment.get(gt_track_id)
			if previous_pred_track_id is not None and previous_pred_track_id != pred_track_id:
				idsw_total += 1
				frame_idsw += 1
				frame_idsw_by_class[class_name] = frame_idsw_by_class.get(class_name, 0) + 1
			last_assignment[gt_track_id] = pred_track_id

		for fp_entry in match_result.get("false_positives", []):
			pred_box = fp_entry.get("pred_box", {})
			class_name = str(pred_box.get("category_name", "unknown"))
			class_counts = per_class_counts.setdefault(
				class_name,
				{"gt": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "idsw": 0.0, "distance_sum": 0.0},
			)
			class_counts["fp"] += 1

		for fn_entry in match_result.get("false_negatives", []):
			gt_box = fn_entry.get("gt_box", {})
			class_name = str(gt_box.get("category_name", "unknown"))
			class_counts = per_class_counts.setdefault(
				class_name,
				{"gt": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "idsw": 0.0, "distance_sum": 0.0},
			)
			class_counts["fn"] += 1

		for class_name, switch_count in frame_idsw_by_class.items():
			per_class_counts[class_name]["idsw"] += switch_count

		for bucket in (scene_name, scenario_name):
			container = per_scene_counts if bucket == scene_name else per_scenario_counts
			counts = container.setdefault(bucket, {"gt": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "idsw": 0.0})
			counts["gt"] += len(gt_boxes)
			counts["tp"] += tp
			counts["fp"] += fp
			counts["fn"] += fn
			counts["idsw"] += frame_idsw

	mota = 1.0 - (fn_total + fp_total + idsw_total) / max(1, gt_total)
	motp = distance_sum / max(1, tp_total)
	track_quality = _aggregate_track_quality(track_totals)
	motmetrics_overall = _compute_motmetrics_summary(overall_accumulator)

	if motmetrics_overall["motmetrics_available"]:
		mota = motmetrics_overall["mota"]
		motp = motmetrics_overall["motp"]
		idsw_total = int(motmetrics_overall["num_switches"])
		track_quality["mt"] = int(motmetrics_overall["mostly_tracked"])
		track_quality["ml"] = int(motmetrics_overall["mostly_lost"])
		if track_quality["num_tracks"] > 0:
			track_quality["mt_ratio"] = track_quality["mt"] / track_quality["num_tracks"]
			track_quality["ml_ratio"] = track_quality["ml"] / track_quality["num_tracks"]

	def _finalize_bucket_metrics(raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
		final: Dict[str, Dict[str, float]] = {}
		for name, counts in raw.items():
			mota_value = 1.0 - (counts["fn"] + counts["fp"] + counts["idsw"]) / max(1.0, counts["gt"])
			final[name] = {
				"gt": int(counts["gt"]),
				"tp": int(counts["tp"]),
				"fp": int(counts["fp"]),
				"fn": int(counts["fn"]),
				"idsw": int(counts["idsw"]),
				"mota": mota_value,
			}
		return final

	def _attach_motmetrics_bucket_metrics(
		base: Dict[str, Dict[str, float]],
		accumulators: Dict[str, Any],
	) -> Dict[str, Dict[str, float]]:
		for name, accumulator in accumulators.items():
			if name not in base:
				continue
			mot_summary = _compute_motmetrics_summary(accumulator)
			if mot_summary["motmetrics_available"]:
				base[name]["mota"] = mot_summary["mota"]
				base[name]["motp"] = mot_summary["motp"]
				base[name]["idsw"] = int(mot_summary["num_switches"])
		return base

	per_scene_result = _attach_motmetrics_bucket_metrics(_finalize_bucket_metrics(per_scene_counts), per_scene_accumulators)
	per_scenario_result = _attach_motmetrics_bucket_metrics(
		_finalize_bucket_metrics(per_scenario_counts), per_scenario_accumulators
	)

	per_class_result: Dict[str, Dict[str, float]] = {}
	for class_name, counts in per_class_counts.items():
		gt_count = int(counts["gt"])
		tp_count = int(counts["tp"])
		fp_count = int(counts["fp"])
		fn_count = int(counts["fn"])
		idsw_count = int(counts["idsw"])
		motp_value = float(counts["distance_sum"]) / max(1, tp_count)
		mota_value = 1.0 - (fn_count + fp_count + idsw_count) / max(1, gt_count)
		per_class_result[class_name] = {
			"gt": gt_count,
			"tp": tp_count,
			"fp": fp_count,
			"fn": fn_count,
			"idsw": idsw_count,
			"mota": mota_value,
			"motp": motp_value,
		}

	return {
		"num_frames": len(frame_records_list),
		"matcher": matcher if isinstance(matcher, str) else getattr(matcher, "__name__", "custom"),
		"iou_threshold": iou_threshold,
		"overall": {
			"gt": gt_total,
			"tp": tp_total,
			"fp": fp_total,
			"fn": fn_total,
			"idsw": idsw_total,
			"mota": mota,
			"motp": motp,
			"idf1": motmetrics_overall["idf1"] if motmetrics_overall["motmetrics_available"] else 0.0,
			"motmetrics_available": motmetrics_overall["motmetrics_available"],
			**track_quality,
		},
		"per_scene": per_scene_result,
		"per_scenario": per_scenario_result,
		"per_class": per_class_result,
	}


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Perception tracking evaluator")
	parser.add_argument("--data-root", default="data/nuscenes-mini", help="Path to nuScenes data root")
	parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
	parser.add_argument("--max-frames", type=int, default=5, help="Number of frames")
	parser.add_argument("--seed", type=int, default=42, help="Generator seed")
	parser.add_argument("--matcher", choices=["greedy", "hungarian"], default="hungarian", help="Matcher")
	parser.add_argument("--iou-threshold", type=float, default=0.3, help="Tracking IoU threshold")
	return parser


if __name__ == "__main__":
	args = _build_arg_parser().parse_args()

	loader = NuScenesLoader(data_root=args.data_root, version=args.version, verbose=False)
	loader.load()
	scene_id = loader.get_scene_ids()[0]
	frames = list(loader.iter_frame_records(scene_id=scene_id, max_frames=args.max_frames))
	generator = DetectionGenerator(seed=args.seed)
	preds = [generator.generate_frame_predictions(frame) for frame in frames]

	result = evaluate_tracking_frames(frames, preds, iou_threshold=args.iou_threshold, matcher=args.matcher)
	overall = result["overall"]
	print(f"[tracking] frames={result['num_frames']} matcher={result['matcher']}")
	print(
		f"[tracking] MOTA={overall['mota']:.4f} MOTP={overall['motp']:.4f} IDF1={overall['idf1']:.4f} "
		f"IDSW={overall['idsw']} MT={overall['mt']} ML={overall['ml']}"
	)
	loader.close()
