from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

try:
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.metrics import (
		aggregate_fp_breakdowns,
		compute_distance_bucket_metrics,
		compute_fp_breakdown,
		compute_occlusion_bucket_metrics,
		compute_occlusion_distance_bucket_metrics,
	)
	from generators.detection_generator import DetectionGenerator
	from matching.greedy_match import greedy_match_detections
	from matching.hungarian import hungarian_match_detections
	from metrics.ap_map import compute_map
	from metrics.precision_recall import aggregate_frame_summaries, summarize_by_class, summarize_detection_frame
	from utils.distance_bucket import DEFAULT_BUCKET_BOUNDARIES, bucket_label_with_ranges
	from utils.occlusion_bucket import occlusion_bucket_labels, VISIBILITY_LEVELS
	from utils.visualization import save_detection_bev_plot
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[2]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.metrics import (
		aggregate_fp_breakdowns,
		compute_distance_bucket_metrics,
		compute_fp_breakdown,
		compute_occlusion_bucket_metrics,
		compute_occlusion_distance_bucket_metrics,
	)
	from generators.detection_generator import DetectionGenerator
	from matching.greedy_match import greedy_match_detections
	from matching.hungarian import hungarian_match_detections
	from metrics.ap_map import compute_map
	from metrics.precision_recall import aggregate_frame_summaries, summarize_by_class, summarize_detection_frame
	from utils.distance_bucket import DEFAULT_BUCKET_BOUNDARIES, bucket_label_with_ranges
	from utils.occlusion_bucket import occlusion_bucket_labels, VISIBILITY_LEVELS
	from utils.visualization import save_detection_bev_plot


MatcherFn = Callable[..., Dict[str, Any]]


def _resolve_matcher(matcher: str | MatcherFn) -> MatcherFn:
	if callable(matcher):
		return matcher
	matcher_name = matcher.lower()
	if matcher_name == "greedy":
		return greedy_match_detections
	if matcher_name == "hungarian":
		return hungarian_match_detections
	raise ValueError(f"Unsupported matcher: {matcher}")


def evaluate_detection_frames(
	frame_records: Iterable[Dict[str, Any]],
	prediction_records: Iterable[Dict[str, Any]],
	iou_threshold: float = 0.5,
	matcher: str | MatcherFn = "greedy",
	class_aware: bool = True,
	distance_boundaries: Tuple[float, float] = DEFAULT_BUCKET_BOUNDARIES,
	topn_visualizations: int = 0,
	topn_per_scenario: int = 0,
	visualization_dir: str | None = None,
	use_occlusion_bucketing: bool = False,
	use_combined_bucketing: bool = False,
) -> Dict[str, Any]:
	matcher_fn = _resolve_matcher(matcher)
	frame_records_list = list(frame_records)
	prediction_records_list = list(prediction_records)
	if len(frame_records_list) != len(prediction_records_list):
		raise ValueError(
			f"Mismatched frame/prediction counts: {len(frame_records_list)} vs {len(prediction_records_list)}"
		)

	frame_summaries: List[Dict[str, Any]] = []
	frame_fp_breakdowns: List[Dict[str, int]] = []
	frame_details: List[Dict[str, Any]] = []
	all_gt_boxes: List[Dict[str, Any]] = []
	all_pred_boxes: List[Dict[str, Any]] = []
	bucket_accumulator: Dict[str, List[Dict[str, Any]]] = {"near": [], "medium": [], "far": []}
	scene_accumulator: Dict[str, List[Dict[str, Any]]] = {}
	scenario_accumulator: Dict[str, List[Dict[str, Any]]] = {}

	# Initialize occlusion bucket accumulators if needed
	occlusion_bucket_accumulator: Dict[str, List[Dict[str, Any]]] = {}
	occlusion_distance_bucket_accumulator: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
	if use_occlusion_bucketing:
		occlusion_bucket_accumulator = {
			"fully_visible": [],
			"mostly_visible": [],
			"partially_occluded": [],
			"mostly_occluded": [],
			"unknown": [],
		}
	if use_combined_bucketing:
		for occ_level in ["fully_visible", "mostly_visible", "partially_occluded", "mostly_occluded", "unknown"]:
			occlusion_distance_bucket_accumulator[occ_level] = {
				"near": [],
				"medium": [],
				"far": [],
			}

	for frame_index, (frame_record, pred_record) in enumerate(zip(frame_records_list, prediction_records_list), start=1):
		gt_boxes = frame_record.get("gt_boxes", [])
		pred_boxes = pred_record.get("pred_boxes", [])
		scene_name = str(frame_record.get("scene_name", "unknown_scene"))
		scenario_bucket = _infer_scenario_bucket(frame_record)

		summary = summarize_detection_frame(
			gt_boxes=gt_boxes,
			pred_boxes=pred_boxes,
			iou_threshold=iou_threshold,
			class_aware=class_aware,
			matcher_fn=matcher_fn,
		)
		frame_summaries.append(summary)
		scene_accumulator.setdefault(scene_name, []).append(summary)
		scenario_accumulator.setdefault(scenario_bucket, []).append(summary)

		frame_fp_breakdowns.append(compute_fp_breakdown(gt_boxes, summary["match_result"], iou_threshold=iou_threshold))

		for bucket_name, bucket_summary in compute_distance_bucket_metrics(
			gt_boxes=gt_boxes,
			pred_boxes=pred_boxes,
			matcher_fn=matcher_fn,
			iou_threshold=iou_threshold,
			class_aware=class_aware,
			boundaries=distance_boundaries,
		).items():
			bucket_accumulator[bucket_name].append(bucket_summary)

		# Compute occlusion bucket metrics if enabled
		if use_occlusion_bucketing:
			for bucket_name, bucket_summary in compute_occlusion_bucket_metrics(
				gt_boxes=gt_boxes,
				pred_boxes=pred_boxes,
				matcher_fn=matcher_fn,
				iou_threshold=iou_threshold,
				class_aware=class_aware,
				visibility_mapping=VISIBILITY_LEVELS,
			).items():
				occlusion_bucket_accumulator[bucket_name].append(bucket_summary)

		# Compute combined occlusion+distance bucket metrics if enabled
		if use_combined_bucketing:
			for occ_level, dist_buckets in compute_occlusion_distance_bucket_metrics(
				gt_boxes=gt_boxes,
				pred_boxes=pred_boxes,
				matcher_fn=matcher_fn,
				iou_threshold=iou_threshold,
				class_aware=class_aware,
				distance_boundaries=distance_boundaries,
				visibility_mapping=VISIBILITY_LEVELS,
			).items():
				for dist_name, bucket_summary in dist_buckets.items():
					occlusion_distance_bucket_accumulator[occ_level][dist_name].append(bucket_summary)

		all_gt_boxes.extend(gt_boxes)
		all_pred_boxes.extend(pred_boxes)
		frame_details.append(
			{
				"frame_index": frame_index,
				"scene_name": scene_name,
				"scenario_bucket": scenario_bucket,
				"sample_token": frame_record.get("sample_token", ""),
				"summary": summary,
				"gt_boxes": gt_boxes,
				"pred_boxes": pred_boxes,
			}
		)

	overall = aggregate_frame_summaries(frame_summaries)
	fp_breakdown = aggregate_fp_breakdowns(frame_fp_breakdowns)

	class_names = sorted({box.get("category_name") for box in all_gt_boxes if box.get("category_name")})
	per_class_pr = summarize_by_class(
		gt_boxes=all_gt_boxes,
		pred_boxes=all_pred_boxes,
		classes=class_names,
		iou_threshold=iou_threshold,
		matcher_fn=matcher_fn,
	)
	map_result = compute_map(all_gt_boxes, all_pred_boxes, class_names, iou_threshold=iou_threshold)

	bucket_metrics = {
		bucket_name: aggregate_frame_summaries(bucket_summaries)
		for bucket_name, bucket_summaries in bucket_accumulator.items()
	}
	scene_metrics = {
		scene_name: aggregate_frame_summaries(scene_summaries)
		for scene_name, scene_summaries in scene_accumulator.items()
	}
	scenario_metrics = {
		scenario_name: aggregate_frame_summaries(scenario_summaries)
		for scenario_name, scenario_summaries in scenario_accumulator.items()
	}

	localization_errors = [entry["center_distance"] for summary in frame_summaries for entry in summary["match_result"]["matches"]]
	mean_localization_error = (
		sum(localization_errors) / len(localization_errors) if localization_errors else 0.0
	)

	# Aggregate occlusion bucket results
	occlusion_bucket_metrics = {}
	if use_occlusion_bucketing:
		occlusion_bucket_metrics = {
			bucket_name: aggregate_frame_summaries(bucket_summaries)
			for bucket_name, bucket_summaries in occlusion_bucket_accumulator.items()
		}

	occlusion_distance_bucket_metrics = {}
	if use_combined_bucketing:
		occlusion_distance_bucket_metrics = {
			occ_level: {
				dist_name: aggregate_frame_summaries(bucket_summaries)
				for dist_name, bucket_summaries in dist_buckets.items()
			}
			for occ_level, dist_buckets in occlusion_distance_bucket_accumulator.items()
		}

	visualized_topn: List[Dict[str, Any]] = []
	visualized_topn_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
	manifest_path = ""
	if topn_visualizations > 0 and visualization_dir:
		visualized_topn = _save_topn_visualizations(
			frame_details=frame_details,
			topn=topn_visualizations,
			output_dir=visualization_dir,
			prefix="global",
		)
	if topn_per_scenario > 0 and visualization_dir:
		visualized_topn_by_scenario = _save_topn_per_scenario_visualizations(
			frame_details=frame_details,
			topn_per_scenario=topn_per_scenario,
			output_dir=visualization_dir,
		)
	if visualization_dir and (visualized_topn or visualized_topn_by_scenario):
		manifest_path = _write_topn_manifest(
			output_dir=visualization_dir,
			global_items=visualized_topn,
			per_scenario_items=visualized_topn_by_scenario,
		)

	result = {
		"num_frames": len(frame_summaries),
		"iou_threshold": iou_threshold,
		"matcher": matcher if isinstance(matcher, str) else getattr(matcher, "__name__", "custom"),
		"overall": overall,
		"map": map_result,
		"per_class_pr": per_class_pr,
		"distance_buckets": bucket_metrics,
		"distance_bucket_labels": bucket_label_with_ranges(distance_boundaries),
		"per_scene": scene_metrics,
		"per_scenario": scenario_metrics,
		"fp_breakdown": fp_breakdown,
		"mean_localization_error": mean_localization_error,
		"topn_visualizations": visualized_topn,
		"topn_per_scenario": visualized_topn_by_scenario,
		"topn_manifest_path": manifest_path,
	}

	if use_occlusion_bucketing:
		result["occlusion_buckets"] = occlusion_bucket_metrics
		result["occlusion_bucket_labels"] = occlusion_bucket_labels()

	if use_combined_bucketing:
		result["occlusion_distance_buckets"] = occlusion_distance_bucket_metrics
		result["combined_bucket_labels"] = {
			"occlusion": occlusion_bucket_labels(),
			"distance": bucket_label_with_ranges(distance_boundaries),
		}

	return result


def _infer_scenario_bucket(frame_record: Dict[str, Any]) -> str:
	description = str(frame_record.get("scene_description", "")).lower()
	location = str(frame_record.get("location", "")).lower()
	text = f"{description} {location}"

	if any(token in text for token in ("rain", "wet", "fog", "snow")):
		return "adverse_weather"
	if "night" in text:
		return "night"
	if any(token in text for token in ("highway", "freeway", "expressway")):
		return "highway"
	if any(token in location for token in ("singapore", "boston")):
		return "urban"
	return "other"


def _frame_badness_score(frame_detail: Dict[str, Any]) -> float:
	summary = frame_detail["summary"]
	tp = float(summary.get("tp", 0))
	fp = float(summary.get("fp", 0))
	fn = float(summary.get("fn", 0))
	f1 = float(summary.get("f1", 0.0))
	denom = max(1.0, tp + fn)
	return (1.0 - f1) * 2.0 + (fp + fn) / denom


def _save_topn_visualizations(
	frame_details: Sequence[Dict[str, Any]],
	topn: int,
	output_dir: str,
	prefix: str = "global",
) -> List[Dict[str, Any]]:
	if topn <= 0:
		return []

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	sorted_frames = sorted(frame_details, key=_frame_badness_score, reverse=True)
	selected = sorted_frames[: min(topn, len(sorted_frames))]
	visualizations: List[Dict[str, Any]] = []

	for rank, frame_detail in enumerate(selected, start=1):
		summary = frame_detail["summary"]
		filename = (
			f"{prefix}_rank_{rank:02d}_scene_{frame_detail['scene_name']}_"
			f"sample_{frame_detail['sample_token'][:8]}.png"
		)
		save_path = output_path / filename
		title = (
			f"Top{rank} {frame_detail['scene_name']} {frame_detail['sample_token'][:8]} "
			f"P={summary['precision']:.2f} R={summary['recall']:.2f} F1={summary['f1']:.2f}"
		)
		saved_file = save_detection_bev_plot(
			gt_boxes=frame_detail["gt_boxes"],
			pred_boxes=frame_detail["pred_boxes"],
			output_path=str(save_path),
			title=title,
			match_result=summary["match_result"],
		)
		visualizations.append(
			{
				"rank": rank,
				"scene_name": frame_detail["scene_name"],
				"scenario_bucket": frame_detail["scenario_bucket"],
				"sample_token": frame_detail["sample_token"],
				"score": _frame_badness_score(frame_detail),
				"precision": float(summary.get("precision", 0.0)),
				"recall": float(summary.get("recall", 0.0)),
				"f1": float(summary.get("f1", 0.0)),
				"tp": int(summary.get("tp", 0)),
				"fp": int(summary.get("fp", 0)),
				"fn": int(summary.get("fn", 0)),
				"path": saved_file,
			}
		)

	return visualizations


def _save_topn_per_scenario_visualizations(
	frame_details: Sequence[Dict[str, Any]],
	topn_per_scenario: int,
	output_dir: str,
) -> Dict[str, List[Dict[str, Any]]]:
	grouped: Dict[str, List[Dict[str, Any]]] = {}
	for frame_detail in frame_details:
		grouped.setdefault(frame_detail["scenario_bucket"], []).append(frame_detail)

	results: Dict[str, List[Dict[str, Any]]] = {}
	for scenario_name, scenario_frames in grouped.items():
		results[scenario_name] = _save_topn_visualizations(
			frame_details=scenario_frames,
			topn=topn_per_scenario,
			output_dir=output_dir,
			prefix=f"scenario_{scenario_name}",
		)
	return results


def _write_topn_manifest(
	output_dir: str,
	global_items: Sequence[Dict[str, Any]],
	per_scenario_items: Dict[str, Sequence[Dict[str, Any]]],
) -> str:
	manifest = {
		"global_topn": list(global_items),
		"per_scenario_topn": {key: list(value) for key, value in per_scenario_items.items()},
	}
	manifest_path = Path(output_dir) / "topn_manifest.json"
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	return str(manifest_path)


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Perception detection evaluator")
	parser.add_argument("--data-root", default="data/nuscenes-mini", help="Path to nuScenes data root")
	parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
	parser.add_argument("--scene-id", default=None, help="Optional scene name/token")
	parser.add_argument("--max-frames", type=int, default=5, help="Number of frames to evaluate")
	parser.add_argument("--seed", type=int, default=42, help="Generator seed")
	parser.add_argument("--matcher", choices=["greedy", "hungarian"], default="greedy", help="Matching algorithm")
	parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
	parser.add_argument("--topn", type=int, default=0, help="Save top-N worst frame BEV visualizations")
	parser.add_argument(
		"--topn-per-scenario",
		type=int,
		default=0,
		help="Save top-N worst frames for each scenario bucket",
	)
	parser.add_argument(
		"--viz-dir",
		default="outputs/perception/topn",
		help="Directory for saving top-N visualization images",
	)
	return parser


if __name__ == "__main__":
	args = _build_arg_parser().parse_args()

	loader = NuScenesLoader(data_root=args.data_root, version=args.version, verbose=False)
	loader.load()
	scene_id = args.scene_id or loader.get_scene_ids()[0]

	frame_records = list(loader.iter_frame_records(scene_id=scene_id, max_frames=args.max_frames))
	generator = DetectionGenerator(seed=args.seed)
	prediction_records = [generator.generate_frame_predictions(frame_record) for frame_record in frame_records]

	result = evaluate_detection_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=args.iou_threshold,
		matcher=args.matcher,
		topn_visualizations=args.topn,
		topn_per_scenario=args.topn_per_scenario,
		visualization_dir=args.viz_dir,
	)

	overall = result["overall"]
	print(f"[eval] scene={scene_id} frames={result['num_frames']} matcher={result['matcher']}")
	print(
		"[eval] overall: "
		f"P={overall['precision']:.4f} R={overall['recall']:.4f} F1={overall['f1']:.4f} "
		f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}"
	)
	print(f"[eval] mAP@{result['iou_threshold']:.2f}={result['map']['map']:.4f}")
	print(f"[eval] mean localization error={result['mean_localization_error']:.4f} m")
	print(f"[eval] fp breakdown={result['fp_breakdown']}")
	for bucket_name, bucket_summary in result["distance_buckets"].items():
		bucket_label = result["distance_bucket_labels"][bucket_name]
		print(
			f"[eval] bucket {bucket_label}: "
			f"P={bucket_summary['precision']:.4f} R={bucket_summary['recall']:.4f} "
			f"TP={bucket_summary['tp']} FP={bucket_summary['fp']} FN={bucket_summary['fn']}"
		)

	print("[eval] per-scenario metrics:")
	for scenario_name, scenario_summary in result["per_scenario"].items():
		print(
			f"[eval] scenario {scenario_name}: "
			f"P={scenario_summary['precision']:.4f} R={scenario_summary['recall']:.4f} "
			f"TP={scenario_summary['tp']} FP={scenario_summary['fp']} FN={scenario_summary['fn']}"
		)

	if result["topn_visualizations"]:
		print("[eval] top-N visualizations:")
		for item in result["topn_visualizations"]:
			print(
				f"[eval] rank={item['rank']} scene={item['scene_name']} "
				f"sample={item['sample_token'][:8]} path={item['path']}"
			)

	if result["topn_per_scenario"]:
		print("[eval] top-N visualizations per scenario:")
		for scenario_name, items in result["topn_per_scenario"].items():
			for item in items:
				print(
					f"[eval] scenario={scenario_name} rank={item['rank']} "
					f"sample={item['sample_token'][:8]} path={item['path']}"
				)

	if result["topn_manifest_path"]:
		print(f"[eval] top-N manifest: {result['topn_manifest_path']}")

	loader.close()
