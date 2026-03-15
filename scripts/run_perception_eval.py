from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.detection_eval import evaluate_detection_frames
	from eval.perception.tracking_eval import evaluate_tracking_frames
	from generators.detection_generator import DetectionGenerator, DetectionGeneratorConfig
	from metrics.ap_map import compute_map
	from matching.iou_matching import set_bev_iou_mode
	from utils.category_remap import CategoryRemapper
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.detection_eval import evaluate_detection_frames
	from eval.perception.tracking_eval import evaluate_tracking_frames
	from generators.detection_generator import DetectionGenerator, DetectionGeneratorConfig
	from metrics.ap_map import compute_map
	from matching.iou_matching import set_bev_iou_mode
	from utils.category_remap import CategoryRemapper


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run full perception evaluation and export reports")
	parser.add_argument("--dataset-config", default="configs/dataset.yaml", help="Path to dataset.yaml")
	parser.add_argument("--dataset-name", default=None, help="Dataset key in dataset.yaml (e.g. nuscenes_mini)")
	parser.add_argument("--dataset-path", default=None, help="Temporary override for dataset root path")
	parser.add_argument("--data-root", default=None, help="Alias of --dataset-path (backward compatibility)")
	parser.add_argument("--version", default=None, help="Temporary override for dataset version")
	parser.add_argument("--scene-id", default=None, help="Optional scene name/token")
	parser.add_argument(
		"--scenes",
		default=None,
		help="Scene selection mode: first | half | full | comma-separated scene names/tokens",
	)
	# No --scene-id means all scenes in the dataset are evaluated.
	parser.add_argument(
		"--max-frames",
		default=None,
		help="Max frames per selected scene (full|0 means all; e.g. 50)",
	)
	parser.add_argument("--matcher", choices=["greedy", "hungarian"], default=None, help="Matcher (overrides strategy)")
	parser.add_argument("--det-iou-threshold", type=float, default=None, help="Detection IoU threshold (overrides strategy)")
	parser.add_argument("--trk-iou-threshold", type=float, default=None, help="Tracking IoU threshold (overrides strategy)")
	parser.add_argument(
		"--metrics",
		choices=["basic", "standard", "full"],
		default=None,
		help="Metrics level override (basic|standard|full)",
	)
	parser.add_argument(
		"--bev-iou-mode", choices=["aabb", "polygon"], default=None,
		help="BEV IoU mode (overrides strategy): aabb (fast) or polygon (yaw-aware, slower)",
	)
	# Strategy selection — reads eval.yaml and applies category remapping + eval params automatically.
	parser.add_argument(
		"--eval-config", default="configs/eval.yaml",
		help="Path to eval.yaml (default: configs/eval.yaml)",
	)
	parser.add_argument(
		"--strategy", default=None,
		help="Eval strategy name from eval.yaml (raw | detection_10cls | l2_planning). "
		     "Omit to use active_strategy from eval.yaml.",
	)

	# Synthetic prediction controls.
	parser.add_argument("--seed", type=int, default=42, help="Generator random seed")
	parser.add_argument("--drop-rate", type=float, default=0.1, help="GT drop probability")
	parser.add_argument("--fp-rate", type=float, default=0.15, help="False positive rate")
	parser.add_argument("--translation-noise-std", type=float, default=0.35, help="Center noise std")
	parser.add_argument("--size-noise-std", type=float, default=0.08, help="Size noise std")
	parser.add_argument("--yaw-noise-std", type=float, default=0.05, help="Yaw noise std")

	# Visualization and report options.
	parser.add_argument("--topn", type=int, default=3, help="Global top-N badcase visualizations")
	parser.add_argument("--topn-per-scenario", type=int, default=1, help="Top-N per scenario")
	parser.add_argument("--output-dir", default="outputs/perception", help="Root output directory")
	parser.add_argument("--run-name", default=None, help="Optional run name for output folder")
	parser.add_argument(
		"--scene-workers",
		type=int,
		default=1,
		help="Number of worker processes for scene-level parallel evaluation (1 disables parallelism)",
	)
	parser.add_argument(
		"--parallel-modes",
		default="standard,basic",
		help="Comma-separated metrics levels allowed to use scene-level parallelism (default: standard,basic)",
	)

	parser.add_argument(
		"--center-distance-threshold", type=float, default=None,
		help="If set, a GT-pred pair only matches when center distance (m) is within this threshold (in addition to IoU).",
	)

	return parser


def _build_run_paths(output_dir: str, run_name: str | None) -> Dict[str, Path]:
	timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
	resolved_name = run_name or f"perception_eval_{timestamp}"
	run_dir = Path(output_dir) / resolved_name
	viz_dir = run_dir / "topn"
	report_json = run_dir / "report.json"
	report_md = run_dir / "report.md"

	run_dir.mkdir(parents=True, exist_ok=True)
	viz_dir.mkdir(parents=True, exist_ok=True)

	return {
		"run_dir": run_dir,
		"viz_dir": viz_dir,
		"report_json": report_json,
		"report_md": report_md,
	}


def _build_markdown_report(result: Dict[str, Any], config: Dict[str, Any]) -> str:
	detection = result["detection"]
	tracking = result["tracking"]
	det_overall = detection["overall"]
	trk_overall = tracking["overall"]
	map_value = detection["map"]["map"]
	bucketing_status = detection.get("bucketing_status", {})

	lines = [
		"# Perception Evaluation Report",
		"",
		"## Run Config",
		"| Key | Value |",
		"| --- | --- |",
		f"| Dataset | {config['dataset']} |",
		f"| Scene | {config['scene']} |",
		f"| Strategy | {config.get('strategy', '—')} |",
		f"| Category schema | {config.get('category_schema', 'raw')} |",
		f"| Frames | {detection['num_frames']} |",
		f"| Matcher | {detection['matcher']} |",
		f"| Detection IoU threshold | {detection['iou_threshold']} |",
		f"| Tracking IoU threshold | {tracking['iou_threshold']} |",
		f"| BEV IoU mode | {config.get('bev_iou_mode', 'aabb')} |",
		f"| Metrics level | {config.get('metrics_level', 'standard')} |",
		*(
			[f"| Center distance threshold (m) | {config['center_distance_threshold']} |"]
			if config.get("center_distance_threshold") is not None else []
		),
		f"| Seed | {config['seed']} |",
		"",
		"## Detection vs Tracking Summary",
		"| Dimension | Detection | Tracking |",
		"| --- | --- | --- |",
		f"| Matching/Association | {detection['matcher']} @ IoU {detection['iou_threshold']} | {tracking['matcher']} @ IoU {tracking['iou_threshold']} |",
		f"| Core Quality | Precision={det_overall['precision']:.4f}, Recall={det_overall['recall']:.4f}, F1={det_overall['f1']:.4f} | MOTA={trk_overall['mota']:.4f}, MOTP={trk_overall['motp']:.4f}, IDF1={trk_overall.get('idf1', 0.0):.4f} |",
		f"| Error Breakdown | FP={det_overall['fp']}, FN={det_overall['fn']} | FP={trk_overall['fp']}, FN={trk_overall['fn']}, IDSW={trk_overall['idsw']} |",
		f"| Coverage/Stability | mAP={map_value:.4f} | MT={trk_overall['mt']} ({trk_overall['mt_ratio']:.4f}), ML={trk_overall['ml']} ({trk_overall['ml_ratio']:.4f}) |",
		"",
		"## Detection Metrics",
		"| Metric | Value |",
		"| --- | --- |",
		f"| Precision | {det_overall['precision']:.4f} |",
		f"| Recall | {det_overall['recall']:.4f} |",
		f"| F1 | {det_overall['f1']:.4f} |",
		f"| TP | {det_overall['tp']} |",
		f"| FP | {det_overall['fp']} |",
		f"| FN | {det_overall['fn']} |",
		f"| mAP@{detection['iou_threshold']:.2f} | {map_value:.4f} |",
		f"| Mean localization error (m) | {detection['mean_localization_error']:.4f} |",
		f"| FP breakdown | {detection['fp_breakdown']} |",
		"",
		"## Detection Distance Buckets",
	]
	if detection.get("distance_buckets") and detection.get("distance_bucket_labels"):
		bucket_labels = detection["distance_bucket_labels"]
		lines.extend([
			"| Bucket | Precision | Recall | TP | FP | FN |",
			"| --- | ---: | ---: | ---: | ---: | ---: |",
		])
		for bucket_name, metrics in detection["distance_buckets"].items():
			lines.append(
				f"| {bucket_labels[bucket_name]} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
				f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
			)
	else:
		reason = bucketing_status.get("distance", {}).get("reason", "no bucket data")
		lines.append(f"- Skipped: {reason}")

	lines.append("")
	lines.append("## Detection Per Scenario")
	lines.append("| Scenario | Precision | Recall | TP | FP | FN |")
	lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
	for scenario_name, metrics in detection["per_scenario"].items():
		lines.append(
			f"| {scenario_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
			f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
		)

	lines.append("")
	lines.append("## Tracking Metrics")
	lines.append("| Metric | Value |")
	lines.append("| --- | --- |")
	lines.append(f"| MOTA | {trk_overall['mota']:.4f} |")
	lines.append(f"| MOTP (mean center distance, m) | {trk_overall['motp']:.4f} |")
	lines.append(f"| IDF1 | {trk_overall.get('idf1', 0.0):.4f} |")
	lines.append(f"| motmetrics active | {trk_overall.get('motmetrics_available', False)} |")
	lines.append(f"| ID Switches | {trk_overall['idsw']} |")
	lines.append(f"| GT Objects | {trk_overall['gt']} |")
	lines.append(f"| TP | {trk_overall['tp']} |")
	lines.append(f"| FP | {trk_overall['fp']} |")
	lines.append(f"| FN | {trk_overall['fn']} |")
	lines.append(f"| Mostly Tracked (MT) | {trk_overall['mt']} ({trk_overall['mt_ratio']:.4f}) |")
	lines.append(f"| Mostly Lost (ML) | {trk_overall['ml']} ({trk_overall['ml_ratio']:.4f}) |")

	lines.append("")
	lines.append("## Tracking Per Scenario")
	lines.append("| Scenario | MOTA | GT | TP | FP | FN | IDSW |")
	lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
	for scenario_name, metrics in tracking["per_scenario"].items():
		lines.append(
			f"| {scenario_name} | {metrics['mota']:.4f} | {metrics['gt']} | {metrics['tp']} | "
			f"{metrics['fp']} | {metrics['fn']} | {metrics['idsw']} |"
		)

	if detection["topn_visualizations"]:
		lines.append("")
		lines.append("## Detection Top-N Visualizations")
		lines.append("| Rank | Scene | Sample | Precision | Recall | F1 | Path |")
		lines.append("| ---: | --- | --- | ---: | ---: | ---: | --- |")
		for item in detection["topn_visualizations"]:
			lines.append(
				f"| {item['rank']} | {item['scene_name']} | {item['sample_token'][:8]} | "
				f"{item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | {item['path']} |"
			)

	if detection.get("topn_manifest_path"):
		lines.append("")
		lines.append(f"- Top-N manifest: {detection['topn_manifest_path']}")

	lines.append("")
	lines.append("## Per-Class Detection vs Tracking Alignment")
	lines.append("| Class | Det AP | Det P | Det R | Det F1 | Trk MOTA | Trk MOTP | Trk TP | Trk FP | Trk FN | Trk IDSW |")
	lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

	detection_per_class_map = detection.get("map", {}).get("per_class", {})
	detection_per_class_pr = detection.get("per_class_pr", {})
	tracking_per_class = tracking.get("per_class", {})
	all_classes = sorted(
		set(detection_per_class_map.keys()) | set(detection_per_class_pr.keys()) | set(tracking_per_class.keys())
	)
	for class_name in all_classes:
		det_ap = float(detection_per_class_map.get(class_name, {}).get("ap", 0.0))
		det_p = float(detection_per_class_pr.get(class_name, {}).get("precision", 0.0))
		det_r = float(detection_per_class_pr.get(class_name, {}).get("recall", 0.0))
		det_f1 = float(detection_per_class_pr.get(class_name, {}).get("f1", 0.0))
		trk_metrics = tracking_per_class.get(class_name, {})
		trk_mota = float(trk_metrics.get("mota", 0.0))
		trk_motp = float(trk_metrics.get("motp", 0.0))
		trk_tp = int(trk_metrics.get("tp", 0))
		trk_fp = int(trk_metrics.get("fp", 0))
		trk_fn = int(trk_metrics.get("fn", 0))
		trk_idsw = int(trk_metrics.get("idsw", 0))
		lines.append(
			f"| {class_name} | {det_ap:.4f} | {det_p:.4f} | {det_r:.4f} | {det_f1:.4f} | "
			f"{trk_mota:.4f} | {trk_motp:.4f} | {trk_tp} | {trk_fp} | {trk_fn} | {trk_idsw} |"
		)

	# --- Occlusion (ODD) Buckets -----------------------------------------------
	if detection.get("occlusion_buckets"):
		occ_labels = detection.get("occlusion_bucket_labels", {})
		lines.append("")
		lines.append("## Detection Occlusion (ODD) Buckets")
		lines.append("| Bucket | Precision | Recall | TP | FP | FN |")
		lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
		for bucket_name, metrics in detection["occlusion_buckets"].items():
			label = occ_labels.get(bucket_name, bucket_name)
			lines.append(
				f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
				f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
			)
	else:
		lines.append("")
		lines.append("## Detection Occlusion (ODD) Buckets")
		reason = bucketing_status.get("occlusion", {}).get("reason", "no bucket data")
		lines.append(f"- Skipped: {reason}")

	# --- Combined Occlusion × Distance Buckets ---------------------------------
	if detection.get("occlusion_distance_buckets"):
		combined_labels = detection.get("combined_bucket_labels", {})
		occ_labels = combined_labels.get("occlusion", {})
		dist_labels = combined_labels.get("distance", {})
		lines.append("")
		lines.append("## Detection Combined Occlusion × Distance Buckets")
		lines.append("| Occlusion | Distance | Precision | Recall | TP | FP | FN |")
		lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
		for occ_level, dist_buckets in detection["occlusion_distance_buckets"].items():
			occ_label = occ_labels.get(occ_level, occ_level)
			for dist_name, metrics in dist_buckets.items():
				dist_label = dist_labels.get(dist_name, dist_name)
				lines.append(
					f"| {occ_label} | {dist_label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
					f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
				)
	else:
		lines.append("")
		lines.append("## Detection Combined Occlusion × Distance Buckets")
		reason = bucketing_status.get("combined", {}).get("reason", "no bucket data")
		lines.append(f"- Skipped: {reason}")

	lines.append("")
	lines.append("## Notes")
	lines.append("- Detection and tracking are evaluated from the same generated prediction stream.")
	lines.append("- Tracking metrics here are MOTA/MOTP/IDSW/MT/ML based on IoU matching and track_id continuity.")

	return "\n".join(lines) + "\n"


def _to_json_compatible(value: Any) -> Any:
	if isinstance(value, dict):
		return {str(key): _to_json_compatible(item) for key, item in value.items()}
	if isinstance(value, list):
		return [_to_json_compatible(item) for item in value]
	if isinstance(value, tuple):
		return [_to_json_compatible(item) for item in value]

	# numpy scalar compatibility without importing numpy directly.
	if hasattr(value, "item") and callable(getattr(value, "item")):
		try:
			return _to_json_compatible(value.item())
		except Exception:
			pass

	return value


def _load_yaml(path: str) -> Dict[str, Any]:
	try:
		import yaml
	except ImportError as exc:
		raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

	config_path = Path(path)
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with config_path.open(encoding="utf-8") as fh:
		return yaml.safe_load(fh) or {}


def _resolve_dataset_entry(dataset_cfg: Dict[str, Any], dataset_name: str | None) -> Tuple[str, Dict[str, Any]]:
	"""Resolve dataset entry from dataset.yaml supporting both new and legacy layouts."""
	datasets = dataset_cfg.get("datasets")
	if isinstance(datasets, dict) and datasets:
		resolved_name = dataset_name or dataset_cfg.get("active_dataset") or next(iter(datasets.keys()))
		if resolved_name not in datasets:
			raise ValueError(f"Unknown dataset_name {resolved_name!r}. Available: {list(datasets.keys())}")
		return resolved_name, dict(datasets[resolved_name])

	legacy_data = dataset_cfg.get("data")
	if isinstance(legacy_data, dict) and legacy_data:
		return dataset_name or "default", dict(legacy_data)

	raise ValueError("dataset.yaml must define either 'datasets' or legacy 'data'")


def _resolve_strategy_params(eval_cfg: Dict[str, Any], strategy_name: str | None) -> Tuple[str, Dict[str, Any]]:
	strategies = eval_cfg.get("strategies", {})
	if not strategies:
		raise ValueError("eval.yaml has no 'strategies' block")

	resolved_strategy = strategy_name or eval_cfg.get("active_strategy") or next(iter(strategies.keys()))
	if resolved_strategy not in strategies:
		raise ValueError(f"Unknown strategy {resolved_strategy!r}. Available: {list(strategies.keys())}")

	params = dict(eval_cfg.get("defaults", {}))
	params.update(strategies[resolved_strategy] or {})
	return resolved_strategy, params


def _parse_max_frames(raw_value: Any) -> int | None:
	if raw_value is None:
		return None
	text = str(raw_value).strip().lower()
	if text in {"", "full", "all", "0", "none", "null"}:
		return None
	try:
		value = int(text)
	except ValueError as exc:
		raise ValueError(f"Invalid max_frames value: {raw_value!r}") from exc
	if value <= 0:
		return None
	return value


def _resolve_scene_ids(all_scene_ids: List[str], scenes_arg: str | None, explicit_scene: str | None) -> List[str]:
	if explicit_scene:
		return [explicit_scene]

	mode = (scenes_arg or "full").strip()
	mode_lower = mode.lower()
	if mode_lower in {"full", "all"}:
		return list(all_scene_ids)
	if mode_lower == "first":
		return all_scene_ids[:1]
	if mode_lower == "half":
		half_count = max(1, len(all_scene_ids) // 2)
		return all_scene_ids[:half_count]

	requested = [item.strip() for item in mode.split(",") if item.strip()]
	if not requested:
		return list(all_scene_ids)

	available = set(all_scene_ids)
	missing = [scene for scene in requested if scene not in available]
	if missing:
		raise ValueError(f"Unknown scenes requested: {missing}. Available sample: {all_scene_ids[:5]}")
	return requested


def _parse_parallel_modes(raw_value: str | None) -> Set[str]:
	if raw_value is None:
		return {"standard", "basic"}
	parsed = {part.strip().lower() for part in str(raw_value).split(",") if part.strip()}
	if not parsed:
		return {"standard", "basic"}
	allowed = {"basic", "standard", "full"}
	unknown = sorted(parsed - allowed)
	if unknown:
		raise ValueError(f"Unknown --parallel-modes values: {unknown}. Allowed: {sorted(allowed)}")
	return parsed


def _compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
	precision = tp / (tp + fp) if tp + fp > 0 else 0.0
	recall = tp / (tp + fn) if tp + fn > 0 else 0.0
	f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
	return {"precision": precision, "recall": recall, "f1": f1}


def _merge_metric_summaries(items: List[Dict[str, Any]]) -> Dict[str, Any]:
	tp = sum(int(item.get("tp", 0)) for item in items)
	fp = sum(int(item.get("fp", 0)) for item in items)
	fn = sum(int(item.get("fn", 0)) for item in items)
	metrics = _compute_precision_recall_f1(tp, fp, fn)
	return {"tp": tp, "fp": fp, "fn": fn, **metrics, "num_frames": len(items)}


def _merge_named_metric_summaries(named_items: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
	buckets: Dict[str, List[Dict[str, Any]]] = {}
	for item in named_items:
		for name, metrics in item.items():
			buckets.setdefault(name, []).append(metrics)
	return {name: _merge_metric_summaries(values) for name, values in buckets.items()}


def _merge_detection_scene_results(
	scene_results: List[Dict[str, Any]],
	iou_threshold: float,
	matcher: str,
) -> Dict[str, Any]:
	if not scene_results:
		return {
			"num_frames": 0,
			"iou_threshold": iou_threshold,
			"matcher": matcher,
			"overall": {"tp": 0, "fp": 0, "fn": 0, **_compute_precision_recall_f1(0, 0, 0), "num_frames": 0},
			"map": {"iou_threshold": iou_threshold, "classes": [], "per_class": {}, "map": 0.0},
			"per_class_pr": {},
			"per_scene": {},
			"per_scenario": {},
			"fp_breakdown": {"localization": 0, "classification": 0, "duplicate": 0, "background": 0},
			"mean_localization_error": 0.0,
			"topn_visualizations": [],
			"topn_per_scenario": {},
			"topn_manifest_path": "",
			"bucketing_status": {
				"distance": {"requested": True, "enabled": False, "reason": "no data"},
				"occlusion": {"enabled": False, "reason": "no data"},
				"combined": {"enabled": False, "reason": "no data"},
			},
		}

	all_gt_boxes: List[Dict[str, Any]] = []
	all_pred_boxes: List[Dict[str, Any]] = []
	per_scene: Dict[str, Dict[str, Any]] = {}
	per_scenario_inputs: List[Dict[str, Dict[str, Any]]] = []
	distance_bucket_inputs: List[Dict[str, Dict[str, Any]]] = []
	occlusion_bucket_inputs: List[Dict[str, Dict[str, Any]]] = []
	occlusion_distance_inputs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
	per_class_counts: Dict[str, Dict[str, int]] = {}
	fp_breakdown = {"localization": 0, "classification": 0, "duplicate": 0, "background": 0}
	mean_error_sum = 0.0
	total_tp = 0
	total_frames = 0
	bucketing_status = scene_results[0]["detection"].get("bucketing_status", {})

	for scene_result in scene_results:
		detection = scene_result["detection"]
		all_gt_boxes.extend(scene_result.get("all_gt_boxes", []))
		all_pred_boxes.extend(scene_result.get("all_pred_boxes", []))
		total_frames += int(detection.get("num_frames", 0))

		for name, metrics in detection.get("per_scene", {}).items():
			per_scene[name] = metrics

		per_scenario_inputs.append(detection.get("per_scenario", {}))
		if detection.get("distance_buckets"):
			distance_bucket_inputs.append(detection["distance_buckets"])
		if detection.get("occlusion_buckets"):
			occlusion_bucket_inputs.append(detection["occlusion_buckets"])
		if detection.get("occlusion_distance_buckets"):
			occlusion_distance_inputs.append(detection["occlusion_distance_buckets"])

		for bucket_name in fp_breakdown:
			fp_breakdown[bucket_name] += int(detection.get("fp_breakdown", {}).get(bucket_name, 0))

		overall = detection.get("overall", {})
		tp = int(overall.get("tp", 0))
		total_tp += tp
		mean_error_sum += float(detection.get("mean_localization_error", 0.0)) * tp

		for class_name, metrics in detection.get("per_class_pr", {}).items():
			counts = per_class_counts.setdefault(class_name, {"tp": 0, "fp": 0, "fn": 0})
			counts["tp"] += int(metrics.get("tp", 0))
			counts["fp"] += int(metrics.get("fp", 0))
			counts["fn"] += int(metrics.get("fn", 0))

	if isinstance(bucketing_status, dict):
		for dim in ("distance", "occlusion", "combined"):
			enabled_all = all(
				bool(scene_result["detection"].get("bucketing_status", {}).get(dim, {}).get("enabled", False))
				for scene_result in scene_results
			)
			reason = "ok" if enabled_all else "partial_or_missing_fields"
			if dim not in bucketing_status:
				bucketing_status[dim] = {}
			bucketing_status[dim]["enabled"] = enabled_all
			bucketing_status[dim]["reason"] = reason
			if dim == "distance":
				bucketing_status[dim]["requested"] = True

	overall_metrics = _merge_metric_summaries([scene_result["detection"].get("overall", {}) for scene_result in scene_results])
	overall_metrics["num_frames"] = total_frames

	class_names = sorted({str(box.get("category_name")) for box in all_gt_boxes if box.get("category_name")})
	map_result = compute_map(all_gt_boxes, all_pred_boxes, class_names, iou_threshold=iou_threshold)

	per_class_pr = {}
	for class_name, counts in per_class_counts.items():
		metrics = _compute_precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
		per_class_pr[class_name] = {
			"tp": counts["tp"],
			"fp": counts["fp"],
			"fn": counts["fn"],
			**metrics,
			"num_frames": total_frames,
		}

	result = {
		"num_frames": total_frames,
		"iou_threshold": iou_threshold,
		"matcher": matcher,
		"overall": overall_metrics,
		"map": map_result,
		"per_class_pr": per_class_pr,
		"per_scene": per_scene,
		"per_scenario": _merge_named_metric_summaries(per_scenario_inputs),
		"fp_breakdown": fp_breakdown,
		"mean_localization_error": mean_error_sum / max(1, total_tp),
		"topn_visualizations": [],
		"topn_per_scenario": {},
		"topn_manifest_path": "",
		"bucketing_status": bucketing_status,
	}

	if distance_bucket_inputs:
		result["distance_buckets"] = _merge_named_metric_summaries(distance_bucket_inputs)
		result["distance_bucket_labels"] = scene_results[0]["detection"].get("distance_bucket_labels", {})

	if occlusion_bucket_inputs:
		result["occlusion_buckets"] = _merge_named_metric_summaries(occlusion_bucket_inputs)
		result["occlusion_bucket_labels"] = scene_results[0]["detection"].get("occlusion_bucket_labels", {})

	if occlusion_distance_inputs:
		merged_combined: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
		for combined_item in occlusion_distance_inputs:
			for occ_level, dist_dict in combined_item.items():
				occ_container = merged_combined.setdefault(occ_level, {})
				for dist_name, metrics in dist_dict.items():
					occ_container.setdefault(dist_name, []).append(metrics)
		result["occlusion_distance_buckets"] = {
			occ_level: {
				dist_name: _merge_metric_summaries(items)
				for dist_name, items in dist_dict.items()
			}
			for occ_level, dist_dict in merged_combined.items()
		}
		result["combined_bucket_labels"] = scene_results[0]["detection"].get("combined_bucket_labels", {})

	return result


def _merge_tracking_scene_results(
	scene_results: List[Dict[str, Any]],
	iou_threshold: float,
	matcher: str,
	metrics_level: str,
) -> Dict[str, Any]:
	total_frames = sum(int(scene_result["tracking"].get("num_frames", 0)) for scene_result in scene_results)
	gt = tp = fp = fn = idsw = mt = ml = num_tracks = 0
	motp_weighted_sum = 0.0
	per_scene: Dict[str, Dict[str, Any]] = {}
	per_scenario_raw: Dict[str, Dict[str, float]] = {}
	per_class_raw: Dict[str, Dict[str, float]] = {}

	for scene_result in scene_results:
		tracking = scene_result["tracking"]
		overall = tracking.get("overall", {})
		gt += int(overall.get("gt", 0))
		tp += int(overall.get("tp", 0))
		fp += int(overall.get("fp", 0))
		fn += int(overall.get("fn", 0))
		idsw += int(overall.get("idsw", 0))
		mt += int(overall.get("mt", 0))
		ml += int(overall.get("ml", 0))
		num_tracks += int(overall.get("num_tracks", 0))
		motp_weighted_sum += float(overall.get("motp", 0.0)) * int(overall.get("tp", 0))

		for name, metrics in tracking.get("per_scene", {}).items():
			per_scene[name] = metrics

		for scenario_name, metrics in tracking.get("per_scenario", {}).items():
			counts = per_scenario_raw.setdefault(
				scenario_name,
				{"gt": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "idsw": 0.0},
			)
			counts["gt"] += float(metrics.get("gt", 0))
			counts["tp"] += float(metrics.get("tp", 0))
			counts["fp"] += float(metrics.get("fp", 0))
			counts["fn"] += float(metrics.get("fn", 0))
			counts["idsw"] += float(metrics.get("idsw", 0))

		for class_name, metrics in tracking.get("per_class", {}).items():
			counts = per_class_raw.setdefault(
				class_name,
				{"gt": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "idsw": 0.0, "motp_weighted": 0.0},
			)
			class_tp = float(metrics.get("tp", 0))
			counts["gt"] += float(metrics.get("gt", 0))
			counts["tp"] += class_tp
			counts["fp"] += float(metrics.get("fp", 0))
			counts["fn"] += float(metrics.get("fn", 0))
			counts["idsw"] += float(metrics.get("idsw", 0))
			counts["motp_weighted"] += float(metrics.get("motp", 0.0)) * class_tp

	mota = 1.0 - (fn + fp + idsw) / max(1, gt)
	motp = motp_weighted_sum / max(1, tp)
	per_scenario = {}
	for scenario_name, counts in per_scenario_raw.items():
		scenario_gt = int(counts["gt"])
		scenario_tp = int(counts["tp"])
		scenario_fp = int(counts["fp"])
		scenario_fn = int(counts["fn"])
		scenario_idsw = int(counts["idsw"])
		scenario_mota = 1.0 - (scenario_fn + scenario_fp + scenario_idsw) / max(1, scenario_gt)
		per_scenario[scenario_name] = {
			"gt": scenario_gt,
			"tp": scenario_tp,
			"fp": scenario_fp,
			"fn": scenario_fn,
			"idsw": scenario_idsw,
			"mota": scenario_mota,
		}

	per_class = {}
	for class_name, counts in per_class_raw.items():
		class_gt = int(counts["gt"])
		class_tp = int(counts["tp"])
		class_fp = int(counts["fp"])
		class_fn = int(counts["fn"])
		class_idsw = int(counts["idsw"])
		class_mota = 1.0 - (class_fn + class_fp + class_idsw) / max(1, class_gt)
		class_motp = float(counts["motp_weighted"]) / max(1, class_tp)
		per_class[class_name] = {
			"gt": class_gt,
			"tp": class_tp,
			"fp": class_fp,
			"fn": class_fn,
			"idsw": class_idsw,
			"mota": class_mota,
			"motp": class_motp,
		}

	return {
		"num_frames": total_frames,
		"matcher": matcher,
		"iou_threshold": iou_threshold,
		"metrics_level": metrics_level,
		"overall": {
			"gt": gt,
			"tp": tp,
			"fp": fp,
			"fn": fn,
			"idsw": idsw,
			"mota": mota,
			"motp": motp,
			"idf1": 0.0,
			"motmetrics_available": False,
			"num_tracks": num_tracks,
			"mt": mt,
			"ml": ml,
			"mt_ratio": mt / max(1, num_tracks),
			"ml_ratio": ml / max(1, num_tracks),
		},
		"per_scene": per_scene,
		"per_scenario": per_scenario,
		"per_class": per_class,
	}


def _evaluate_scene_chunk_worker(
	frame_records: List[Dict[str, Any]],
	prediction_records: List[Dict[str, Any]],
	det_iou_threshold: float,
	trk_iou_threshold: float,
	matcher: str,
	metrics_level: str,
	center_distance_threshold: float | None,
) -> Dict[str, Any]:
	detection = evaluate_detection_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=det_iou_threshold,
		matcher=matcher,
		topn_visualizations=0,
		topn_per_scenario=0,
		visualization_dir=None,
		center_distance_threshold=center_distance_threshold,
	)
	tracking = evaluate_tracking_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=trk_iou_threshold,
		matcher=matcher,
		metrics_level=metrics_level,
		center_distance_threshold=center_distance_threshold,
	)
	return {
		"detection": detection,
		"tracking": tracking,
		"all_gt_boxes": [box for frame in frame_records for box in frame.get("gt_boxes", [])],
		"all_pred_boxes": [box for frame in prediction_records for box in frame.get("pred_boxes", [])],
	}


def main() -> int:
	args = _build_arg_parser().parse_args()
	paths = _build_run_paths(args.output_dir, args.run_name)

	# 1) Load YAML configs.
	dataset_cfg = _load_yaml(args.dataset_config)
	eval_cfg = _load_yaml(args.eval_config)

	# 2) Resolve dataset and strategy defaults.
	dataset_name, dataset_entry = _resolve_dataset_entry(dataset_cfg, args.dataset_name)
	strategy_name, strategy_params = _resolve_strategy_params(eval_cfg, args.strategy)

	# 3) CLI overrides (temporary experiments).
	resolved_data_root = args.dataset_path or args.data_root or dataset_entry.get("root", "data/nuscenes-mini")
	resolved_version = args.version or dataset_entry.get("version", "v1.0-mini")
	resolved_matcher = args.matcher or strategy_params.get("matcher", "greedy")
	resolved_det_iou = args.det_iou_threshold or strategy_params.get("iou_threshold", 0.5)
	resolved_trk_iou = args.trk_iou_threshold or strategy_params.get("iou_threshold", 0.5)
	resolved_bev_iou_mode = args.bev_iou_mode or strategy_params.get("bev_iou_mode", "aabb")
	resolved_center_distance = (
		args.center_distance_threshold
		if args.center_distance_threshold is not None
		else strategy_params.get("center_distance_threshold")
	)
	resolved_metrics_level = args.metrics or strategy_params.get("metrics_level", "standard")

	configured_max_frames = strategy_params.get("max_frames")
	cli_max_frames = _parse_max_frames(args.max_frames) if args.max_frames is not None else None
	strategy_max_frames = _parse_max_frames(configured_max_frames)
	resolved_max_frames = cli_max_frames if args.max_frames is not None else strategy_max_frames
	resolved_parallel_modes = _parse_parallel_modes(args.parallel_modes)

	# 4) Build category remapper from resolved schema.
	schema_name = strategy_params.get("category_schema") or dataset_entry.get("category_schema", "raw")
	remapper = CategoryRemapper.from_config(args.dataset_config, schema=schema_name)
	print(
		f"[run] dataset={dataset_name!r} root={resolved_data_root} version={resolved_version} "
		f"strategy={strategy_name!r} schema={schema_name} target_classes={remapper.target_classes or '(raw)'}"
	)

	loader = NuScenesLoader(data_root=resolved_data_root, version=resolved_version, verbose=False)
	set_bev_iou_mode(resolved_bev_iou_mode)
	loader.load()
	all_scene_ids = loader.get_scene_ids()
	selected_scene_ids = _resolve_scene_ids(all_scene_ids, args.scenes, args.scene_id)

	raw_frame_records: List[Dict[str, Any]] = []
	parallel_enabled = False
	if resolved_metrics_level == "full":
		print("[run] scene parallelism disabled for metrics=full (tracking IDF1 flow stays serial).")
	elif args.scene_workers <= 1:
		print("[run] scene parallelism disabled: --scene-workers <= 1")
	elif len(selected_scene_ids) <= 1:
		print("[run] scene parallelism disabled: only one selected scene")
	elif resolved_metrics_level not in resolved_parallel_modes:
		print(
			f"[run] scene parallelism disabled: metrics={resolved_metrics_level} "
			f"not in --parallel-modes={sorted(resolved_parallel_modes)}"
		)
	else:
		parallel_enabled = True

	for scene in selected_scene_ids:
		raw_frame_records.extend(loader.iter_frame_records(scene_id=scene, max_frames=resolved_max_frames))

	covered_scenes = sorted({str(frame.get("scene_name", "unknown_scene")) for frame in raw_frame_records})
	print(f"[run] loaded frames={len(raw_frame_records)}  scenes={len(covered_scenes)}  sample={covered_scenes[:3]}")

	generator_config = DetectionGeneratorConfig(
		translation_noise_std=args.translation_noise_std,
		size_noise_std=args.size_noise_std,
		yaw_noise_std=args.yaw_noise_std,
		drop_rate=args.drop_rate,
		fp_rate=args.fp_rate,
	)
	generator = DetectionGenerator(config=generator_config, seed=args.seed)
	raw_prediction_records = [generator.generate_frame_predictions(f) for f in raw_frame_records]

	# ── Apply category remapping (no-op when schema is "raw") ──────────────────
	if remapper is not None:
		frame_records = list(remapper.apply_to_frames(raw_frame_records))
		prediction_records = list(remapper.apply_to_predictions(raw_prediction_records))
	else:
		frame_records = raw_frame_records
		prediction_records = raw_prediction_records

	if parallel_enabled:
		print(f"[run] scene-level parallel evaluation enabled: workers={args.scene_workers} scenes={len(selected_scene_ids)}")
		scene_chunks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
		for frame_record, pred_record in zip(frame_records, prediction_records):
			scene_name = str(frame_record.get("scene_name", "unknown_scene"))
			chunk = scene_chunks.setdefault(scene_name, {"frames": [], "preds": []})
			chunk["frames"].append(frame_record)
			chunk["preds"].append(pred_record)

		ordered_scene_names = [scene for scene in selected_scene_ids if scene in scene_chunks]
		ordered_scene_names.extend([name for name in scene_chunks.keys() if name not in ordered_scene_names])

		scene_eval_results: Dict[str, Dict[str, Any]] = {}
		with ProcessPoolExecutor(max_workers=args.scene_workers) as executor:
			future_map = {
				executor.submit(
					_evaluate_scene_chunk_worker,
					scene_chunks[scene_name]["frames"],
					scene_chunks[scene_name]["preds"],
					resolved_det_iou,
					resolved_trk_iou,
					resolved_matcher,
					resolved_metrics_level,
					resolved_center_distance,
				): scene_name
				for scene_name in ordered_scene_names
			}
			for future in as_completed(future_map):
				scene_name = future_map[future]
				scene_eval_results[scene_name] = future.result()

		merged_inputs = [scene_eval_results[scene_name] for scene_name in ordered_scene_names]
		result = _merge_detection_scene_results(merged_inputs, iou_threshold=resolved_det_iou, matcher=resolved_matcher)
		tracking_result = _merge_tracking_scene_results(
			merged_inputs,
			iou_threshold=resolved_trk_iou,
			matcher=resolved_matcher,
			metrics_level=resolved_metrics_level,
		)
	else:
		result = evaluate_detection_frames(
			frame_records=frame_records,
			prediction_records=prediction_records,
			iou_threshold=resolved_det_iou,
			matcher=resolved_matcher,
			topn_visualizations=args.topn if resolved_metrics_level == "full" else 0,
			topn_per_scenario=args.topn_per_scenario if resolved_metrics_level == "full" else 0,
			visualization_dir=str(paths["viz_dir"]),
			center_distance_threshold=resolved_center_distance,
		)
		tracking_result = evaluate_tracking_frames(
			frame_records=frame_records,
			prediction_records=prediction_records,
			iou_threshold=resolved_trk_iou,
			matcher=resolved_matcher,
			metrics_level=resolved_metrics_level,
			center_distance_threshold=resolved_center_distance,
		)

	full_result = {
		"detection": result,
		"tracking": tracking_result,
	}
	full_result["config"] = {
		"dataset_name": dataset_name,
		"dataset": resolved_version,
		"dataset_root": resolved_data_root,
		"scene": args.scene_id or args.scenes or "full",
		"max_frames_per_scene": resolved_max_frames,
		"selected_scenes": selected_scene_ids,
		"metrics_level": resolved_metrics_level,
		"parallel": {
			"enabled": parallel_enabled,
			"scene_workers": args.scene_workers,
			"parallel_modes": sorted(resolved_parallel_modes),
		},
		"strategy": strategy_name,
		"category_schema": schema_name,
		"matcher": resolved_matcher,
		"det_iou_threshold": resolved_det_iou,
		"trk_iou_threshold": resolved_trk_iou,
		"bev_iou_mode": resolved_bev_iou_mode,
		"center_distance_threshold": resolved_center_distance,
		"seed": args.seed,
		"generator": {
			"translation_noise_std": args.translation_noise_std,
			"size_noise_std": args.size_noise_std,
			"yaw_noise_std": args.yaw_noise_std,
			"drop_rate": args.drop_rate,
			"fp_rate": args.fp_rate,
		},
	}

	json_safe_result = _to_json_compatible(full_result)
	paths["report_json"].write_text(json.dumps(json_safe_result, indent=2), encoding="utf-8")
	markdown = _build_markdown_report(
		result=full_result,
		config={
			"dataset": f"{dataset_name} ({resolved_version})",
			"scene": args.scene_id or args.scenes or "full",
			"bev_iou_mode": resolved_bev_iou_mode,
			"metrics_level": resolved_metrics_level,
			"center_distance_threshold": resolved_center_distance,
			"seed": args.seed,
			"strategy": strategy_name,
			"category_schema": schema_name,
		},
	)
	paths["report_md"].write_text(markdown, encoding="utf-8")

	det_overall = result["overall"]
	trk_overall = tracking_result["overall"]
	print(
		f"[run] scenes={len(selected_scene_ids)} mode={args.scene_id or args.scenes or 'full'} "
		f"frames={result['num_frames']} matcher={result['matcher']} metrics={resolved_metrics_level}"
	)
	print(
		"[run][detection] "
		f"P={det_overall['precision']:.4f} R={det_overall['recall']:.4f} F1={det_overall['f1']:.4f} "
		f"TP={det_overall['tp']} FP={det_overall['fp']} FN={det_overall['fn']}"
	)
	print(f"[run][detection] mAP@{result['iou_threshold']:.2f}={result['map']['map']:.4f}")
	print(
		"[run][tracking] "
		f"MOTA={trk_overall['mota']:.4f} MOTP={trk_overall['motp']:.4f} IDF1={trk_overall.get('idf1', 0.0):.4f} "
		f"IDSW={trk_overall['idsw']} MT={trk_overall['mt']} ML={trk_overall['ml']}"
	)
	print(f"[run] report(json): {paths['report_json']}")
	print(f"[run] report(md): {paths['report_md']}")
	if result.get("topn_manifest_path"):
		print(f"[run] topn manifest: {result['topn_manifest_path']}")

	loader.close()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
