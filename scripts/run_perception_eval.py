from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.detection_eval import evaluate_detection_frames
	from eval.perception.tracking_eval import evaluate_tracking_frames
	from generators.detection_generator import DetectionGenerator, DetectionGeneratorConfig
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

	result = evaluate_detection_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=resolved_det_iou,
		matcher=resolved_matcher,
		topn_visualizations=0 if resolved_metrics_level == "basic" else args.topn,
		topn_per_scenario=0 if resolved_metrics_level == "basic" else args.topn_per_scenario,
		visualization_dir=str(paths["viz_dir"]),
		center_distance_threshold=resolved_center_distance,
	)
	tracking_result = evaluate_tracking_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=resolved_trk_iou,
		matcher=resolved_matcher,
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
