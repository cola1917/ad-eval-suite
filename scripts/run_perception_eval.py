from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

try:
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.detection_eval import evaluate_detection_frames
	from eval.perception.tracking_eval import evaluate_tracking_frames
	from generators.detection_generator import DetectionGenerator, DetectionGeneratorConfig
	from utils.category_remap import build_remapper_from_eval_config
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.detection_eval import evaluate_detection_frames
	from eval.perception.tracking_eval import evaluate_tracking_frames
	from generators.detection_generator import DetectionGenerator, DetectionGeneratorConfig
	from utils.category_remap import build_remapper_from_eval_config


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run full perception evaluation and export reports")
	parser.add_argument("--data-root", default="data/nuscenes-mini", help="Path to nuScenes data root")
	parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
	parser.add_argument("--scene-id", default=None, help="Optional scene name/token")
	parser.add_argument("--max-frames", type=int, default=10, help="Number of frames to evaluate")
	parser.add_argument("--matcher", choices=["greedy", "hungarian"], default=None, help="Matcher (overrides strategy)")
	parser.add_argument("--det-iou-threshold", type=float, default=None, help="Detection IoU threshold (overrides strategy)")
	parser.add_argument("--trk-iou-threshold", type=float, default=None, help="Tracking IoU threshold (overrides strategy)")
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

	# ODD / occlusion bucketing options.
	parser.add_argument("--occlusion-bucketing", action="store_true", default=False,
	                    help="Add occlusion-level (ODD) breakdown to report")
	parser.add_argument("--combined-bucketing", action="store_true", default=False,
	                    help="Add combined occlusion × distance breakdown to report")
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
	bucket_labels = detection["distance_bucket_labels"]

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
		"| Bucket | Precision | Recall | TP | FP | FN |",
		"| --- | ---: | ---: | ---: | ---: | ---: |",
	]

	for bucket_name, metrics in detection["distance_buckets"].items():
		lines.append(
			f"| {bucket_labels[bucket_name]} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
			f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
		)

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


def _load_strategy_params(eval_config: str, strategy: str | None) -> Dict[str, Any]:
	"""Read eval.yaml and return the resolved strategy parameters."""
	try:
		import yaml
	except ImportError:
		return {}
	config_path = Path(eval_config)
	if not config_path.exists():
		return {}
	with config_path.open(encoding="utf-8") as fh:
		eval_cfg = yaml.safe_load(fh) or {}
	resolved_strategy = strategy or eval_cfg.get("active_strategy", "raw")
	strategies = eval_cfg.get("strategies", {})
	if resolved_strategy not in strategies:
		print(f"[run] warning: strategy {resolved_strategy!r} not found in {eval_config}, ignoring")
		return {}
	params = dict(eval_cfg.get("defaults", {}))
	params.update(strategies[resolved_strategy])
	params["strategy_name"] = resolved_strategy
	return params


def main() -> int:
	args = _build_arg_parser().parse_args()
	paths = _build_run_paths(args.output_dir, args.run_name)

	# ── Load strategy config from eval.yaml ──────────────────────────────────
	strategy_params = _load_strategy_params(args.eval_config, args.strategy)
	strategy_name = strategy_params.get("strategy_name", "(none)")

	# CLI args override strategy when explicitly provided
	resolved_matcher = args.matcher or strategy_params.get("matcher", "hungarian")
	resolved_det_iou = args.det_iou_threshold or strategy_params.get("iou_threshold", 0.3)
	resolved_trk_iou = args.trk_iou_threshold or strategy_params.get("iou_threshold", 0.3)

	# ── Build category remapper from the chosen strategy ─────────────────────
	remapper = None
	if Path(args.eval_config).exists():
		try:
			remapper = build_remapper_from_eval_config(
				args.eval_config,
				strategy=args.strategy,
			)
			print(f"[run] strategy={strategy_name!r}  schema={strategy_params.get('category_schema', 'raw')}  "
			      f"target_classes={remapper.target_classes or '(raw)'}")
		except Exception as exc:
			print(f"[run] warning: could not build remapper — {exc}")

	loader = NuScenesLoader(data_root=args.data_root, version=args.version, verbose=False)
	loader.load()
	scene_id = args.scene_id or loader.get_scene_ids()[0]

	raw_frame_records = list(loader.iter_frame_records(scene_id=scene_id, max_frames=args.max_frames))

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
		topn_visualizations=args.topn,
		topn_per_scenario=args.topn_per_scenario,
		visualization_dir=str(paths["viz_dir"]),
		use_occlusion_bucketing=args.occlusion_bucketing,
		use_combined_bucketing=args.combined_bucketing,
	)
	tracking_result = evaluate_tracking_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=resolved_trk_iou,
		matcher=resolved_matcher,
	)

	full_result = {
		"detection": result,
		"tracking": tracking_result,
	}
	full_result["config"] = {
		"dataset": args.version,
		"scene": scene_id,
		"max_frames": args.max_frames,
		"strategy": strategy_name,
		"category_schema": strategy_params.get("category_schema", "raw"),
		"matcher": resolved_matcher,
		"det_iou_threshold": resolved_det_iou,
		"trk_iou_threshold": resolved_trk_iou,
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
			"dataset": args.version,
			"scene": scene_id,
			"seed": args.seed,
			"strategy": strategy_name,
			"category_schema": strategy_params.get("category_schema", "raw"),
		},
	)
	paths["report_md"].write_text(markdown, encoding="utf-8")

	det_overall = result["overall"]
	trk_overall = tracking_result["overall"]
	print(f"[run] scene={scene_id} frames={result['num_frames']} matcher={result['matcher']}")
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
