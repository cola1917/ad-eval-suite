from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from mining.error_aggregation import aggregate_error_categories
from mining.rank_frames import FailureScoreWeights, rank_frames_and_scenes
from mining.rank_scenes import top_k_scenes
from visualization.snapshot_schema import build_frame_snapshot, serialize_scene_snapshots


@dataclass(frozen=True)
class FailureMiningConfig:
	top_k_scenes: int = 3
	top_k_frames: int = 10
	weights: FailureScoreWeights = FailureScoreWeights()
	coordinate_system: str = "nuscenes_global"


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return str(path)


def run_failure_mining(
	*,
	run_dir: Path,
	frame_records: Sequence[Dict[str, Any]],
	prediction_records: Sequence[Dict[str, Any]],
	iou_threshold: float,
	matcher: str,
	class_aware: bool,
	config: FailureMiningConfig,
) -> Dict[str, Any]:
	failure_root = run_dir / "failure_mining"
	snapshots_dir = failure_root / "snapshots"
	summary_dir = failure_root / "summary"

	ranked = rank_frames_and_scenes(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=iou_threshold,
		matcher=matcher,
		class_aware=class_aware,
		weights=config.weights,
	)

	frames = ranked["frames"]
	scenes = ranked["scenes"]
	top_frames = frames[: min(config.top_k_frames, len(frames))] if config.top_k_frames > 0 else []
	top_scenes = top_k_scenes(scenes, config.top_k_scenes)
	top_scene_ids = {str(scene["scene_id"]) for scene in top_scenes}

	scene_metrics_map = {str(scene["scene_id"]): scene for scene in scenes}
	snapshots_by_scene: Dict[str, List[Dict[str, Any]]] = {}

	for row in frames:
		scene_id = str(row["scene_id"])
		if scene_id not in top_scene_ids:
			continue
		record_index = int(row.get("record_index", 0)) - 1
		if record_index < 0 or record_index >= len(frame_records):
			continue
		fr = frame_records[record_index]
		snapshot = build_frame_snapshot(
			scene_id=scene_id,
			frame_index=int(row.get("frame_index", 0)),
			frame_record=fr,
			prediction_record=prediction_records[record_index],
			frame_summary={
				"tp": row["tp"],
				"fp": row["fp"],
				"fn": row["fn"],
				"precision": row["precision"],
				"recall": row["recall"],
				"f1": row["f1"],
				"match_result": row.get("match_result", {}),
			},
			scene_metrics=scene_metrics_map.get(scene_id, {}),
			coordinate_system=config.coordinate_system,
			meta={
				"failure_score": row["failure_score"],
				"idf_switches": row["idf_switches"],
				"location": str(fr.get("location", "")),
			},
		)
		snapshots_by_scene.setdefault(scene_id, []).append(snapshot)

	snapshot_files: List[str] = []
	for scene_id, scene_snapshots in snapshots_by_scene.items():
		scene_snapshots = sorted(scene_snapshots, key=lambda item: int(item.get("frame_index", 0)))
		scene_payload = serialize_scene_snapshots(scene_snapshots)
		snapshot_files.append(
			_write_json(snapshots_dir / f"{scene_id}.json", scene_payload)
		)

	error_categories = aggregate_error_categories(top_frames)
	top_scene_path = _write_json(
		failure_root / "top_k_scenes.json",
		{
			"failure_score_name": "failure_score",
			"weights": asdict(config.weights),
			"top_k": config.top_k_scenes,
			"items": top_scenes,
		},
	)
	top_frame_path = _write_json(
		failure_root / "top_k_frames.json",
		{
			"failure_score_name": "failure_score",
			"weights": asdict(config.weights),
			"top_k": config.top_k_frames,
			"items": [
				{key: value for key, value in row.items() if key != "match_result"}
				for row in top_frames
			],
		},
	)
	error_summary_path = _write_json(summary_dir / "error_categories.json", error_categories)

	return {
		"enabled": True,
		"weights": asdict(config.weights),
		"top_k_scenes_path": top_scene_path,
		"top_k_frames_path": top_frame_path,
		"error_categories_path": error_summary_path,
		"snapshot_files": snapshot_files,
		"num_snapshots": len(snapshot_files),
	}
