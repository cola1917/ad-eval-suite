from __future__ import annotations

import json
from pathlib import Path

from failure_mining.pipeline import FailureMiningConfig, run_failure_mining
from failure_mining.rank_frames import FailureScoreWeights


def _gt(track_id: str, category: str, x: float, y: float) -> dict:
	return {
		"track_id": track_id,
		"instance_token": track_id,
		"category_name": category,
		"translation": [x, y, 0.0],
		"size": [2.0, 4.0, 1.5],
		"yaw": 0.0,
		"velocity": [0.0, 0.0],
	}


def _pred(track_id: str, category: str, x: float, y: float, score: float) -> dict:
	return {
		"track_id": track_id,
		"category_name": category,
		"translation": [x, y, 0.0],
		"size": [2.0, 4.0, 1.5],
		"yaw": 0.0,
		"velocity": [0.0, 0.0],
		"score": score,
	}


def test_failure_mining_exports_topk_and_snapshots(tmp_path: Path) -> None:
	frames = [
		{
			"scene_name": "scene-001",
			"sample_token": "sample-a",
			"timestamp": 1.0,
			"ego_pose": {"translation": [0.0, 0.0, 0.0], "yaw": 0.0},
			"gt_boxes": [_gt("car-1", "car", 0.0, 0.0)],
		},
		{
			"scene_name": "scene-002",
			"sample_token": "sample-b",
			"timestamp": 2.0,
			"ego_pose": {"translation": [0.0, 0.0, 0.0], "yaw": 0.0},
			"gt_boxes": [_gt("car-2", "car", 5.0, 0.0), _gt("ped-2", "pedestrian", 10.0, 0.0)],
		},
	]
	preds = [
		{"pred_boxes": [_pred("pred-car-1", "car", 0.1, 0.1, 0.95)]},
		{"pred_boxes": [_pred("pred-car-2", "car", 5.0, 0.0, 0.9)]},
	]

	result = run_failure_mining(
		run_dir=tmp_path,
		frame_records=frames,
		prediction_records=preds,
		iou_threshold=0.5,
		matcher="greedy",
		class_aware=True,
		config=FailureMiningConfig(
			top_k_scenes=1,
			top_k_frames=2,
			weights=FailureScoreWeights(w_fn=1.0, w_fp=0.5, w_map=2.0, w_idsw=1.0),
		),
	)

	assert result["enabled"] is True
	assert Path(result["top_k_scenes_path"]).exists()
	assert Path(result["top_k_frames_path"]).exists()
	assert Path(result["error_categories_path"]).exists()
	assert len(result["snapshot_files"]) == 1

	top_scene_payload = json.loads(Path(result["top_k_scenes_path"]).read_text(encoding="utf-8"))
	assert top_scene_payload["top_k"] == 1
	assert len(top_scene_payload["items"]) == 1
	assert top_scene_payload["items"][0]["scene_id"] == "scene-002"

	snapshot_payload = json.loads(Path(result["snapshot_files"][0]).read_text(encoding="utf-8"))
	assert snapshot_payload["schema_version"] == "ScenarioSnapshot-1.0"
	assert snapshot_payload["scene_id"] == "scene-002"
	assert snapshot_payload["num_frames"] == 1
