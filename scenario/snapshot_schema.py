from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


SCENARIO_SNAPSHOT_SCHEMA_VERSION = "ScenarioSnapshot-1.0"


def _yaw_from_box(box: Dict[str, Any]) -> float:
	return float(box.get("yaw", 0.0))


def _agent_from_box(box: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
	translation = box.get("translation", [0.0, 0.0, 0.0])
	size = box.get("size", [0.0, 0.0, 0.0])
	velocity = box.get("velocity", [0.0, 0.0])
	agent_id = (
		box.get("track_id")
		or box.get("instance_token")
		or box.get("annotation_token")
		or box.get("source_gt_instance_token")
		or fallback_id
	)
	return {
		"id": str(agent_id),
		"type": str(box.get("category_name", "unknown")),
		"bbox": {
			"x": float(translation[0]),
			"y": float(translation[1]),
			"z": float(translation[2]) if len(translation) > 2 else 0.0,
			"w": float(size[0]) if len(size) > 0 else 0.0,
			"l": float(size[1]) if len(size) > 1 else 0.0,
			"h": float(size[2]) if len(size) > 2 else 0.0,
			"yaw": _yaw_from_box(box),
		},
		"velocity": [
			float(velocity[0]) if len(velocity) > 0 else 0.0,
			float(velocity[1]) if len(velocity) > 1 else 0.0,
		],
		"score": float(box.get("score", 0.0)),
	}


def _to_match_entries(match_result: Dict[str, Any]) -> List[Dict[str, Any]]:
	entries: List[Dict[str, Any]] = []
	for item in match_result.get("matches", []):
		gt_box = item.get("gt_box", {})
		pred_box = item.get("pred_box", {})
		entries.append(
			{
				"kind": "tp",
				"gt_id": str(gt_box.get("track_id") or gt_box.get("instance_token") or item.get("gt_index", "")),
				"pred_id": str(pred_box.get("track_id") or item.get("pred_index", "")),
				"gt_index": int(item.get("gt_index", -1)),
				"pred_index": int(item.get("pred_index", -1)),
				"iou": float(item.get("iou", 0.0)),
				"center_distance": float(item.get("center_distance", 0.0)),
			}
		)

	for item in match_result.get("false_positives", []):
		pred_box = item.get("pred_box", {})
		entries.append(
			{
				"kind": "fp",
				"pred_id": str(pred_box.get("track_id") or item.get("pred_index", "")),
				"pred_index": int(item.get("pred_index", -1)),
				"score": float(item.get("score", pred_box.get("score", 0.0))),
			}
		)

	for item in match_result.get("false_negatives", []):
		gt_box = item.get("gt_box", {})
		entries.append(
			{
				"kind": "fn",
				"gt_id": str(gt_box.get("track_id") or gt_box.get("instance_token") or item.get("gt_index", "")),
				"gt_index": int(item.get("gt_index", -1)),
			}
		)

	return entries


def build_frame_snapshot(
	*,
	scene_id: str,
	frame_index: int,
	frame_record: Dict[str, Any],
	prediction_record: Dict[str, Any],
	frame_summary: Dict[str, Any],
	scene_metrics: Dict[str, Any],
	coordinate_system: str = "nuscenes_global",
	meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
	ego_pose = frame_record.get("ego_pose", {})
	ego_translation = ego_pose.get("translation", [0.0, 0.0, 0.0])
	ego_velocity = ego_pose.get("velocity", [0.0, 0.0])

	gt_boxes = frame_record.get("gt_boxes", [])
	pred_boxes = prediction_record.get("pred_boxes", [])
	match_result = frame_summary.get("match_result", {})

	return {
		"schema_version": SCENARIO_SNAPSHOT_SCHEMA_VERSION,
		"coordinate_system": coordinate_system,
		"scene_id": scene_id,
		"frame_index": int(frame_index),
		"timestamp": float(frame_record.get("timestamp", 0.0)),
		"sample_token": str(frame_record.get("sample_token", "")),
		"ego": {
			"pose": [
				float(ego_translation[0]) if len(ego_translation) > 0 else 0.0,
				float(ego_translation[1]) if len(ego_translation) > 1 else 0.0,
				float(ego_pose.get("yaw", 0.0)),
			],
			"velocity": [
				float(ego_velocity[0]) if len(ego_velocity) > 0 else 0.0,
				float(ego_velocity[1]) if len(ego_velocity) > 1 else 0.0,
			],
		},
		"gt_agents": [_agent_from_box(box, fallback_id=f"gt_{idx}") for idx, box in enumerate(gt_boxes)],
		"pred_agents": [_agent_from_box(box, fallback_id=f"pred_{idx}") for idx, box in enumerate(pred_boxes)],
		"matches": _to_match_entries(match_result),
		"frame_metrics": {
			"tp": int(frame_summary.get("tp", 0)),
			"fp": int(frame_summary.get("fp", 0)),
			"fn": int(frame_summary.get("fn", 0)),
			"precision": float(frame_summary.get("precision", 0.0)),
			"recall": float(frame_summary.get("recall", 0.0)),
			"f1": float(frame_summary.get("f1", 0.0)),
		},
		"scene_metrics": scene_metrics,
		"meta": meta or {},
	}


def serialize_scene_snapshots(snapshots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	if not snapshots:
		return {
			"schema_version": SCENARIO_SNAPSHOT_SCHEMA_VERSION,
			"num_frames": 0,
			"frames": [],
		}

	first_meta = snapshots[0].get("meta", {})
	return {
		"schema_version": SCENARIO_SNAPSHOT_SCHEMA_VERSION,
		"coordinate_system": snapshots[0].get("coordinate_system", "nuscenes_global"),
		"scene_id": snapshots[0].get("scene_id", "unknown_scene"),
		"location": str(first_meta.get("location", "")),
		"num_frames": len(snapshots),
		"frames": list(snapshots),
	}
