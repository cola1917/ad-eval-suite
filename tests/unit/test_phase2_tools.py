from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from simulator_export.export_openscenario import export_snapshot_to_xosc
from tools.replay_scene import _load_scene, _match_sets, export_scene_frames


def _snapshot_payload() -> dict:
	return {
		"schema_version": "ScenarioSnapshot-1.0",
		"coordinate_system": "nuscenes_global",
		"scene_id": "scene-001",
		"num_frames": 2,
		"frames": [
			{
				"schema_version": "ScenarioSnapshot-1.0",
				"scene_id": "scene-001",
				"frame_index": 1,
				"timestamp": 1.0,
				"ego": {"pose": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0]},
				"gt_agents": [{"id": "car-1", "type": "car", "bbox": {"x": 1.0, "y": 1.0, "w": 2.0, "l": 4.0, "h": 1.5, "z": 0.0, "yaw": 0.0}}],
				"pred_agents": [{"id": "pred-1", "type": "car", "bbox": {"x": 1.0, "y": 1.0, "w": 2.0, "l": 4.0, "h": 1.5, "z": 0.0, "yaw": 0.0}, "score": 0.9}],
				"matches": [{"kind": "tp", "gt_id": "car-1", "pred_id": "pred-1", "gt_index": 0, "pred_index": 0, "iou": 1.0, "center_distance": 0.0}],
				"frame_metrics": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
				"scene_metrics": {},
				"meta": {},
			},
			{
				"schema_version": "ScenarioSnapshot-1.0",
				"scene_id": "scene-001",
				"frame_index": 2,
				"timestamp": 2.0,
				"ego": {"pose": [0.5, 0.0, 0.0], "velocity": [0.0, 0.0]},
				"gt_agents": [{"id": "car-1", "type": "car", "bbox": {"x": 2.0, "y": 1.0, "w": 2.0, "l": 4.0, "h": 1.5, "z": 0.0, "yaw": 0.0}}],
				"pred_agents": [{"id": "pred-2", "type": "car", "bbox": {"x": 30.0, "y": 20.0, "w": 2.0, "l": 4.0, "h": 1.5, "z": 0.0, "yaw": 0.0}, "score": 0.3}],
				"matches": [
					{"kind": "fp", "pred_id": "pred-2", "pred_index": 0, "score": 0.3},
					{"kind": "fn", "gt_id": "car-1", "gt_index": 0},
				],
				"frame_metrics": {"tp": 0, "fp": 1, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0},
				"scene_metrics": {},
				"meta": {},
			},
		],
	}


def test_export_snapshot_to_xosc(tmp_path: Path) -> None:
	payload = _snapshot_payload()
	out_file = tmp_path / "scene-001.xosc"
	exported = export_snapshot_to_xosc(snapshot_payload=payload, output_path=str(out_file), map_file="maps/demo.xodr")

	assert Path(exported).exists()
	root = ET.parse(exported).getroot()
	assert root.tag == "OpenSCENARIO"
	assert len(root.findall("./Entities/ScenarioObject")) >= 2
	assert root.find("./RoadNetwork/LogicFile").attrib.get("filepath") == "maps/demo.xodr"
	assert len(root.findall(".//Vertex")) >= 4
	assert len(root.findall(".//AbsoluteTargetSpeed")) >= 1
	assert len(root.findall(".//SimulationTimeCondition")) >= 1


def test_replay_match_sets_and_loader(tmp_path: Path) -> None:
	payload = _snapshot_payload()
	file_path = tmp_path / "scene.json"
	file_path.write_text(json.dumps(payload), encoding="utf-8")
	loaded = _load_scene(file_path)

	assert loaded["scene_id"] == "scene-001"
	match_sets = _match_sets(loaded["frames"][1])
	assert "pred-2" in match_sets["fp_pred_ids"]
	assert "car-1" in match_sets["fn_gt_ids"]


def test_export_scene_frames_writes_images(tmp_path: Path) -> None:
	payload = _snapshot_payload()
	output_dir = tmp_path / "replay"
	written = export_scene_frames(snapshot_payload=payload, output_dir=str(output_dir), show_trajectories=True)

	assert len(written) == 2
	for path in written:
		assert Path(path).exists()
