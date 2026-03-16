from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from visualization.map_overlay import draw_map_overlay, load_map_geometry, query_map_patch
from simulation.export_openscenario import export_snapshot_to_xosc
from visualization.replay_scene import _load_scene, _match_sets, export_scene_frames, export_scene_gif


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
	written = export_scene_frames(
		snapshot_payload=payload,
		output_dir=str(output_dir),
		show_trajectories=True,
		view_mode="ego_fixed",
		view_half_extent=50.0,
		dpi=150,
	)

	assert len(written) == 2
	for path in written:
		assert Path(path).exists()

	# Exported frame pixel size should remain consistent across frames.
	with Image.open(written[0]) as img0, Image.open(written[1]) as img1:
		assert img0.size == img1.size


def test_export_scene_gif_writes_animation(tmp_path: Path) -> None:
	payload = _snapshot_payload()
	gif_path = tmp_path / "replay" / "scene-001.gif"
	written = export_scene_gif(
		snapshot_payload=payload,
		output_file=str(gif_path),
		show_trajectories=True,
		fps=4,
	)

	assert Path(written).exists()
	assert Path(written).suffix == ".gif"
	assert Path(written).stat().st_size > 0


def _mini_map_expansion_json(tmp_path: Path) -> Path:
	"""Write a minimal expansion JSON with two nodes and one polygon/line per layer."""
	expansion_dir = tmp_path / "maps" / "expansion"
	expansion_dir.mkdir(parents=True, exist_ok=True)
	node_a = {"token": "na", "x": 10.0, "y": 20.0}
	node_b = {"token": "nb", "x": 12.0, "y": 22.0}
	node_c = {"token": "nc", "x": 10.0, "y": 22.0}
	poly = {"token": "poly1", "exterior_node_tokens": ["na", "nb", "nc"]}
	line = {"token": "line1", "node_tokens": ["na", "nb"]}
	expansion = {
		"node": [node_a, node_b, node_c],
		"polygon": [poly],
		"line": [line],
		"drivable_area": [{"token": "da1", "polygon_tokens": ["poly1"]}],
		"lane_divider": [{"token": "ld1", "line_token": "line1"}],
		"road_divider": [{"token": "rd1", "line_token": "line1"}],
		"ped_crossing": [{"token": "pc1", "polygon_token": "poly1"}],
		"lane": [],
	}
	out = expansion_dir / "boston-seaport.json"
	out.write_text(json.dumps(expansion), encoding="utf-8")
	return tmp_path


def test_load_map_geometry_returns_indices(tmp_path: Path) -> None:
	data_root = _mini_map_expansion_json(tmp_path)
	mg = load_map_geometry(str(data_root), "boston-seaport")

	assert mg is not None
	assert mg["location"] == "boston-seaport"
	assert "na" in mg["_nodes"]
	assert "poly1" in mg["_polygons"]
	assert "line1" in mg["_lines"]


def test_load_map_geometry_missing_file_returns_none(tmp_path: Path) -> None:
	mg = load_map_geometry(str(tmp_path), "boston-seaport")
	assert mg is None


def test_query_map_patch_within_viewport(tmp_path: Path) -> None:
	data_root = _mini_map_expansion_json(tmp_path)
	mg = load_map_geometry(str(data_root), "boston-seaport")
	# Centre exactly at the nodes, small viewport still captures them.
	patch = query_map_patch(mg, cx=11.0, cy=21.0, half_extent=5.0)

	assert len(patch["drivable_polys"]) == 1
	assert len(patch["lane_div_lines"]) == 1
	assert len(patch["road_div_lines"]) == 1
	assert len(patch["ped_crossing_polys"]) == 1


def test_query_map_patch_outside_viewport_empty(tmp_path: Path) -> None:
	data_root = _mini_map_expansion_json(tmp_path)
	mg = load_map_geometry(str(data_root), "boston-seaport")
	# Centre far away – nothing should match.
	patch = query_map_patch(mg, cx=9000.0, cy=9000.0, half_extent=5.0)

	assert patch["drivable_polys"] == []
	assert patch["lane_div_lines"] == []
	assert patch["road_div_lines"] == []
	assert patch["ped_crossing_polys"] == []


def test_draw_map_overlay_does_not_raise(tmp_path: Path) -> None:
	data_root = _mini_map_expansion_json(tmp_path)
	mg = load_map_geometry(str(data_root), "boston-seaport")
	patch = query_map_patch(mg, cx=11.0, cy=21.0, half_extent=5.0)

	fig, ax = plt.subplots()
	draw_map_overlay(ax, patch)  # must not raise
	plt.close(fig)


def test_export_scene_frames_with_map_overlay(tmp_path: Path) -> None:
	"""Frames export succeeds when map_data_root is provided and location matches."""
	data_root = _mini_map_expansion_json(tmp_path)
	payload = _snapshot_payload()
	payload["location"] = "boston-seaport"
	output_dir = tmp_path / "replay_map"
	written = export_scene_frames(
		snapshot_payload=payload,
		output_dir=str(output_dir),
		map_data_root=str(data_root),
	)
	assert len(written) == 2
	for path in written:
		assert Path(path).exists()
