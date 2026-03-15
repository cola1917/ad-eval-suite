from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET


def _indent(elem: ET.Element, level: int = 0) -> None:
	i = "\n" + level * "  "
	if len(elem):
		if not elem.text or not elem.text.strip():
			elem.text = i + "  "
		for child in elem:
			_indent(child, level + 1)
		if not elem[-1].tail or not elem[-1].tail.strip():
			elem[-1].tail = i
	if level and (not elem.tail or not elem.tail.strip()):
		elem.tail = i


def _entity_label(agent_type: str, is_ego: bool) -> str:
	if is_ego:
		return "car"
	text = agent_type.lower()
	if "pedestrian" in text:
		return "pedestrian"
	if "bicycle" in text or "cyclist" in text:
		return "bicycle"
	return "car"


def _collect_entity_trajectories(snapshot: Dict[str, Any]) -> Dict[str, List[Tuple[float, float, float, float]]]:
	frames = snapshot.get("frames", [])
	if not frames:
		return {}

	t0 = float(frames[0].get("timestamp", 0.0))
	trajectories: Dict[str, List[Tuple[float, float, float, float]]] = {"ego": []}
	for frame in frames:
		t = max(0.0, float(frame.get("timestamp", 0.0)) - t0)
		ego_pose = frame.get("ego", {}).get("pose", [0.0, 0.0, 0.0])
		trajectories["ego"].append((t, float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])))
		for agent in frame.get("gt_agents", []):
			agent_id = str(agent.get("id", ""))
			bbox = agent.get("bbox", {})
			trajectories.setdefault(agent_id, []).append(
				(t, float(bbox.get("x", 0.0)), float(bbox.get("y", 0.0)), float(bbox.get("yaw", 0.0)))
			)
	return trajectories


def _collect_entity_types(snapshot: Dict[str, Any]) -> Dict[str, str]:
	types: Dict[str, str] = {"ego": "ego"}
	for frame in snapshot.get("frames", []):
		for agent in frame.get("gt_agents", []):
			agent_id = str(agent.get("id", ""))
			types.setdefault(agent_id, str(agent.get("type", "car")))
	return types


def _initial_speed(points: List[Tuple[float, float, float, float]]) -> float:
	if len(points) < 2:
		return 0.0
	t0, x0, y0, _ = points[0]
	t1, x1, y1, _ = points[1]
	dt = max(1e-6, t1 - t0)
	return math.hypot(x1 - x0, y1 - y0) / dt


def _build_vehicle_object(parent: ET.Element, name: str, category: str) -> None:
	scenario_object = ET.SubElement(parent, "ScenarioObject", attrib={"name": name})
	vehicle = ET.SubElement(
		scenario_object,
		"Vehicle",
		attrib={
			"name": f"{name}_vehicle",
			"vehicleCategory": category,
		},
	)
	ET.SubElement(vehicle, "ParameterDeclarations")
	ET.SubElement(vehicle, "Performance", attrib={"maxSpeed": "70", "maxAcceleration": "8", "maxDeceleration": "8"})
	ET.SubElement(vehicle, "BoundingBox")
	ET.SubElement(vehicle, "Axles")
	ET.SubElement(vehicle, "Properties")


def _add_world_position(parent: ET.Element, x: float, y: float, h: float) -> None:
	ET.SubElement(
		parent,
		"WorldPosition",
		attrib={
			"x": f"{x:.6f}",
			"y": f"{y:.6f}",
			"z": "0.0",
			"h": f"{h:.6f}",
			"p": "0.0",
			"r": "0.0",
		},
	)


def _add_simulation_time_trigger(parent: ET.Element, *, value: float, rule: str, name: str) -> None:
	condition_group = ET.SubElement(parent, "ConditionGroup")
	condition = ET.SubElement(
		condition_group,
		"Condition",
		attrib={"name": name, "delay": "0", "conditionEdge": "none"},
	)
	by_value = ET.SubElement(condition, "ByValueCondition")
	ET.SubElement(by_value, "SimulationTimeCondition", attrib={"value": f"{value:.3f}", "rule": rule})


def export_snapshot_to_xosc(
	*,
	snapshot_payload: Dict[str, Any],
	output_path: str,
	map_file: str = "",
	scene_graph_file: str = "",
) -> str:
	frames = snapshot_payload.get("frames", [])
	if not frames:
		raise ValueError("Scenario snapshot must contain non-empty frames")

	trajectories = _collect_entity_trajectories(snapshot_payload)
	entity_types = _collect_entity_types(snapshot_payload)

	root = ET.Element("OpenSCENARIO")
	ET.SubElement(
		root,
		"FileHeader",
		attrib={
			"revMajor": "1",
			"revMinor": "0",
			"date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
			"description": f"Exported from {snapshot_payload.get('schema_version', 'ScenarioSnapshot')}",
			"author": "ad-eval-suite",
		},
	)
	ET.SubElement(root, "ParameterDeclarations")
	ET.SubElement(root, "CatalogLocations")

	road_network = ET.SubElement(root, "RoadNetwork")
	ET.SubElement(road_network, "LogicFile", attrib={"filepath": map_file})
	ET.SubElement(road_network, "SceneGraphFile", attrib={"filepath": scene_graph_file})

	entities = ET.SubElement(root, "Entities")
	for entity_id in sorted(trajectories.keys()):
		category = _entity_label(entity_types.get(entity_id, "car"), is_ego=(entity_id == "ego"))
		_build_vehicle_object(entities, name=entity_id, category=category)

	storyboard = ET.SubElement(root, "Storyboard")
	init = ET.SubElement(storyboard, "Init")
	init_actions = ET.SubElement(init, "Actions")
	for entity_id, points in trajectories.items():
		if not points:
			continue
		private = ET.SubElement(init_actions, "Private", attrib={"entityRef": entity_id})
		private_action = ET.SubElement(private, "PrivateAction")
		teleport = ET.SubElement(private_action, "TeleportAction")
		position = ET.SubElement(teleport, "Position")
		_, x0, y0, h0 = points[0]
		_add_world_position(position, x=x0, y=y0, h=h0)
		speed_action = ET.SubElement(private, "PrivateAction")
		longitudinal = ET.SubElement(speed_action, "LongitudinalAction")
		speed = ET.SubElement(longitudinal, "SpeedAction")
		ET.SubElement(speed, "SpeedActionDynamics", attrib={"dynamicsShape": "step", "value": "0.0", "dynamicsDimension": "time"})
		ET.SubElement(speed, "SpeedActionTarget")
		speed_target = speed.find("SpeedActionTarget")
		assert speed_target is not None
		ET.SubElement(speed_target, "AbsoluteTargetSpeed", attrib={"value": f"{_initial_speed(points):.3f}"})

	story = ET.SubElement(storyboard, "Story", attrib={"name": "scenario_story"})
	end_time = max(points[-1][0] for points in trajectories.values() if points)
	for entity_id, points in trajectories.items():
		act = ET.SubElement(story, "Act", attrib={"name": f"act_{entity_id}"})
		act_start = ET.SubElement(act, "StartTrigger")
		_add_simulation_time_trigger(act_start, value=0.0, rule="greaterThan", name=f"act_start_{entity_id}")
		maneuver_group = ET.SubElement(act, "ManeuverGroup", attrib={"name": f"mg_{entity_id}", "maximumExecutionCount": "1"})
		actors = ET.SubElement(maneuver_group, "Actors", attrib={"selectTriggeringEntities": "false"})
		ET.SubElement(actors, "EntityRef", attrib={"entityRef": entity_id})
		maneuver = ET.SubElement(maneuver_group, "Maneuver", attrib={"name": f"man_{entity_id}"})
		event = ET.SubElement(maneuver, "Event", attrib={"name": f"event_{entity_id}", "priority": "overwrite"})
		action = ET.SubElement(event, "Action", attrib={"name": f"action_{entity_id}"})
		private_action = ET.SubElement(action, "PrivateAction")
		routing_action = ET.SubElement(private_action, "RoutingAction")
		follow_action = ET.SubElement(routing_action, "FollowTrajectoryAction")
		trajectory = ET.SubElement(follow_action, "Trajectory", attrib={"name": f"traj_{entity_id}", "closed": "false"})
		shape = ET.SubElement(trajectory, "Shape")
		polyline = ET.SubElement(shape, "Polyline")
		for t, x, y, h in points:
			vertex = ET.SubElement(polyline, "Vertex", attrib={"time": f"{t:.3f}"})
			position = ET.SubElement(vertex, "Position")
			_add_world_position(position, x=x, y=y, h=h)
		time_reference = ET.SubElement(follow_action, "TimeReference")
		ET.SubElement(time_reference, "Timing", attrib={"domainAbsoluteRelative": "relative", "scale": "1.0", "offset": "0.0"})
		ET.SubElement(
			follow_action,
			"TrajectoryFollowingMode",
			attrib={"followingMode": "position"},
		)
		event_start = ET.SubElement(event, "StartTrigger")
		_add_simulation_time_trigger(event_start, value=0.0, rule="greaterThan", name=f"event_start_{entity_id}")

	stop_trigger = ET.SubElement(storyboard, "StopTrigger")
	_add_simulation_time_trigger(stop_trigger, value=end_time, rule="greaterThan", name="scenario_stop")

	_indent(root)
	tree = ET.ElementTree(root)
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	tree.write(output, encoding="utf-8", xml_declaration=True)
	return str(output)


def _load_snapshot(path: Path) -> Dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Export ScenarioSnapshot JSON to a minimal OpenSCENARIO (.xosc) file")
	parser.add_argument("scene_file", help="Path to snapshot scene json")
	parser.add_argument("--output", default=None, help="Output .xosc path (defaults beside input file)")
	parser.add_argument("--map-file", default="", help="Optional OpenDRIVE map path reference")
	parser.add_argument("--scene-graph-file", default="", help="Optional scene graph file reference")
	return parser


def main() -> int:
	args = _build_arg_parser().parse_args()
	input_path = Path(args.scene_file)
	output_path = Path(args.output) if args.output else input_path.with_suffix(".xosc")
	snapshot_payload = _load_snapshot(input_path)
	path = export_snapshot_to_xosc(
		snapshot_payload=snapshot_payload,
		output_path=str(output_path),
		map_file=args.map_file,
		scene_graph_file=args.scene_graph_file,
	)
	print(f"[xosc] exported: {path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
