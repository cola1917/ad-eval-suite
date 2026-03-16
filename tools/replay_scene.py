from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if os.environ.get("DISPLAY", "") == "":  # pragma: no cover
	import matplotlib

	matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation

try:
	from scenario.map_overlay import draw_map_overlay, load_map_geometry, query_map_patch
except ImportError:  # pragma: no cover
	import sys
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from scenario.map_overlay import draw_map_overlay, load_map_geometry, query_map_patch


def _rotation_corners(x: float, y: float, w: float, l: float, yaw: float) -> List[Tuple[float, float]]:
	half_w = w / 2.0
	half_l = l / 2.0
	local = [
		(+half_l, +half_w),
		(+half_l, -half_w),
		(-half_l, -half_w),
		(-half_l, +half_w),
	]
	cos_yaw = float(math.cos(yaw))
	sin_yaw = float(math.sin(yaw))
	return [
		(x + px * cos_yaw - py * sin_yaw, y + px * sin_yaw + py * cos_yaw)
		for px, py in local
	]


def _draw_bbox(ax: Any, bbox: Dict[str, Any], color: str, linewidth: float = 1.2) -> None:
	x = float(bbox.get("x", 0.0))
	y = float(bbox.get("y", 0.0))
	w = max(float(bbox.get("w", 0.0)), 0.1)
	l = max(float(bbox.get("l", 0.0)), 0.1)
	yaw = float(bbox.get("yaw", 0.0))
	corners = _rotation_corners(x, y, w, l, yaw)
	xs = [pt[0] for pt in corners] + [corners[0][0]]
	ys = [pt[1] for pt in corners] + [corners[0][1]]
	ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.9)
	ax.scatter([x], [y], color=color, s=10, alpha=0.9)


def _match_sets(frame: Dict[str, Any]) -> Dict[str, set[str]]:
	tp_pred_ids: set[str] = set()
	fp_pred_ids: set[str] = set()
	fn_gt_ids: set[str] = set()
	for item in frame.get("matches", []):
		kind = str(item.get("kind", "")).lower()
		if kind == "tp":
			tp_pred_ids.add(str(item.get("pred_id", "")))
		if kind == "fp":
			fp_pred_ids.add(str(item.get("pred_id", "")))
		if kind == "fn":
			fn_gt_ids.add(str(item.get("gt_id", "")))
	return {
		"tp_pred_ids": tp_pred_ids,
		"fp_pred_ids": fp_pred_ids,
		"fn_gt_ids": fn_gt_ids,
	}


def _collect_trajectories(frames: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Tuple[float, float]]]:
	trajectories: Dict[str, List[Tuple[float, float]]] = {}
	for frame in frames:
		for agent in frame.get(key, []):
			agent_id = str(agent.get("id", ""))
			bbox = agent.get("bbox", {})
			point = (float(bbox.get("x", 0.0)), float(bbox.get("y", 0.0)))
			trajectories.setdefault(agent_id, []).append(point)
	return trajectories


def _axis_limits(frame: Dict[str, Any], margin: float = 15.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	points: List[Tuple[float, float]] = []
	for key in ("gt_agents", "pred_agents"):
		for agent in frame.get(key, []):
			bbox = agent.get("bbox", {})
			points.append((float(bbox.get("x", 0.0)), float(bbox.get("y", 0.0))))
	ego = frame.get("ego", {}).get("pose", [0.0, 0.0, 0.0])
	points.append((float(ego[0]), float(ego[1])))

	min_x = min(point[0] for point in points)
	max_x = max(point[0] for point in points)
	min_y = min(point[1] for point in points)
	max_y = max(point[1] for point in points)
	return ((min_x - margin, max_x + margin), (min_y - margin, max_y + margin))


def render_frame(
	*,
	ax: Any,
	frame: Dict[str, Any],
	show_trajectories: bool,
	gt_trajectories: Dict[str, List[Tuple[float, float]]],
	pred_trajectories: Dict[str, List[Tuple[float, float]]],
	map_context: Dict[str, Any] | None = None,
) -> None:
	ax.clear()
	# Dark background so map layer contrast is better.
	ax.set_facecolor("#1a1a1a")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, linestyle="--", alpha=0.15, color="#555555")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")

	# ── Map overlay (drawn first so agents render on top) ─────────────────
	if map_context is not None:
		ego_pose = frame.get("ego", {}).get("pose", [0.0, 0.0, 0.0])
		cx, cy = float(ego_pose[0]), float(ego_pose[1])
		patch = query_map_patch(map_context, cx=cx, cy=cy, half_extent=60.0)
		draw_map_overlay(ax, patch)

	match_sets = _match_sets(frame)

	if show_trajectories:
		for traj in gt_trajectories.values():
			xs = [pt[0] for pt in traj]
			ys = [pt[1] for pt in traj]
			ax.plot(xs, ys, color="tab:blue", linewidth=0.8, alpha=0.25)
		for traj in pred_trajectories.values():
			xs = [pt[0] for pt in traj]
			ys = [pt[1] for pt in traj]
			ax.plot(xs, ys, color="tab:green", linewidth=0.8, alpha=0.2)

	for agent in frame.get("gt_agents", []):
		agent_id = str(agent.get("id", ""))
		color = "yellow" if agent_id in match_sets["fn_gt_ids"] else "tab:blue"
		_draw_bbox(ax, agent.get("bbox", {}), color=color)

	for agent in frame.get("pred_agents", []):
		agent_id = str(agent.get("id", ""))
		if agent_id in match_sets["tp_pred_ids"]:
			color = "tab:green"
		elif agent_id in match_sets["fp_pred_ids"]:
			color = "tab:red"
		else:
			color = "tab:gray"
		_draw_bbox(ax, agent.get("bbox", {}), color=color)

	ego_pose = frame.get("ego", {}).get("pose", [0.0, 0.0, 0.0])
	ax.scatter([float(ego_pose[0])], [float(ego_pose[1])], marker="*", s=120, color="black")

	frame_metrics = frame.get("frame_metrics", {})
	title = (
		f"scene={frame.get('scene_id', 'unknown')} frame={frame.get('frame_index', 0)} "
		f"TP={frame_metrics.get('tp', 0)} FP={frame_metrics.get('fp', 0)} FN={frame_metrics.get('fn', 0)}"
	)
	ax.set_title(title)

	(xlim, ylim) = _axis_limits(frame)
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)


def _load_scene(path: Path) -> Dict[str, Any]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if "frames" not in payload or not isinstance(payload["frames"], list):
		raise ValueError("Snapshot file must include a list field: frames")
	if not payload["frames"]:
		raise ValueError("Snapshot frames is empty")
	return payload


def _save_frame_image(fig: Any, out_dir: Path, frame: Dict[str, Any]) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	filename = f"frame_{int(frame.get('frame_index', 0)):04d}.png"
	fig.savefig(out_dir / filename, dpi=150)


def _load_map_for_payload(
	snapshot_payload: Dict[str, Any],
	map_data_root: str | None,
) -> Dict[str, Any] | None:
	"""Load map geometry when *map_data_root* is given and the payload has a location."""
	if not map_data_root:
		return None
	location = str(snapshot_payload.get("location", ""))
	if not location:
		return None
	return load_map_geometry(map_data_root, location)


def export_scene_frames(
	*,
	snapshot_payload: Dict[str, Any],
	output_dir: str,
	show_trajectories: bool = False,
	frame_indices: Sequence[int] | None = None,
	map_data_root: str | None = None,
) -> List[str]:
	frames = sorted(snapshot_payload.get("frames", []), key=lambda frame: int(frame.get("frame_index", 0)))
	if not frames:
		return []

	selected = frames
	if frame_indices is not None:
		frame_index_set = {int(index) for index in frame_indices}
		selected = [frame for frame in frames if int(frame.get("frame_index", -1)) in frame_index_set]

	map_context = _load_map_for_payload(snapshot_payload, map_data_root)
	gt_trajectories = _collect_trajectories(frames, "gt_agents")
	pred_trajectories = _collect_trajectories(frames, "pred_agents")
	fig, ax = plt.subplots(figsize=(9, 9))
	out_dir = Path(output_dir)
	written_files: List[str] = []
	for frame in selected:
		render_frame(
			ax=ax,
			frame=frame,
			show_trajectories=show_trajectories,
			gt_trajectories=gt_trajectories,
			pred_trajectories=pred_trajectories,
			map_context=map_context,
		)
		_save_frame_image(fig, out_dir, frame)
		written_files.append(str(out_dir / f"frame_{int(frame.get('frame_index', 0)):04d}.png"))
	plt.close(fig)
	return written_files


def export_scene_gif(
	*,
	snapshot_payload: Dict[str, Any],
	output_file: str,
	show_trajectories: bool = False,
	fps: int = 3,
	map_data_root: str | None = None,
) -> str:
	"""Export a scene snapshot payload as an animated GIF file."""
	frames = sorted(snapshot_payload.get("frames", []), key=lambda frame: int(frame.get("frame_index", 0)))
	if not frames:
		raise ValueError("Snapshot payload has no frames to animate")

	map_context = _load_map_for_payload(snapshot_payload, map_data_root)
	gt_trajectories = _collect_trajectories(frames, "gt_agents")
	pred_trajectories = _collect_trajectories(frames, "pred_agents")
	fig, ax = plt.subplots(figsize=(9, 9))

	def _update(idx: int) -> None:
		render_frame(
			ax=ax,
			frame=frames[idx],
			show_trajectories=show_trajectories,
			gt_trajectories=gt_trajectories,
			pred_trajectories=pred_trajectories,
			map_context=map_context,
		)

	ani = animation.FuncAnimation(
		fig,
		_update,
		frames=len(frames),
		interval=max(int(1000 / max(fps, 1)), 1),
		repeat=True,
	)

	output_path = Path(output_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	ani.save(str(output_path), writer=animation.PillowWriter(fps=max(fps, 1)))
	plt.close(fig)
	return str(output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Replay ScenarioSnapshot scene files in a 2D top-down view")
	parser.add_argument("scene_file", help="Path to snapshot scene json file")
	parser.add_argument("--frame-index", type=int, default=None, help="Render one specific frame index")
	parser.add_argument("--animate", action="store_true", help="Play all frames sequentially")
	parser.add_argument("--interval-ms", type=int, default=350, help="Animation step interval in milliseconds")
	parser.add_argument("--save-dir", default=None, help="Directory to save rendered frame images")
	parser.add_argument("--save-gif", default=None, help="Path to save an animated GIF")
	parser.add_argument("--gif-fps", type=int, default=3, help="Animated GIF frame rate")
	parser.add_argument("--map-data-root", default=None, help="nuScenes data root for lane/map overlay (e.g. data/nuscenes-mini)")
	parser.add_argument("--show-trajectories", action="store_true", help="Overlay GT and prediction trajectories")
	parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
	return parser


def main() -> int:
	args = _build_arg_parser().parse_args()
	scene_path = Path(args.scene_file)
	payload = _load_scene(scene_path)
	frames = sorted(payload.get("frames", []), key=lambda frame: int(frame.get("frame_index", 0)))

	gt_trajectories = _collect_trajectories(frames, "gt_agents")
	pred_trajectories = _collect_trajectories(frames, "pred_agents")

	fig, ax = plt.subplots(figsize=(9, 9))

	if args.frame_index is not None:
		candidates = [frame for frame in frames if int(frame.get("frame_index", -1)) == int(args.frame_index)]
		if not candidates:
			raise ValueError(f"frame_index={args.frame_index} not found in scene")
		render_frame(
			ax=ax,
			frame=candidates[0],
			show_trajectories=args.show_trajectories,
			gt_trajectories=gt_trajectories,
			pred_trajectories=pred_trajectories,
		)
		if args.save_dir:
			_save_frame_image(fig, Path(args.save_dir), candidates[0])
	elif args.animate:
		for frame in frames:
			render_frame(
				ax=ax,
				frame=frame,
				show_trajectories=args.show_trajectories,
				gt_trajectories=gt_trajectories,
				pred_trajectories=pred_trajectories,
			)
			if args.save_dir:
				_save_frame_image(fig, Path(args.save_dir), frame)
			if not args.no_show:
				plt.pause(max(args.interval_ms, 1) / 1000.0)
	else:
		render_frame(
			ax=ax,
			frame=frames[0],
			show_trajectories=args.show_trajectories,
			gt_trajectories=gt_trajectories,
			pred_trajectories=pred_trajectories,
		)
		if args.save_dir:
			_save_frame_image(fig, Path(args.save_dir), frames[0])

	if args.save_gif:
		export_scene_gif(
			snapshot_payload=payload,
			output_file=args.save_gif,
			show_trajectories=args.show_trajectories,
			fps=args.gif_fps,
			map_data_root=args.map_data_root,
		)

	if not args.no_show:
		plt.show()
	plt.close(fig)
	print(f"[replay] loaded scene={payload.get('scene_id', 'unknown')} frames={len(frames)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
