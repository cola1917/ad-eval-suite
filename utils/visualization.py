from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matching.iou_matching import box_to_bev_corners


def save_detection_bev_plot(
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
	output_path: str,
	title: str,
	match_result: Dict[str, Any] | None = None,
) -> str:
	"""Save a BEV visualization of GT and prediction boxes.

	GT boxes are blue, matched predictions are green, unmatched predictions are red.
	"""

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	fig, ax = plt.subplots(figsize=(8, 8))
	ax.set_title(title)
	ax.set_xlabel("x (m)")
	ax.set_ylabel("y (m)")
	ax.grid(True, linestyle="--", alpha=0.35)
	ax.set_aspect("equal", adjustable="box")

	for gt_box in gt_boxes:
		_draw_box(ax, gt_box, edge_color="tab:blue", label="gt")

	matched_pred_indices = set()
	matched_gt_indices = set()
	if match_result is not None:
		for match in match_result.get("matches", []):
			matched_pred_indices.add(match["pred_index"])
			matched_gt_indices.add(match["gt_index"])

	for pred_index, pred_box in enumerate(pred_boxes):
		is_matched = pred_index in matched_pred_indices
		edge_color = "tab:green" if is_matched else "tab:red"
		_draw_box(ax, pred_box, edge_color=edge_color, label="pred")

	# Ego vehicle marker at the origin in ego coordinates.
	ax.scatter([0.0], [0.0], marker="*", s=120, color="black", alpha=0.8)

	tp = len(matched_pred_indices)
	fp = max(0, len(pred_boxes) - tp)
	fn = max(0, len(gt_boxes) - len(matched_gt_indices))
	subtitle = f"GT={len(gt_boxes)} Pred={len(pred_boxes)} TP={tp} FP={fp} FN={fn}"
	ax.text(
		0.01,
		0.99,
		subtitle,
		transform=ax.transAxes,
		verticalalignment="top",
		horizontalalignment="left",
		fontsize=9,
		bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
	)
	_add_legend(ax)

	_apply_axis_limits(ax, gt_boxes, pred_boxes)
	fig.tight_layout()
	fig.savefig(path, dpi=160)
	plt.close(fig)
	return str(path)


def _add_legend(ax: Any) -> None:
	handles = [
		Line2D([0], [0], color="tab:blue", lw=1.6, label="GT"),
		Line2D([0], [0], color="tab:green", lw=1.6, label="Pred Matched (TP)"),
		Line2D([0], [0], color="tab:red", lw=1.6, label="Pred Unmatched (FP)"),
		Line2D([0], [0], marker="*", color="black", markersize=10, linestyle="None", label="Ego (0,0)"),
	]
	ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)


def _draw_box(ax: Any, box: Dict[str, Any], edge_color: str, label: str) -> None:
	corners = box_to_bev_corners(box)
	xs = [pt[0] for pt in corners] + [corners[0][0]]
	ys = [pt[1] for pt in corners] + [corners[0][1]]
	ax.plot(xs, ys, color=edge_color, linewidth=1.1, alpha=0.9)
	center = box.get("translation", [0.0, 0.0, 0.0])
	ax.scatter([center[0]], [center[1]], color=edge_color, s=8, alpha=0.8)

	if label == "pred":
		score = box.get("score")
		if score is not None:
			ax.text(center[0], center[1], f"{float(score):.2f}", fontsize=6, color=edge_color)


def _apply_axis_limits(
	ax: Any,
	gt_boxes: Sequence[Dict[str, Any]],
	pred_boxes: Sequence[Dict[str, Any]],
) -> None:
	centers: List[Tuple[float, float]] = []
	for box in list(gt_boxes) + list(pred_boxes):
		translation = box.get("translation", [0.0, 0.0, 0.0])
		centers.append((float(translation[0]), float(translation[1])))

	if not centers:
		ax.set_xlim(-20, 20)
		ax.set_ylim(-20, 20)
		return

	x_values = [center[0] for center in centers]
	y_values = [center[1] for center in centers]
	min_x, max_x = min(x_values), max(x_values)
	min_y, max_y = min(y_values), max(y_values)
	margin = 8.0
	ax.set_xlim(min_x - margin, max_x + margin)
	ax.set_ylim(min_y - margin, max_y + margin)

