from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Dict, List, Sequence, Tuple

try:
	from shapely.geometry import Polygon
except Exception:  # pragma: no cover
	Polygon = None


BoxRecord = Dict[str, Any]


def bev_iou(box_a: BoxRecord, box_b: BoxRecord) -> float:
	"""Compute BEV IoU. Uses yaw-aware polygons when Shapely is available."""

	if Polygon is not None:
		poly_a = _polygon_from_key(_box_cache_key(box_a))
		poly_b = _polygon_from_key(_box_cache_key(box_b))
		if poly_a is None or poly_b is None:
			return _axis_aligned_bev_iou(box_a, box_b)
		if not poly_a.is_valid or not poly_b.is_valid:
			return _axis_aligned_bev_iou(box_a, box_b)
		intersection = poly_a.intersection(poly_b).area
		if intersection <= 0.0:
			return 0.0
		union = poly_a.area + poly_b.area - intersection
		if union <= 0.0:
			return 0.0
		return float(intersection / union)

	return _axis_aligned_bev_iou(box_a, box_b)


def pairwise_iou_matrix(gt_boxes: Sequence[BoxRecord], pred_boxes: Sequence[BoxRecord]) -> List[List[float]]:
	return [[bev_iou(gt_box, pred_box) for pred_box in pred_boxes] for gt_box in gt_boxes]


def center_distance(box_a: BoxRecord, box_b: BoxRecord) -> float:
	translation_a = box_a.get("translation", [0.0, 0.0])
	translation_b = box_b.get("translation", [0.0, 0.0])
	return math.hypot(translation_a[0] - translation_b[0], translation_a[1] - translation_b[1])


def box_to_bev_corners(box: BoxRecord) -> List[Tuple[float, float]]:
	translation = box.get("translation", [0.0, 0.0, 0.0])
	size = box.get("size", [0.0, 0.0, 0.0])
	x = float(translation[0]) if len(translation) > 0 else 0.0
	y = float(translation[1]) if len(translation) > 1 else 0.0
	width = float(size[0]) if len(size) > 0 else 0.0
	length = float(size[1]) if len(size) > 1 else width
	yaw = float(box.get("yaw", 0.0))

	half_w = width / 2.0
	half_l = length / 2.0
	local_corners = [
		(half_l, half_w),
		(half_l, -half_w),
		(-half_l, -half_w),
		(-half_l, half_w),
	]

	cos_yaw = math.cos(yaw)
	sin_yaw = math.sin(yaw)
	world_corners: List[Tuple[float, float]] = []
	for lx, ly in local_corners:
		wx = x + lx * cos_yaw - ly * sin_yaw
		wy = y + lx * sin_yaw + ly * cos_yaw
		world_corners.append((wx, wy))
	return world_corners


def _box_cache_key(box: BoxRecord) -> Tuple[float, float, float, float, float]:
	translation = box.get("translation", [0.0, 0.0, 0.0])
	size = box.get("size", [0.0, 0.0, 0.0])
	yaw = float(box.get("yaw", 0.0))
	x = float(translation[0]) if len(translation) > 0 else 0.0
	y = float(translation[1]) if len(translation) > 1 else 0.0
	width = float(size[0]) if len(size) > 0 else 0.0
	length = float(size[1]) if len(size) > 1 else width
	# Rounded keys keep cache size bounded while preserving enough geometric precision.
	return (round(x, 3), round(y, 3), round(width, 3), round(length, 3), round(yaw, 4))


@lru_cache(maxsize=200000)
def _polygon_from_key(key: Tuple[float, float, float, float, float]) -> Any:
	if Polygon is None:
		return None
	x, y, width, length, yaw = key
	half_w = width / 2.0
	half_l = length / 2.0
	local_corners = [
		(half_l, half_w),
		(half_l, -half_w),
		(-half_l, -half_w),
		(-half_l, half_w),
	]
	cos_yaw = math.cos(yaw)
	sin_yaw = math.sin(yaw)
	world_corners = []
	for lx, ly in local_corners:
		world_corners.append((x + lx * cos_yaw - ly * sin_yaw, y + lx * sin_yaw + ly * cos_yaw))
	return Polygon(world_corners)


def _axis_aligned_bev_iou(box_a: BoxRecord, box_b: BoxRecord) -> float:
	min_ax, min_ay, max_ax, max_ay = _bev_bounds(box_a)
	min_bx, min_by, max_bx, max_by = _bev_bounds(box_b)

	inter_min_x = max(min_ax, min_bx)
	inter_min_y = max(min_ay, min_by)
	inter_max_x = min(max_ax, max_bx)
	inter_max_y = min(max_ay, max_by)

	inter_width = max(0.0, inter_max_x - inter_min_x)
	inter_height = max(0.0, inter_max_y - inter_min_y)
	intersection = inter_width * inter_height
	if intersection <= 0.0:
		return 0.0

	area_a = max(0.0, max_ax - min_ax) * max(0.0, max_ay - min_ay)
	area_b = max(0.0, max_bx - min_bx) * max(0.0, max_by - min_by)
	union = area_a + area_b - intersection
	if union <= 0.0:
		return 0.0
	return intersection / union


def _bev_bounds(box: BoxRecord) -> Tuple[float, float, float, float]:
	translation = box.get("translation", [0.0, 0.0, 0.0])
	size = box.get("size", [0.0, 0.0, 0.0])
	width = float(size[0]) if len(size) > 0 else 0.0
	length = float(size[1]) if len(size) > 1 else width
	center_x = float(translation[0]) if len(translation) > 0 else 0.0
	center_y = float(translation[1]) if len(translation) > 1 else 0.0
	half_width = width / 2.0
	half_length = length / 2.0
	return (
		center_x - half_width,
		center_y - half_length,
		center_x + half_width,
		center_y + half_length,
	)


if __name__ == "__main__":
	gt_box = {"translation": [10.0, 5.0, 0.0], "size": [2.0, 4.0, 1.5]}
	pred_box = {"translation": [10.3, 5.1, 0.0], "size": [2.0, 4.0, 1.5]}
	print(f"[self-test] iou={bev_iou(gt_box, pred_box):.4f}")
	print(f"[self-test] distance={center_distance(gt_box, pred_box):.4f}")
