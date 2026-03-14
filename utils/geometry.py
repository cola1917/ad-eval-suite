"""BEV geometry helpers: coordinate transforms and box corner calculations."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

BoxRecord = Dict[str, Any]


def box_to_bev_corners(box: BoxRecord) -> List[Tuple[float, float]]:
	"""Return the four BEV world-frame corners of a box, accounting for yaw."""
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
