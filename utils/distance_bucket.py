from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_BUCKET_BOUNDARIES = (20.0, 40.0)


def assign_distance_bucket(
	distance: float,
	boundaries: Tuple[float, float] = DEFAULT_BUCKET_BOUNDARIES,
) -> str:
	near_boundary, far_boundary = boundaries
	if distance < near_boundary:
		return "near"
	if distance < far_boundary:
		return "medium"
	return "far"


def box_distance_to_ego(box: Dict[str, Any]) -> float:
	distance = box.get("distance_to_ego")
	if distance is None:
		translation = box.get("translation", [0.0, 0.0, 0.0])
		distance = (translation[0] ** 2 + translation[1] ** 2) ** 0.5
	return float(distance)


def bucketize_boxes(
	boxes: Iterable[Dict[str, Any]],
	boundaries: Tuple[float, float] = DEFAULT_BUCKET_BOUNDARIES,
) -> Dict[str, List[Dict[str, Any]]]:
	buckets: Dict[str, List[Dict[str, Any]]] = {"near": [], "medium": [], "far": []}
	for box in boxes:
		bucket = assign_distance_bucket(box_distance_to_ego(box), boundaries)
		buckets[bucket].append(box)
	return buckets


def bucket_label_with_ranges(boundaries: Tuple[float, float]) -> Dict[str, str]:
	near_boundary, far_boundary = boundaries
	return {
		"near": f"near(<{near_boundary:g}m)",
		"medium": f"medium([{near_boundary:g},{far_boundary:g})m)",
		"far": f"far(>={far_boundary:g}m)",
	}


if __name__ == "__main__":
	boxes = [
		{"distance_to_ego": 10.0},
		{"distance_to_ego": 27.0},
		{"distance_to_ego": 51.0},
	]
	print(f"[self-test] buckets={ {name: len(items) for name, items in bucketize_boxes(boxes).items()} }")
