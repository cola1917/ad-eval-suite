from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


# nuScenes visibility levels: 4 (fully visible), 3 (mostly visible), 2 (partially visible), 1 (mostly occluded)
VISIBILITY_LEVELS = {
	"4": "fully_visible",
	"3": "mostly_visible",
	"2": "partially_occluded",
	"1": "mostly_occluded",
}


def assign_occlusion_bucket(
	box: Dict[str, Any],
	visibility_mapping: Dict[str, str] = VISIBILITY_LEVELS,
) -> str:
	"""Assign a box to an occlusion bucket based on its visibility token."""
	visibility_token = box.get("visibility_token")
	if visibility_token is None:
		return "unknown"
	
	visibility_str = str(visibility_token)
	return visibility_mapping.get(visibility_str, "unknown")


def bucketize_boxes_by_occlusion(
	boxes: Iterable[Dict[str, Any]],
	visibility_mapping: Dict[str, str] = VISIBILITY_LEVELS,
) -> Dict[str, List[Dict[str, Any]]]:
	"""Bucketize boxes by occlusion level."""
	buckets: Dict[str, List[Dict[str, Any]]] = {
		"fully_visible": [],
		"mostly_visible": [],
		"partially_occluded": [],
		"mostly_occluded": [],
		"unknown": [],
	}
	for box in boxes:
		bucket = assign_occlusion_bucket(box, visibility_mapping)
		buckets[bucket].append(box)
	return buckets


def occlusion_bucket_labels() -> Dict[str, str]:
	"""Return human-readable labels for occlusion buckets."""
	return {
		"fully_visible": "Fully Visible (visibility=4)",
		"mostly_visible": "Mostly Visible (visibility=3)",
		"partially_occluded": "Partially Occluded (visibility=2)",
		"mostly_occluded": "Mostly Occluded (visibility=1)",
		"unknown": "Unknown",
	}


def bucketize_boxes_by_distance_and_occlusion(
	boxes: Iterable[Dict[str, Any]],
	distance_boundaries: Tuple[float, float] = (20.0, 40.0),
	visibility_mapping: Dict[str, str] = VISIBILITY_LEVELS,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
	"""
	Bucketize boxes by both distance and occlusion level.
	Returns nested dict: {occlusion_level: {distance_level: [boxes]}}
	"""
	try:
		from distance_bucket import assign_distance_bucket
	except ImportError:
		import sys
		from pathlib import Path
		workspace_root = Path(__file__).resolve().parents[1]
		if str(workspace_root) not in sys.path:
			sys.path.insert(0, str(workspace_root))
		from distance_bucket import assign_distance_bucket
	
	occlusion_buckets = {
		"fully_visible": {},
		"mostly_visible": {},
		"partially_occluded": {},
		"mostly_occluded": {},
		"unknown": {},
	}
	
	# Initialize distance buckets for each occlusion level
	for occ_level in occlusion_buckets:
		occlusion_buckets[occ_level] = {"near": [], "medium": [], "far": []}
	
	for box in boxes:
		occlusion_bucket = assign_occlusion_bucket(box, visibility_mapping)
		distance_bucket = assign_distance_bucket(
			box.get("distance_to_ego", 0.0),
			boundaries=distance_boundaries,
		)
		occlusion_buckets[occlusion_bucket][distance_bucket].append(box)
	
	return occlusion_buckets


if __name__ == "__main__":
	# Self-test
	boxes = [
		{"distance_to_ego": 10.0, "visibility_token": "4"},
		{"distance_to_ego": 27.0, "visibility_token": "3"},
		{"distance_to_ego": 51.0, "visibility_token": "1"},
		{"distance_to_ego": 15.0, "visibility_token": "2"},
	]
	
	# Test occlusion bucketing
	occ_buckets = bucketize_boxes_by_occlusion(boxes)
	print(f"[self-test] occlusion buckets: {{{', '.join(f'{name}: {len(items)}' for name, items in occ_buckets.items())}}}")
	
	# Test combined bucketing
	combined = bucketize_boxes_by_distance_and_occlusion(boxes)
	print(f"[self-test] combined buckets:")
	for occ_level, dist_buckets in combined.items():
		count_str = ", ".join(f"{dist}: {len(boxes)}" for dist, boxes in dist_buckets.items() if boxes)
		if count_str:
			print(f"  {occ_level}: {{{count_str}}}")
