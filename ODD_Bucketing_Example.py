"""
Example script demonstrating how to use occlusion (ODD) bucketing with distance bucketing.

ODD (Operational Design Domain) bucketing in this context refers to occlusion-based bucketing
that categorizes objects by their visibility level (fully visible, mostly visible, etc.).

This script shows three approaches:
1. Distance bucketing only (existing)
2. Occlusion bucketing only (new)
3. Combined occlusion + distance bucketing (new)
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.detection_eval import evaluate_detection_frames
	from generators.detection_generator import DetectionGenerator
	from utils.distance_bucket import DEFAULT_BUCKET_BOUNDARIES
except ImportError:
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from datasets.nuscenes_loader import NuScenesLoader
	from eval.perception.detection_eval import evaluate_detection_frames
	from generators.detection_generator import DetectionGenerator
	from utils.distance_bucket import DEFAULT_BUCKET_BOUNDARIES


def example_distance_bucketing_only():
	"""Example 1: Distance bucketing only (existing functionality)."""
	print("\n" + "="*60)
	print("EXAMPLE 1: Distance Bucketing Only")
	print("="*60)
	
	loader = NuScenesLoader(data_root="data/nuscenes-mini", version="v1.0-mini", verbose=False)
	loader.load()
	scene_id = loader.get_scene_ids()[0]
	
	frame_records = list(loader.iter_frame_records(scene_id=scene_id, max_frames=5))
	generator = DetectionGenerator(seed=42)
	prediction_records = [generator.generate_frame_predictions(frame_record) for frame_record in frame_records]
	
	result = evaluate_detection_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=0.5,
		use_occlusion_bucketing=False,
		use_combined_bucketing=False,
	)
	
	print(f"\nscene: {scene_id}")
	print(f"frames: {result['num_frames']}")
	print(f"\nDistance Buckets:")
	for bucket_name, metrics in result["distance_buckets"].items():
		label = result["distance_bucket_labels"][bucket_name]
		print(f"  {label}:")
		print(f"    TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
		print(f"    P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
	
	loader.close()


def example_occlusion_bucketing_only():
	"""Example 2: Occlusion bucketing only (new functionality)."""
	print("\n" + "="*60)
	print("EXAMPLE 2: Occlusion Bucketing Only")
	print("="*60)
	
	loader = NuScenesLoader(data_root="data/nuscenes-mini", version="v1.0-mini", verbose=False)
	loader.load()
	scene_id = loader.get_scene_ids()[0]
	
	frame_records = list(loader.iter_frame_records(scene_id=scene_id, max_frames=5))
	generator = DetectionGenerator(seed=42)
	prediction_records = [generator.generate_frame_predictions(frame_record) for frame_record in frame_records]
	
	result = evaluate_detection_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=0.5,
		use_occlusion_bucketing=True,
		use_combined_bucketing=False,
	)
	
	print(f"\nscene: {scene_id}")
	print(f"frames: {result['num_frames']}")
	print(f"\nOcclusion Buckets (Visibility Levels):")
	for bucket_name, metrics in result["occlusion_buckets"].items():
		label = result["occlusion_bucket_labels"][bucket_name]
		print(f"  {label}:")
		print(f"    TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
		print(f"    P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
	
	loader.close()


def example_combined_bucketing():
	"""Example 3: Combined occlusion + distance bucketing (new functionality)."""
	print("\n" + "="*60)
	print("EXAMPLE 3: Combined Occlusion + Distance Bucketing")
	print("="*60)
	print("This shows detection performance across all combinations of:")
	print("  - Occlusion levels: fully_visible, mostly_visible, partially_occluded, mostly_occluded")
	print("  - Distance levels: near (<20m), medium (20-40m), far (>40m)")
	
	loader = NuScenesLoader(data_root="data/nuscenes-mini", version="v1.0-mini", verbose=False)
	loader.load()
	scene_id = loader.get_scene_ids()[0]
	
	frame_records = list(loader.iter_frame_records(scene_id=scene_id, max_frames=5))
	generator = DetectionGenerator(seed=42)
	prediction_records = [generator.generate_frame_predictions(frame_record) for frame_record in frame_records]
	
	result = evaluate_detection_frames(
		frame_records=frame_records,
		prediction_records=prediction_records,
		iou_threshold=0.5,
		use_occlusion_bucketing=False,
		use_combined_bucketing=True,
	)
	
	print(f"\nscene: {scene_id}")
	print(f"frames: {result['num_frames']}")
	print(f"\nCombined Buckets (Occlusion x Distance):")
	
	occ_labels = result["combined_bucket_labels"]["occlusion"]
	dist_labels = result["combined_bucket_labels"]["distance"]
	
	for occ_level, dist_buckets in result["occlusion_distance_buckets"].items():
		occ_label = occ_labels[occ_level]
		print(f"\n  {occ_label}:")
		for dist_name, metrics in dist_buckets.items():
			dist_label = dist_labels[dist_name]
			tp = metrics['tp']
			if tp > 0 or metrics['fp'] > 0 or metrics['fn'] > 0:
				print(f"    {dist_label}:")
				print(f"      TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
				print(f"      P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
	
	loader.close()


def main():
	"""Run all examples."""
	print("\n" + "="*60)
	print("Occlusion (ODD) Bucketing Examples")
	print("="*60)
	
	try:
		example_distance_bucketing_only()
	except Exception as e:
		print(f"\nExample 1 error: {e}")
	
	try:
		example_occlusion_bucketing_only()
	except Exception as e:
		print(f"\nExample 2 error: {e}")
	
	try:
		example_combined_bucketing()
	except Exception as e:
		print(f"\nExample 3 error: {e}")
	
	print("\n" + "="*60)
	print("Examples completed!")
	print("="*60)


if __name__ == "__main__":
	main()
