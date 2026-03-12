from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
	from datasets.nuscenes_loader import NuScenesLoader
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from datasets.nuscenes_loader import NuScenesLoader


PredictionRecord = Dict[str, Any]


@dataclass(frozen=True)
class DetectionGeneratorConfig:
	translation_noise_std: float = 0.35
	size_noise_std: float = 0.08
	yaw_noise_std: float = 0.05
	score_mean: float = 0.86
	score_std: float = 0.08
	score_min: float = 0.05
	score_max: float = 0.99
	drop_rate: float = 0.1
	fp_rate: float = 0.15
	min_box_size: float = 0.1
	false_positive_range: float = 70.0
	id_switch_rate: float = 0.02


class DetectionGenerator:
	"""Generate synthetic detection predictions from GT frame records.

	The generator simulates an imperfect detector by perturbing GT boxes,
	assigning confidence scores, randomly dropping boxes, and adding false
	positives. The output schema is suitable for downstream detection evaluators.
	"""

	def __init__(
		self,
		config: Optional[DetectionGeneratorConfig] = None,
		seed: Optional[int] = None,
	) -> None:
		self.config = config or DetectionGeneratorConfig()
		self.rng = np.random.default_rng(seed)

	def generate_frame_predictions(self, frame_record: Dict[str, Any]) -> PredictionRecord:
		gt_boxes = frame_record.get("gt_boxes", [])
		ego_translation = frame_record.get("ego_pose", {}).get("translation", [0.0, 0.0, 0.0])

		pred_boxes: List[Dict[str, Any]] = []
		surviving_gt_boxes = 0
		for gt_box in gt_boxes:
			if self._should_drop():
				continue

			pred_boxes.append(self._generate_prediction_from_gt(gt_box))
			surviving_gt_boxes += 1

		pred_boxes.extend(self._generate_false_positives(gt_boxes, ego_translation))
		pred_boxes.sort(key=lambda box: box["score"], reverse=True)

		return {
			"scene_name": frame_record.get("scene_name"),
			"scene_token": frame_record.get("scene_token"),
			"sample_token": frame_record.get("sample_token"),
			"timestamp": frame_record.get("timestamp"),
			"num_gt_boxes": len(gt_boxes),
			"num_pred_boxes": len(pred_boxes),
			"num_dropped_gt": len(gt_boxes) - surviving_gt_boxes,
			"pred_boxes": pred_boxes,
		}

	def _generate_prediction_from_gt(self, gt_box: Dict[str, Any]) -> Dict[str, Any]:
		translation = list(gt_box["translation"])
		translation[0] += self.rng.normal(0.0, self.config.translation_noise_std)
		translation[1] += self.rng.normal(0.0, self.config.translation_noise_std)
		if len(translation) > 2:
			translation[2] += self.rng.normal(0.0, self.config.translation_noise_std * 0.15)

		size = [
			max(self.config.min_box_size, dimension + self.rng.normal(0.0, self.config.size_noise_std))
			for dimension in gt_box["size"]
		]

		yaw = float(gt_box["yaw"] + self.rng.normal(0.0, self.config.yaw_noise_std))
		score = self._generate_score(gt_box, translation)
		track_id = str(gt_box.get("track_id") or gt_box.get("instance_token") or "")
		if track_id and self.rng.random() < self.config.id_switch_rate:
			track_id = f"sw_{track_id}_{int(self.rng.integers(1000, 9999))}"

		pred_box = {
			"category_name": gt_box["category_name"],
			"translation": translation,
			"size": size,
			"rotation": self._quaternion_from_yaw(yaw),
			"yaw": yaw,
			"score": score,
			"velocity": copy.deepcopy(gt_box.get("velocity", [0.0, 0.0])),
			"attribute_names": copy.deepcopy(gt_box.get("attribute_names", [])),
			"num_lidar_pts": gt_box.get("num_lidar_pts", 0),
			"num_radar_pts": gt_box.get("num_radar_pts", 0),
			"distance_to_ego": gt_box.get("distance_to_ego"),
			"source_gt_annotation_token": gt_box.get("annotation_token"),
			"source_gt_instance_token": gt_box.get("instance_token"),
			"track_id": track_id,
			"is_false_positive": False,
		}
		return pred_box

	def _generate_false_positives(
		self,
		gt_boxes: Sequence[Dict[str, Any]],
		ego_translation: Sequence[float],
	) -> List[Dict[str, Any]]:
		if not gt_boxes:
			return []

		num_fp = int(self.rng.binomial(len(gt_boxes), self.config.fp_rate))
		if num_fp == 0:
			return []

		categories = [box["category_name"] for box in gt_boxes]
		fp_boxes: List[Dict[str, Any]] = []
		for _ in range(num_fp):
			anchor_box = gt_boxes[int(self.rng.integers(0, len(gt_boxes)))]
			radius = float(self.rng.uniform(5.0, self.config.false_positive_range))
			angle = float(self.rng.uniform(-math.pi, math.pi))

			translation = [
				float(ego_translation[0] + radius * math.cos(angle)),
				float(ego_translation[1] + radius * math.sin(angle)),
				float(anchor_box["translation"][2] + self.rng.normal(0.0, 0.5)),
			]
			yaw = float(self.rng.uniform(-math.pi, math.pi))
			size = [
				max(self.config.min_box_size, dimension + self.rng.normal(0.0, self.config.size_noise_std * 2.0))
				for dimension in anchor_box["size"]
			]
			fp_boxes.append(
				{
					"category_name": str(categories[int(self.rng.integers(0, len(categories)))]),
					"translation": translation,
					"size": size,
					"rotation": self._quaternion_from_yaw(yaw),
					"yaw": yaw,
					"score": float(self.rng.uniform(0.05, min(0.6, self.config.score_mean))),
					"velocity": [0.0, 0.0],
					"attribute_names": [],
					"num_lidar_pts": 0,
					"num_radar_pts": 0,
					"distance_to_ego": self._distance_xy(translation, ego_translation),
					"source_gt_annotation_token": None,
					"source_gt_instance_token": None,
					"track_id": f"fp_{int(self.rng.integers(1_000_000, 9_999_999))}",
					"is_false_positive": True,
				}
			)

		return fp_boxes

	def _should_drop(self) -> bool:
		return bool(self.rng.random() < self.config.drop_rate)

	def _generate_score(self, gt_box: Dict[str, Any], pred_translation: Sequence[float]) -> float:
		base_score = self.rng.normal(self.config.score_mean, self.config.score_std)
		localization_error = self._distance_xy(pred_translation, gt_box["translation"])
		distance_penalty = min(0.25, float(gt_box.get("distance_to_ego", 0.0)) / 200.0)
		score = base_score - 0.12 * localization_error - distance_penalty
		return float(np.clip(score, self.config.score_min, self.config.score_max))

	@staticmethod
	def _quaternion_from_yaw(yaw: float) -> List[float]:
		return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

	@staticmethod
	def _distance_xy(position_a: Sequence[float], position_b: Sequence[float]) -> float:
		return math.hypot(position_a[0] - position_b[0], position_a[1] - position_b[1])


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Synthetic detection generator self-test")
	parser.add_argument("--data-root", default="data/nuscenes-mini", help="Path to nuScenes data root")
	parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
	parser.add_argument("--max-frames", type=int, default=2, help="Frames to generate for smoke test")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument("--drop-rate", type=float, default=0.1, help="Probability of dropping each GT box")
	parser.add_argument("--fp-rate", type=float, default=0.15, help="False positive rate per GT box")
	parser.add_argument(
		"--translation-noise-std",
		type=float,
		default=0.35,
		help="Stddev for center perturbation in meters",
	)
	return parser


if __name__ == "__main__":
	args = _build_arg_parser().parse_args()

	config = DetectionGeneratorConfig(
		translation_noise_std=args.translation_noise_std,
		drop_rate=args.drop_rate,
		fp_rate=args.fp_rate,
	)
	generator = DetectionGenerator(config=config, seed=args.seed)

	loader = NuScenesLoader(data_root=args.data_root, version=args.version, verbose=False)
	loader.load()
	first_scene_id = loader.get_scene_ids()[0]

	print(f"[self-test] detection generator seed={args.seed} scene={first_scene_id}")
	for frame_index, frame_record in enumerate(
		loader.iter_frame_records(scene_id=first_scene_id, max_frames=args.max_frames),
		start=1,
	):
		prediction_record = generator.generate_frame_predictions(frame_record)
		print(
			f"[self-test] frame {frame_index}: "
			f"gt={prediction_record['num_gt_boxes']} "
			f"pred={prediction_record['num_pred_boxes']} "
			f"dropped={prediction_record['num_dropped_gt']}"
		)
		if prediction_record["pred_boxes"]:
			top_pred = prediction_record["pred_boxes"][0]
			print(
				"[self-test] top pred: "
				f"category={top_pred['category_name']} "
				f"score={top_pred['score']:.2f} "
				f"fp={top_pred['is_false_positive']}"
			)

	loader.close()
	print("[self-test] done")
