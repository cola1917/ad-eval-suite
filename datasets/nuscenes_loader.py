from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Iterable, List, Optional

try:
	from .base_dataset import BaseDataset, FrameRecord
except ImportError:  # pragma: no cover
	from base_dataset import BaseDataset, FrameRecord


class NuScenesLoader(BaseDataset):
	"""nuScenes implementation of BaseDataset.

	This class intentionally keeps the output schema lightweight for now and
	provides a stable import/interface layer for downstream evaluators.
	"""

	def __init__(
		self,
		data_root: str,
		version: str = "v1.0-mini",
		split: str = "mini",
		verbose: bool = False,
	) -> None:
		super().__init__(data_root=data_root, split=split)
		self.version = version
		self.verbose = verbose
		self.nusc: Any = None

	@property
	def dataset_name(self) -> str:
		return "nuScenes"

	def load(self) -> None:
		try:
			from nuscenes.nuscenes import NuScenes
		except Exception as exc:  # pragma: no cover
			raise ImportError(
				"nuscenes-devkit is required. Install with `pip install nuscenes-devkit`."
			) from exc

		self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=self.verbose)
		self._is_loaded = True

	def get_scene_ids(self) -> List[str]:
		self._ensure_loaded()
		assert self.nusc is not None
		return [scene["name"] for scene in self.nusc.scene]

	def iter_frame_records(
		self, scene_id: Optional[str] = None, max_frames: Optional[int] = None
	) -> Iterable[FrameRecord]:
		self._ensure_loaded()
		assert self.nusc is not None

		scenes = [self._resolve_scene(scene_id)] if scene_id else list(self.nusc.scene)

		yielded = 0
		for scene in scenes:
			log_record = self.nusc.get("log", scene["log_token"])
			sample_token = scene["first_sample_token"]
			while sample_token:
				sample = self.nusc.get("sample", sample_token)
				ego_pose = self._build_ego_pose(sample)
				gt_boxes = self._build_gt_boxes(sample, ego_pose)

				record: Dict[str, Any] = {
					"scene_name": scene["name"],
					"scene_token": scene["token"],
					"scene_description": scene.get("description", ""),
					"location": log_record.get("location", ""),
					"logfile": log_record.get("logfile", ""),
					"sample_token": sample["token"],
					"timestamp": sample["timestamp"],
					"sample_prev_token": sample["prev"],
					"sample_next_token": sample["next"],
					"ego_pose": ego_pose,
					"sensor_data": self._build_sensor_data(sample),
					"gt_boxes": gt_boxes,
					"num_annotations": len(gt_boxes),
					"annotations": sample["anns"],
					"data": sample["data"],
				}
				yield record

				yielded += 1
				if max_frames is not None and yielded >= max_frames:
					return

				sample_token = sample["next"]

	def close(self) -> None:
		self.nusc = None
		self._is_loaded = False

	def _ensure_loaded(self) -> None:
		if not self.is_loaded:
			raise RuntimeError("Dataset is not loaded. Call `load()` first.")

	def _resolve_scene(self, scene_id: Optional[str]) -> Dict[str, Any]:
		assert self.nusc is not None
		if scene_id is None:
			raise ValueError("scene_id cannot be None when resolving a scene.")

		for scene in self.nusc.scene:
			if scene["name"] == scene_id or scene["token"] == scene_id:
				return scene
		raise ValueError(f"Scene not found: {scene_id}")

	def _build_ego_pose(self, sample: Dict[str, Any]) -> Dict[str, Any]:
		assert self.nusc is not None
		reference_channel = self._get_reference_channel(sample)
		sample_data = self.nusc.get("sample_data", sample["data"][reference_channel])
		ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

		return {
			"reference_channel": reference_channel,
			"sample_data_token": sample_data["token"],
			"ego_pose_token": sample_data["ego_pose_token"],
			"calibrated_sensor_token": sample_data["calibrated_sensor_token"],
			"translation": list(ego_pose["translation"]),
			"rotation": list(ego_pose["rotation"]),
			"yaw": self._yaw_from_quaternion(ego_pose["rotation"]),
		}

	def _build_sensor_data(self, sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
		assert self.nusc is not None
		sensor_data: Dict[str, Dict[str, Any]] = {}
		for channel, sample_data_token in sample["data"].items():
			sample_data = self.nusc.get("sample_data", sample_data_token)
			calibrated_sensor = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
			sensor = self.nusc.get("sensor", calibrated_sensor["sensor_token"])
			sensor_data[channel] = {
				"sample_data_token": sample_data_token,
				"filename": sample_data["filename"],
				"is_key_frame": sample_data["is_key_frame"],
				"timestamp": sample_data["timestamp"],
				"modality": sensor["modality"],
				"channel": sensor["channel"],
				"translation": list(calibrated_sensor["translation"]),
				"rotation": list(calibrated_sensor["rotation"]),
				"camera_intrinsic": calibrated_sensor.get("camera_intrinsic", []),
			}
		return sensor_data

	def _build_gt_boxes(
		self, sample: Dict[str, Any], ego_pose: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		assert self.nusc is not None
		gt_boxes: List[Dict[str, Any]] = []
		ego_translation = ego_pose["translation"]

		for annotation_token in sample["anns"]:
			annotation = self.nusc.get("sample_annotation", annotation_token)
			instance = self.nusc.get("instance", annotation["instance_token"])
			attribute_names = [
				self.nusc.get("attribute", attribute_token)["name"]
				for attribute_token in annotation["attribute_tokens"]
			]
			velocity = self.nusc.box_velocity(annotation_token)

			translation = list(annotation["translation"])
			gt_boxes.append(
				{
					"annotation_token": annotation_token,
					"sample_token": annotation["sample_token"],
					"instance_token": annotation["instance_token"],
					"track_id": annotation["instance_token"],
					"category_name": annotation["category_name"],
					"translation": translation,
					"size": list(annotation["size"]),
					"rotation": list(annotation["rotation"]),
					"yaw": self._yaw_from_quaternion(annotation["rotation"]),
					"velocity": self._sanitize_velocity(velocity),
					"attribute_names": attribute_names,
					"visibility_token": annotation.get("visibility_token"),
					"num_lidar_pts": annotation.get("num_lidar_pts", 0),
					"num_radar_pts": annotation.get("num_radar_pts", 0),
					"distance_to_ego": self._distance_xy(translation, ego_translation),
					"distance_to_ego_3d": self._distance_xyz(translation, ego_translation),
					"instance_category_token": instance["category_token"],
					"is_moving": self._is_moving(velocity),
				}
			)

		return gt_boxes

	@staticmethod
	def _get_reference_channel(sample: Dict[str, Any]) -> str:
		for channel in ("LIDAR_TOP", "CAM_FRONT"):
			if channel in sample["data"]:
				return channel
		return next(iter(sample["data"]))

	@staticmethod
	def _yaw_from_quaternion(rotation: List[float]) -> float:
		w, x, y, z = rotation
		siny_cosp = 2.0 * (w * z + x * y)
		cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
		return math.atan2(siny_cosp, cosy_cosp)

	@staticmethod
	def _distance_xy(position_a: List[float], position_b: List[float]) -> float:
		return math.hypot(position_a[0] - position_b[0], position_a[1] - position_b[1])

	@staticmethod
	def _distance_xyz(position_a: List[float], position_b: List[float]) -> float:
		return math.sqrt(
			(position_a[0] - position_b[0]) ** 2
			+ (position_a[1] - position_b[1]) ** 2
			+ (position_a[2] - position_b[2]) ** 2
		)

	@staticmethod
	def _sanitize_velocity(velocity: Any) -> List[float]:
		if velocity is None:
			return [0.0, 0.0]
		velocity_x = float(velocity[0]) if not math.isnan(float(velocity[0])) else 0.0
		velocity_y = float(velocity[1]) if not math.isnan(float(velocity[1])) else 0.0
		return [velocity_x, velocity_y]

	@classmethod
	def _is_moving(cls, velocity: Any, threshold: float = 0.2) -> bool:
		velocity_xy = cls._sanitize_velocity(velocity)
		return math.hypot(velocity_xy[0], velocity_xy[1]) >= threshold


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Quick self-test for NuScenesLoader")
	parser.add_argument("--data-root", default="data/nuscenes-mini", help="Path to nuScenes data root")
	parser.add_argument("--version", default="v1.0-mini", help="nuScenes version")
	parser.add_argument("--max-frames", type=int, default=3, help="Frames to print for smoke test")
	return parser


if __name__ == "__main__":
	args = _build_arg_parser().parse_args()

	loader = NuScenesLoader(data_root=args.data_root, version=args.version, verbose=False)
	print(f"[self-test] loading {loader.dataset_name} from: {args.data_root}")
	loader.load()

	scene_ids = loader.get_scene_ids()
	print(f"[self-test] scenes loaded: {len(scene_ids)}")
	if not scene_ids:
		print("[self-test] no scenes found")
	else:
		print(f"[self-test] first scene: {scene_ids[0]}")
		for idx, frame in enumerate(loader.iter_frame_records(scene_id=scene_ids[0], max_frames=args.max_frames), start=1):
			first_box = frame["gt_boxes"][0] if frame["gt_boxes"] else None
			print(
				f"[self-test] frame {idx}: sample={frame['sample_token']} "
				f"timestamp={frame['timestamp']} anns={frame['num_annotations']} "
				f"ref={frame['ego_pose']['reference_channel']}"
			)
			if first_box is not None:
				print(
					"[self-test] first gt: "
					f"category={first_box['category_name']} "
					f"track_id={first_box['track_id']} "
					f"distance_xy={first_box['distance_to_ego']:.2f}m"
				)

	loader.close()
	print("[self-test] done")
