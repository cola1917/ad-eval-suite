from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, List, Optional

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
			sample_token = scene["first_sample_token"]
			while sample_token:
				sample = self.nusc.get("sample", sample_token)

				# Keep this schema stable as a cross-dataset internal contract.
				record: Dict[str, Any] = {
					"scene_name": scene["name"],
					"scene_token": scene["token"],
					"sample_token": sample["token"],
					"timestamp": sample["timestamp"],
					"num_annotations": len(sample["anns"]),
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
			print(
				f"[self-test] frame {idx}: sample={frame['sample_token']} "
				f"timestamp={frame['timestamp']} anns={frame['num_annotations']}"
			)

	loader.close()
	print("[self-test] done")
