from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional


FrameRecord = Dict[str, Any]


class BaseDataset(ABC):
	"""Abstract dataset contract for all evaluation stages.

	Concrete loaders (nuScenes, Waymo, KITTI, etc.) should implement this
	interface and normalize outputs so evaluation modules can stay dataset-agnostic.
	"""

	def __init__(self, data_root: str, split: str = "mini") -> None:
		self.data_root = data_root
		self.split = split
		self._is_loaded = False

	@property
	def is_loaded(self) -> bool:
		return self._is_loaded

	@property
	@abstractmethod
	def dataset_name(self) -> str:
		"""Human-readable dataset name."""

	@abstractmethod
	def load(self) -> None:
		"""Load and initialize dataset handles/indexes."""

	@abstractmethod
	def get_scene_ids(self) -> List[str]:
		"""Return scene identifiers for the current split."""

	@abstractmethod
	def iter_frame_records(
		self, scene_id: Optional[str] = None, max_frames: Optional[int] = None
	) -> Iterable[FrameRecord]:
		"""Yield normalized frame records.

		Each frame record should be a dict containing consistent keys expected by
		evaluation modules (timestamp, pose, annotations, predictions, etc.).
		"""

	@abstractmethod
	def close(self) -> None:
		"""Release resources and clear cached handles."""
