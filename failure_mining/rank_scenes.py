from __future__ import annotations

from typing import Any, Dict, List, Sequence


def top_k_scenes(scene_rows: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
	if k <= 0:
		return []
	return list(scene_rows[: min(k, len(scene_rows))])
