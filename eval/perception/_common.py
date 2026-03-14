"""Shared helpers for perception evaluation modules (detection + tracking)."""

from __future__ import annotations

from typing import Any, Callable, Dict

try:
	from matching.greedy_match import greedy_match_detections
	from matching.hungarian import hungarian_match_detections
except ImportError:  # pragma: no cover
	import sys
	from pathlib import Path
	workspace_root = Path(__file__).resolve().parents[2]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from matching.greedy_match import greedy_match_detections
	from matching.hungarian import hungarian_match_detections


# Callable type for any matcher function (greedy / hungarian / custom).
MatcherFn = Callable[..., Dict[str, Any]]


def _resolve_matcher(matcher: str | MatcherFn) -> MatcherFn:
	"""Return a matcher callable from a name string or an already-callable object."""
	if callable(matcher):
		return matcher
	matcher_name = matcher.lower()
	if matcher_name == "greedy":
		return greedy_match_detections
	if matcher_name == "hungarian":
		return hungarian_match_detections
	raise ValueError(f"Unsupported matcher: {matcher}")


def _infer_scenario_bucket(frame_record: Dict[str, Any]) -> str:
	"""Map a frame record to a coarse scenario category based on free-text fields."""
	description = str(frame_record.get("scene_description", "")).lower()
	location = str(frame_record.get("location", "")).lower()
	text = f"{description} {location}"

	if any(token in text for token in ("rain", "wet", "fog", "snow")):
		return "adverse_weather"
	if "night" in text:
		return "night"
	if any(token in text for token in ("highway", "freeway", "expressway")):
		return "highway"
	if any(token in location for token in ("singapore", "boston")):
		return "urban"
	return "other"
