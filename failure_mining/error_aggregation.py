from __future__ import annotations

from typing import Any, Dict, Sequence


def aggregate_error_categories(frame_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	fp_by_class: Dict[str, int] = {}
	fn_by_class: Dict[str, int] = {}

	for row in frame_rows:
		match_result = row.get("match_result", {})
		for fp_entry in match_result.get("false_positives", []):
			pred_box = fp_entry.get("pred_box", {})
			name = str(pred_box.get("category_name", "unknown"))
			fp_by_class[name] = fp_by_class.get(name, 0) + 1
		for fn_entry in match_result.get("false_negatives", []):
			gt_box = fn_entry.get("gt_box", {})
			name = str(gt_box.get("category_name", "unknown"))
			fn_by_class[name] = fn_by_class.get(name, 0) + 1

	return {
		"fp_by_class": dict(sorted(fp_by_class.items(), key=lambda item: item[1], reverse=True)),
		"fn_by_class": dict(sorted(fn_by_class.items(), key=lambda item: item[1], reverse=True)),
	}
