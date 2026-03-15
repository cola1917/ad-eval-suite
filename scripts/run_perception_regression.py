from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from eval.perception.regression_fixture import summarize_perception_regression_case
except ImportError:  # pragma: no cover
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from eval.perception.regression_fixture import summarize_perception_regression_case


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


KEY_METRIC_PATHS: tuple[str, ...] = (
    "detection.map",
    "detection.overall.precision",
    "detection.overall.recall",
    "detection.overall.f1",
    "tracking.overall.mota",
    "tracking.overall.motp",
    "tracking.overall.idsw",
)


def _get_nested_value(data: dict[str, Any], dot_path: str) -> Any:
    value: Any = data
    for key in dot_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _build_key_metric_diff_table(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    header = f"{'metric':<32} {'expected':>14} {'actual':>14} {'delta':>14}"
    rows.append(header)
    rows.append("-" * len(header))

    for metric_path in KEY_METRIC_PATHS:
        expected_value = _get_nested_value(expected, metric_path)
        actual_value = _get_nested_value(actual, metric_path)
        if expected_value is None and actual_value is None:
            continue

        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            delta = float(actual_value) - float(expected_value)
            rows.append(
                f"{metric_path:<32} {float(expected_value):>14.8f} {float(actual_value):>14.8f} {delta:>14.8f}"
            )
        else:
            rows.append(f"{metric_path:<32} {str(expected_value):>14} {str(actual_value):>14} {'n/a':>14}")

    return rows


def _collect_nested_diff(actual: Any, expected: Any, path: str, atol: float, out: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            out.append(f"{path}: expected dict, got {type(actual).__name__}")
            return

        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            if missing:
                out.append(f"{path}: missing keys={missing}")
            if extra:
                out.append(f"{path}: extra keys={extra}")

        for key in expected:
            if key in actual:
                child_path = f"{path}.{key}" if path else key
                _collect_nested_diff(actual[key], expected[key], child_path, atol=atol, out=out)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            out.append(f"{path}: expected list, got {type(actual).__name__}")
            return
        if len(actual) != len(expected):
            out.append(f"{path}: expected len={len(expected)}, got len={len(actual)}")
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            child_path = f"{path}[{index}]"
            _collect_nested_diff(actual_item, expected_item, child_path, atol=atol, out=out)
        return

    if isinstance(expected, float):
        if abs(float(actual) - expected) > atol:
            out.append(f"{path}: expected {expected}, got {actual}")
        return

    if actual != expected:
        out.append(f"{path}: expected {expected!r}, got {actual!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Perception metrics golden checker")
    parser.add_argument(
        "--golden-file",
        type=Path,
        default=Path("tests/golden/perception_metrics_golden.json"),
        help="Path to perception golden JSON file",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for float comparisons",
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Overwrite golden file with current computed regression summary",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    actual = summarize_perception_regression_case()

    if args.update_golden:
        args.golden_file.parent.mkdir(parents=True, exist_ok=True)
        args.golden_file.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        print(f"[perception-regression] updated golden at {args.golden_file}")
        return 0

    if not args.golden_file.exists():
        print(
            f"[perception-regression] golden file not found: {args.golden_file}. "
            "Run with --update-golden to create it.",
            file=sys.stderr,
        )
        return 2

    expected = _load_json(args.golden_file)
    errors: list[str] = []
    _collect_nested_diff(actual, expected, path="", atol=args.atol, out=errors)

    if errors:
        print("[perception-regression] mismatch detected:", file=sys.stderr)
        print("[perception-regression] key metric diff:", file=sys.stderr)
        for line in _build_key_metric_diff_table(actual, expected):
            print(f"  {line}", file=sys.stderr)
        print("[perception-regression] detailed diffs:", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("[perception-regression] golden check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
