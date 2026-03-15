from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.perception.regression_fixture import summarize_perception_regression_case


GOLDEN_FILE = Path(__file__).resolve().parents[1] / "golden" / "perception_metrics_golden.json"


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
        if actual != pytest.approx(expected, abs=atol):
            out.append(f"{path}: expected {expected}, got {actual}")
        return

    if actual != expected:
        out.append(f"{path}: expected {expected!r}, got {actual!r}")


def _assert_nested_close(actual: Any, expected: Any, atol: float = 1e-9) -> None:
    errors: list[str] = []
    _collect_nested_diff(actual, expected, path="", atol=atol, out=errors)
    if errors:
        pytest.fail("\n".join(errors))


def test_perception_metrics_match_golden_snapshot() -> None:
    actual = summarize_perception_regression_case()
    expected = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    _assert_nested_close(actual, expected)
