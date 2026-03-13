"""Category remapping utilities.

Reads the ``category_schemas`` block from ``dataset.yaml`` and applies a
chosen schema to ``gt_boxes`` / ``pred_boxes`` lists so every downstream
evaluator works on a consistent, strategy-specific set of category names.

Supported schemas (defined in ``configs/dataset.yaml``)
────────────────────────────────────────────────────────
  raw              – 23 original nuScenes classes, no change
  detection_10cls  – NuScenes 10-class benchmark (ignore unlisted boxes)
  l2_planning      – 4 planning-level classes: vehicle / pedestrian /
                     cyclist / object

Quick-start
───────────
    # Load a schema directly from dataset.yaml
    from utils.category_remap import CategoryRemapper
    remapper = CategoryRemapper.from_config("configs/dataset.yaml", schema="detection_10cls")
    frames   = list(remapper.apply_to_frames(frame_records))
    preds    = list(remapper.apply_to_predictions(prediction_records))

    # Or load the active strategy straight from eval.yaml
    from utils.category_remap import build_remapper_from_eval_config
    remapper = build_remapper_from_eval_config("configs/eval.yaml")          # uses active_strategy
    remapper = build_remapper_from_eval_config("configs/eval.yaml", strategy="l2_planning")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc


class CategoryRemapper:
    """Applies a category taxonomy schema to boxes inside frame/prediction records.

    Parameters
    ----------
    mode : str
        ``"raw"`` (no-op) or ``"remap"``.
    raw_to_target : dict
        Reverse index mapping original nuScenes ``category_name`` → target
        class name.  Only used when *mode* is ``"remap"``.
    ignore_unlisted : bool
        When ``True``, boxes whose ``category_name`` is not present in
        ``raw_to_target`` are dropped.  When ``False`` they are kept under
        their original name.
    """

    def __init__(
        self,
        mode: str,
        raw_to_target: Dict[str, str],
        ignore_unlisted: bool = True,
    ) -> None:
        self._mode = mode
        self._raw_to_target = raw_to_target
        self._ignore_unlisted = ignore_unlisted

    # ── class-level constructors ──────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path, schema: str) -> "CategoryRemapper":
        """Build a remapper from a ``dataset.yaml`` file.

        Parameters
        ----------
        config_path : str | Path
            Path to ``dataset.yaml``.
        schema : str
            One of ``"raw"``, ``"detection_10cls"``, or ``"l2_planning"``.
        """
        config = _load_yaml(config_path)
        schemas: Dict[str, Any] = config.get("category_schemas", {})
        if schema not in schemas:
            available = list(schemas.keys())
            raise ValueError(
                f"Unknown category schema {schema!r}. "
                f"Available schemas: {available}"
            )
        return cls._from_schema_config(schemas[schema])

    @classmethod
    def _from_schema_config(cls, schema_cfg: Dict[str, Any]) -> "CategoryRemapper":
        mode: str = schema_cfg.get("mode", "raw")
        if mode == "raw":
            return cls(mode="raw", raw_to_target={}, ignore_unlisted=False)

        mapping: Dict[str, List[str]] = schema_cfg.get("mapping", {})
        ignore_unlisted: bool = schema_cfg.get("ignore_unlisted", True)

        # Build reverse index: raw_name → target_class
        raw_to_target: Dict[str, str] = {}
        for target_class, raw_list in mapping.items():
            for raw_name in raw_list:
                raw_to_target[raw_name] = target_class

        return cls(
            mode="remap",
            raw_to_target=raw_to_target,
            ignore_unlisted=ignore_unlisted,
        )

    # ── read-only properties ──────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """``"raw"`` or ``"remap"``."""
        return self._mode

    @property
    def target_classes(self) -> List[str]:
        """Sorted list of target category names (empty when mode is ``"raw"``)."""
        return sorted(set(self._raw_to_target.values()))

    # ── core API ──────────────────────────────────────────────────────────────

    def remap_category(self, raw_name: str) -> Optional[str]:
        """Return the target class for *raw_name*, or ``None`` if the box should be dropped.

        Returns *raw_name* unchanged when mode is ``"raw"`` or when
        *ignore_unlisted* is ``False`` and the name has no mapping.
        """
        if self._mode == "raw":
            return raw_name
        target = self._raw_to_target.get(raw_name)
        if target is None and not self._ignore_unlisted:
            return raw_name  # keep under original name
        return target  # None → drop this box

    def remap_boxes(self, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a new list of boxes with ``category_name`` remapped.

        Boxes that map to ``None`` (i.e. not covered by the current schema)
        are excluded from the output.  A ``"raw_category_name"`` key is added
        to every remapped box so the original label is preserved for debugging.
        """
        if self._mode == "raw":
            return boxes  # nothing changes; skip copy

        remapped: List[Dict[str, Any]] = []
        for box in boxes:
            raw_name: str = box.get("category_name", "")
            target = self.remap_category(raw_name)
            if target is None:
                continue  # drop box not covered by this schema
            new_box = dict(box)
            new_box["category_name"] = target
            new_box["raw_category_name"] = raw_name
            remapped.append(new_box)
        return remapped

    def apply_to_frames(
        self,
        frame_records: Iterable[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        """Yield frame records with ``gt_boxes`` category-remapped.

        The source records are **not** mutated; a shallow copy is yielded.
        """
        for frame in frame_records:
            new_frame = dict(frame)
            new_frame["gt_boxes"] = self.remap_boxes(frame.get("gt_boxes", []))
            yield new_frame

    def apply_to_predictions(
        self,
        prediction_records: Iterable[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        """Yield prediction records with ``pred_boxes`` category-remapped."""
        for pred in prediction_records:
            new_pred = dict(pred)
            new_pred["pred_boxes"] = self.remap_boxes(pred.get("pred_boxes", []))
            yield new_pred


# ── module-level helpers ──────────────────────────────────────────────────────

def build_remapper_from_eval_config(
    eval_config_path: str | Path,
    strategy: Optional[str] = None,
) -> CategoryRemapper:
    """Convenience function: load ``eval.yaml`` + ``dataset.yaml`` and return a remapper.

    Parameters
    ----------
    eval_config_path : str | Path
        Path to ``eval.yaml``.
    strategy : str, optional
        Strategy name (e.g. ``"l2_planning"``).  When ``None``, the
        ``active_strategy`` field in ``eval.yaml`` is used.
    """
    eval_cfg = _load_yaml(eval_config_path)

    if strategy is None:
        strategy = eval_cfg.get("active_strategy", "raw")

    strategies: Dict[str, Any] = eval_cfg.get("strategies", {})
    if strategy not in strategies:
        available = list(strategies.keys())
        raise ValueError(
            f"Unknown eval strategy {strategy!r}. Available: {available}"
        )

    schema_name: str = strategies[strategy].get("category_schema", "raw")

    # Resolve dataset.yaml relative to eval.yaml's directory
    dataset_config_path = (
        Path(eval_config_path).parent
        / Path(eval_cfg.get("dataset_config", "configs/dataset.yaml")).name
    )
    return CategoryRemapper.from_config(dataset_config_path, schema=schema_name)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    workspace = Path(__file__).resolve().parents[1]
    dataset_cfg = workspace / "configs" / "dataset.yaml"
    eval_cfg    = workspace / "configs" / "eval.yaml"

    sample_boxes = [
        {"category_name": "vehicle.car",                    "translation": [5, 0, 0]},
        {"category_name": "vehicle.bus.rigid",              "translation": [10, 0, 0]},
        {"category_name": "human.pedestrian.adult",         "translation": [3, 1, 0]},
        {"category_name": "vehicle.motorcycle",             "translation": [8, 0, 0]},
        {"category_name": "movable_object.trafficcone",     "translation": [1, 0, 0]},
        {"category_name": "vehicle.emergency.ambulance",    "translation": [15, 0, 0]},
        {"category_name": "animal",                         "translation": [2, 2, 0]},
        {"category_name": "static_object.bicycle_rack",    "translation": [0, 3, 0]},
    ]

    for schema in ("raw", "detection_10cls", "l2_planning"):
        remapper = CategoryRemapper.from_config(dataset_cfg, schema=schema)
        remapped = remapper.remap_boxes(sample_boxes)
        print(f"\n── schema: {schema}  (target classes: {remapper.target_classes}) ──")
        for box in remapped:
            raw = box.get("raw_category_name", box["category_name"])
            print(f"  {raw:<40s} → {box['category_name']}")
        dropped = len(sample_boxes) - len(remapped)
        if dropped:
            print(f"  ({dropped} box(es) dropped by ignore_unlisted)")

    print("\n── build_remapper_from_eval_config (active_strategy) ──")
    r = build_remapper_from_eval_config(eval_cfg)
    print(f"  active strategy target classes: {r.target_classes}")
