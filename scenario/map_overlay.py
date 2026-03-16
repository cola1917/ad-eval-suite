"""nuScenes map expansion overlay helper.

Loads the raw expansion JSON directly (bypassing the version-check in the
NuScenesMap class) and provides a simple viewport-based spatial query that
returns drivable area, lane dividers and pedestrian crossings as plain
coordinate lists suitable for matplotlib rendering.

Usage::

    from scenario.map_overlay import load_map_geometry, query_map_patch, draw_map_overlay

    map_data = load_map_geometry("data/nuscenes-mini", "singapore-onenorth")
    patch   = query_map_patch(map_data, cx=620.0, cy=1180.0, half_extent=60.0)
    draw_map_overlay(ax, patch)

``map_data`` is a ``MapGeometry`` named-dict cached object; it is cheap to
pass around and can safely be shared across frames.

All coordinates are in the same global 2-D reference frame (metrics, metres)
as nuScenes ego_pose ``translation[0:2]``.  No further transformation is
required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Public type aliases (just plain dicts / lists for portability)
# ---------------------------------------------------------------------------

# A loaded, indexed map  (result of load_map_geometry)
MapGeometry = Dict[str, Any]

# Result of query_map_patch
MapPatch = Dict[str, Any]


# ---------------------------------------------------------------------------
# Location → filename mapping
# ---------------------------------------------------------------------------

_LOCATION_TO_FILE: Dict[str, str] = {
    "singapore-onenorth":     "singapore-onenorth.json",
    "singapore-hollandvillage": "singapore-hollandvillage.json",
    "singapore-queenstown":   "singapore-queenstown.json",
    "boston-seaport":         "boston-seaport.json",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_map_geometry(data_root: str, location: str) -> Optional[MapGeometry]:
    """Load and index the map expansion JSON for *location*.

    Returns ``None`` when the expansion file is not found (graceful fallback).

    The returned dict contains pre-built indices for fast viewport queries:
    ``_nodes``, ``_polygons``, ``_lines``, plus the raw layer lists.
    """
    filename = _LOCATION_TO_FILE.get(location)
    if filename is None:
        return None

    expansion_path = Path(data_root) / "maps" / "expansion" / filename
    if not expansion_path.exists():
        return None

    with open(expansion_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # Build node token → (x, y) index
    node_index: Dict[str, Tuple[float, float]] = {
        n["token"]: (float(n["x"]), float(n["y"]))
        for n in raw.get("node", [])
    }

    # Build polygon token → exterior [(x,y), …] index
    polygon_index: Dict[str, List[Tuple[float, float]]] = {}
    for poly in raw.get("polygon", []):
        pts = [
            node_index[nt]
            for nt in poly.get("exterior_node_tokens", [])
            if nt in node_index
        ]
        if pts:
            polygon_index[poly["token"]] = pts

    # Build line token → [(x,y), …] index
    line_index: Dict[str, List[Tuple[float, float]]] = {}
    for line in raw.get("line", []):
        pts = [
            node_index[nt]
            for nt in line.get("node_tokens", [])
            if nt in node_index
        ]
        if pts:
            line_index[line["token"]] = pts

    return {
        "location": location,
        "_nodes": node_index,
        "_polygons": polygon_index,
        "_lines": line_index,
        "drivable_area": raw.get("drivable_area", []),
        "lane_divider": raw.get("lane_divider", []),
        "road_divider": raw.get("road_divider", []),
        "ped_crossing": raw.get("ped_crossing", []),
        "lane": raw.get("lane", []),
    }


# ---------------------------------------------------------------------------
# Viewport spatial query
# ---------------------------------------------------------------------------

def query_map_patch(
    map_data: MapGeometry,
    cx: float,
    cy: float,
    half_extent: float = 60.0,
) -> MapPatch:
    """Return all geometry elements that intersect a square viewport.

    Args:
        map_data:    Result of :func:`load_map_geometry`.
        cx, cy:      Centre of the viewport in global nuScenes coordinates.
        half_extent: Half-side-length of the square patch in metres.

    Returns:
        Dict with keys:
        ``drivable_polys``  – list of exterior coordinate rings (Lists of (x,y))
        ``lane_div_lines``  – list of polylines (Lists of (x,y)), colour hint "white"
        ``road_div_lines``  – list of polylines (Lists of (x,y)), colour hint "yellow"
        ``ped_crossing_polys`` – list of exterior coordinate rings
    """
    xmin, xmax = cx - half_extent, cx + half_extent
    ymin, ymax = cy - half_extent, cy + half_extent

    polygon_index = map_data["_polygons"]
    line_index = map_data["_lines"]

    def _poly_in_patch(pts: List[Tuple[float, float]]) -> bool:
        return any(xmin <= x <= xmax and ymin <= y <= ymax for x, y in pts)

    def _line_in_patch(pts: List[Tuple[float, float]]) -> bool:
        return any(xmin <= x <= xmax and ymin <= y <= ymax for x, y in pts)

    # ── drivable area ────────────────────────────────────────────────────────
    drivable_polys: List[List[Tuple[float, float]]] = []
    for da in map_data["drivable_area"]:
        for pt in da.get("polygon_tokens", []):
            if pt in polygon_index and _poly_in_patch(polygon_index[pt]):
                drivable_polys.append(polygon_index[pt])

    # ── lane dividers ────────────────────────────────────────────────────────
    lane_div_lines: List[List[Tuple[float, float]]] = []
    for ld in map_data["lane_divider"]:
        lt = ld.get("line_token", "")
        if lt in line_index and _line_in_patch(line_index[lt]):
            lane_div_lines.append(line_index[lt])

    # ── road dividers (centre-lines / double-yellow) ─────────────────────────
    road_div_lines: List[List[Tuple[float, float]]] = []
    for rd in map_data["road_divider"]:
        lt = rd.get("line_token", "")
        if lt in line_index and _line_in_patch(line_index[lt]):
            road_div_lines.append(line_index[lt])

    # ── pedestrian crossings ─────────────────────────────────────────────────
    ped_crossing_polys: List[List[Tuple[float, float]]] = []
    for pc in map_data["ped_crossing"]:
        pt = pc.get("polygon_token", "")
        if pt in polygon_index and _poly_in_patch(polygon_index[pt]):
            ped_crossing_polys.append(polygon_index[pt])

    return {
        "drivable_polys": drivable_polys,
        "lane_div_lines": lane_div_lines,
        "road_div_lines": road_div_lines,
        "ped_crossing_polys": ped_crossing_polys,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_map_overlay(ax: Any, patch: MapPatch) -> None:
    """Draw map geometry layers onto a matplotlib Axes.

    Draw order (back → front):
    1. Drivable area – light gray fill
    2. Pedestrian crossings – light orange fill
    3. Road dividers – yellow dashed centre lines
    4. Lane dividers – white/light-gray dashed lines
    """
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    # ── drivable area ────────────────────────────────────────────────────────
    if patch["drivable_polys"]:
        patches = [MplPolygon(pts, closed=True) for pts in patch["drivable_polys"]]
        coll = PatchCollection(patches, facecolor="#404040", edgecolor="none", alpha=0.30, zorder=0)
        ax.add_collection(coll)

    # ── pedestrian crossings ─────────────────────────────────────────────────
    if patch["ped_crossing_polys"]:
        patches = [MplPolygon(pts, closed=True) for pts in patch["ped_crossing_polys"]]
        coll = PatchCollection(patches, facecolor="#e8a020", edgecolor="none", alpha=0.35, zorder=1)
        ax.add_collection(coll)

    # ── road dividers (double-yellow centre lines) ───────────────────────────
    for pts in patch["road_div_lines"]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#e8e820", linewidth=1.0, linestyle="--", alpha=0.70, zorder=2)

    # ── lane dividers (dashed white) ─────────────────────────────────────────
    for pts in patch["lane_div_lines"]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#c8c8c8", linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.70, zorder=2)
