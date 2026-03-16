from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import xml.etree.ElementTree as ET


@dataclass
class XoscValidationResult:
	path: str
	well_formed: bool = False
	schema_valid: bool | None = None
	semantic_valid: bool = False
	errors: List[str] = field(default_factory=list)
	warnings: List[str] = field(default_factory=list)

	@property
	def is_valid(self) -> bool:
		schema_ok = self.schema_valid is not False
		return self.well_formed and self.semantic_valid and schema_ok

	def to_dict(self) -> Dict[str, Any]:
		return {
			"path": self.path,
			"well_formed": self.well_formed,
			"schema_valid": self.schema_valid,
			"semantic_valid": self.semantic_valid,
			"is_valid": self.is_valid,
			"errors": list(self.errors),
			"warnings": list(self.warnings),
		}


def discover_xosc_files(path: str, pattern: str = "*.xosc") -> List[Path]:
	root = Path(path)
	if root.is_file():
		return [root] if root.suffix.lower() == ".xosc" else []
	if not root.exists():
		return []
	return sorted(root.rglob(pattern))


def _validate_semantics(tree: ET.ElementTree, out: XoscValidationResult) -> None:
	root = tree.getroot()
	if root.tag != "OpenSCENARIO":
		out.errors.append(f"Root element must be OpenSCENARIO, got: {root.tag}")
		out.semantic_valid = False
		return

	required_paths = [
		"./FileHeader",
		"./Entities",
		"./Storyboard",
	]
	for req in required_paths:
		if root.find(req) is None:
			out.errors.append(f"Missing required element: {req}")

	scenario_objects = root.findall("./Entities/ScenarioObject")
	if not scenario_objects:
		out.errors.append("Entities must include at least one ScenarioObject")

	polylines = root.findall(".//Polyline")
	if not polylines:
		out.errors.append("At least one Polyline is required for trajectory playback")
	else:
		for idx, polyline in enumerate(polylines):
			vertices = polyline.findall("./Vertex")
			if len(vertices) < 2:
				out.warnings.append(f"Polyline[{idx}] has < 2 vertices")
				continue
			last_t = None
			for vertex in vertices:
				t = float(vertex.attrib.get("time", "0"))
				if last_t is not None and t < last_t:
					out.errors.append(f"Polyline[{idx}] has non-monotonic vertex time: {t} < {last_t}")
					break
				last_t = t

	out.semantic_valid = len(out.errors) == 0


def _validate_schema_with_lxml(xml_path: Path, xsd_path: Path, out: XoscValidationResult) -> None:
	try:
		from lxml import etree as LET  # type: ignore
	except Exception:
		out.schema_valid = None
		out.warnings.append("lxml not installed; skipped XSD validation")
		return

	if not xsd_path.exists():
		out.schema_valid = False
		out.errors.append(f"XSD file not found: {xsd_path}")
		return

	try:
		xml_doc = LET.parse(str(xml_path))
		xsd_doc = LET.parse(str(xsd_path))
		schema = LET.XMLSchema(xsd_doc)
		schema.assertValid(xml_doc)
		out.schema_valid = True
	except Exception as exc:
		out.schema_valid = False
		out.errors.append(f"XSD validation failed: {exc}")


def validate_xosc_file(path: str, xsd_path: str | None = None) -> XoscValidationResult:
	xml_path = Path(path)
	result = XoscValidationResult(path=str(xml_path))

	if not xml_path.exists():
		result.errors.append(f"File not found: {xml_path}")
		return result

	try:
		tree = ET.parse(xml_path)
		result.well_formed = True
	except Exception as exc:
		result.errors.append(f"XML parse failed: {exc}")
		return result

	_validate_semantics(tree, result)
	if xsd_path:
		_validate_schema_with_lxml(xml_path, Path(xsd_path), result)

	return result


def validate_xosc_paths(paths: Iterable[str], xsd_path: str | None = None) -> List[XoscValidationResult]:
	return [validate_xosc_file(path, xsd_path=xsd_path) for path in paths]


def dump_validation_report(results: Iterable[XoscValidationResult], output_path: str) -> str:
	rows = [item.to_dict() for item in results]
	summary = {
		"total": len(rows),
		"valid": sum(1 for r in rows if r["is_valid"]),
		"invalid": sum(1 for r in rows if not r["is_valid"]),
	}
	payload = {
		"summary": summary,
		"results": rows,
	}
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return str(path)
