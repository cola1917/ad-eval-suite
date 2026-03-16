from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
	from simulator_export.validation import (
		discover_xosc_files,
		dump_validation_report,
		validate_xosc_paths,
	)
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from simulator_export.validation import (
		discover_xosc_files,
		dump_validation_report,
		validate_xosc_paths,
	)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Validate OpenSCENARIO (.xosc) files")
	parser.add_argument("input", help="Input .xosc file or directory")
	parser.add_argument("--glob", default="*.xosc", help="Glob pattern when input is a directory")
	parser.add_argument("--xsd", default=None, help="Optional OpenSCENARIO XSD file path")
	parser.add_argument("--report", default=None, help="Optional JSON report output path")
	parser.add_argument("--fail-on-warn", action="store_true", help="Treat warnings as failures")
	return parser


def main() -> int:
	args = _build_parser().parse_args()
	files = discover_xosc_files(args.input, pattern=args.glob)
	if not files:
		print(f"[xosc-validate] no files found under: {args.input}")
		return 2

	results = validate_xosc_paths([str(path) for path in files], xsd_path=args.xsd)
	valid_count = sum(1 for item in results if item.is_valid)
	warn_count = sum(len(item.warnings) for item in results)
	error_count = sum(len(item.errors) for item in results)

	for item in results:
		status = "PASS" if item.is_valid else "FAIL"
		print(f"[{status}] {item.path}")
		for message in item.errors:
			print(f"  error: {message}")
		for message in item.warnings:
			print(f"  warn: {message}")

	if args.report:
		report_path = dump_validation_report(results, output_path=args.report)
		print(f"[xosc-validate] report: {report_path}")

	print(
		f"[xosc-validate] files={len(results)} pass={valid_count} "
		f"fail={len(results) - valid_count} warnings={warn_count} errors={error_count}"
	)

	if valid_count != len(results):
		return 1
	if args.fail_on_warn and warn_count > 0:
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
