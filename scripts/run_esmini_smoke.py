from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List
import sys

try:
	from simulator_export.validation import discover_xosc_files
except ImportError:  # pragma: no cover
	workspace_root = Path(__file__).resolve().parents[1]
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from simulator_export.validation import discover_xosc_files


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run esmini smoke tests for .xosc files")
	parser.add_argument("input", help="Input .xosc file or directory")
	parser.add_argument("--glob", default="*.xosc", help="Glob pattern when input is a directory")
	parser.add_argument("--esmini-bin", default="esmini", help="esmini executable path")
	parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per-file timeout seconds")
	parser.add_argument("--extra-arg", action="append", default=[], help="Extra esmini arg (repeatable)")
	parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
	parser.add_argument("--report", default=None, help="Optional JSON report output path")
	return parser


def _run_one(
	*,
	esmini_bin: str,
	xosc_file: Path,
	timeout_sec: float,
	extra_args: List[str],
	dry_run: bool,
) -> Dict[str, Any]:
	cmd = [esmini_bin, "--osc", str(xosc_file), "--headless", *extra_args]
	row: Dict[str, Any] = {
		"file": str(xosc_file),
		"command": cmd,
		"return_code": None,
		"timeout": False,
		"ok": False,
		"stdout": "",
		"stderr": "",
	}
	if dry_run:
		row["ok"] = True
		return row

	try:
		proc = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=max(timeout_sec, 1.0),
			check=False,
		)
		row["return_code"] = int(proc.returncode)
		row["stdout"] = (proc.stdout or "")[-4000:]
		row["stderr"] = (proc.stderr or "")[-4000:]
		row["ok"] = proc.returncode == 0
	except subprocess.TimeoutExpired as exc:
		row["timeout"] = True
		row["stderr"] = f"timeout after {timeout_sec}s: {exc}"

	return row


def main() -> int:
	args = _build_parser().parse_args()
	files = discover_xosc_files(args.input, pattern=args.glob)
	if not files:
		print(f"[esmini-smoke] no files found under: {args.input}")
		return 2

	esmini_path = shutil.which(args.esmini_bin)
	if not args.dry_run and not esmini_path:
		print(f"[esmini-smoke] esmini binary not found: {args.esmini_bin}")
		print("[esmini-smoke] install esmini or provide --esmini-bin <path>")
		return 2
	esmini_bin = esmini_path or args.esmini_bin

	rows: List[Dict[str, Any]] = []
	for path in files:
		row = _run_one(
			esmini_bin=esmini_bin,
			xosc_file=path,
			timeout_sec=float(args.timeout_sec),
			extra_args=list(args.extra_arg),
			dry_run=bool(args.dry_run),
		)
		rows.append(row)
		status = "PASS" if row["ok"] else "FAIL"
		print(f"[{status}] {path}")
		if not row["ok"] and row["stderr"]:
			print(f"  stderr: {row['stderr'].splitlines()[-1]}")

	ok_count = sum(1 for row in rows if row["ok"])
	summary = {
		"total": len(rows),
		"passed": ok_count,
		"failed": len(rows) - ok_count,
	}

	if args.report:
		report = {
			"summary": summary,
			"results": rows,
		}
		out = Path(args.report)
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_text(json.dumps(report, indent=2), encoding="utf-8")
		print(f"[esmini-smoke] report: {out}")

	print(f"[esmini-smoke] total={summary['total']} passed={summary['passed']} failed={summary['failed']}")
	return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
