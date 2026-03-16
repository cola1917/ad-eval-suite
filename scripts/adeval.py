"""adeval – unified CLI for the AD Evaluation Suite.

Subcommands
-----------
  eval          Run the full perception evaluation pipeline.
  mine          Re-export artifacts from existing failure-mining snapshots.
  sim validate  Validate OpenSCENARIO (.xosc) files.
  sim smoke     Run esmini headless smoke tests.
  viz replay    Replay a scenario snapshot JSON in a 2-D top-down view.

Quick start
-----------
  python scripts/adeval.py eval
  python scripts/adeval.py eval --strategy l2_planning --scenes first
  python scripts/adeval.py mine outputs/perception/my_run --export-gif --map
  python scripts/adeval.py sim validate outputs/sim/            --report report.json
  python scripts/adeval.py viz replay snapshot.json             --save-gif scene.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── workspace root on path ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# eval  ─ full perception evaluation pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _add_eval_args(p: argparse.ArgumentParser) -> None:
    # ── config files ──────────────────────────────────────────────────────────
    p.add_argument("--eval-config",    default="configs/eval.yaml",
                   help="Evaluation strategy config (default: configs/eval.yaml)")
    p.add_argument("--dataset-config", default="configs/dataset.yaml",
                   help="Dataset registry config (default: configs/dataset.yaml)")
    p.add_argument("--sim-config",     default="configs/sim.yaml",
                   help="Simulation export config (default: configs/sim.yaml)")

    # ── data / strategy selection ─────────────────────────────────────────────
    p.add_argument("--strategy", default=None,
                   help="Eval strategy (raw | detection_10cls | l2_planning); "
                        "default: active_strategy in eval.yaml")
    p.add_argument("--dataset", default=None, dest="dataset_name",
                   help="Dataset key from dataset.yaml; default: active_dataset")
    p.add_argument("--scenes", default=None,
                   help="Scene selection: first | half | full | comma-separated names")
    p.add_argument("--max-frames", default=None,
                   help="Max frames per scene (integer or 'full')")

    # ── execution ─────────────────────────────────────────────────────────────
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker count (default: 1)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed override (default: from eval.yaml defaults.seed)")

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs/perception",
                   help="Root output directory (default: outputs/perception)")
    p.add_argument("--run-name", default=None,
                   help="Run sub-folder name; auto-generated timestamp if omitted")

    # ── export flags ──────────────────────────────────────────────────────────
    p.add_argument("--export-replay", action="store_true",
                   help="Export per-frame PNG images for failure-mined scenes")
    p.add_argument("--export-gif", action="store_true",
                   help="Export animated GIF for each failure-mined scene")
    p.add_argument("--map", action="store_true", dest="overlay_map",
                   help="Overlay lane lines / drivable area on replay outputs")
    p.add_argument("--export-sim", action="store_true",
                   help="Export OpenSCENARIO (.xosc) for failure-mined scenes")

    # ── skip flags ────────────────────────────────────────────────────────────
    p.add_argument("--no-mine", action="store_true",
                   help="Skip failure mining (eval metrics only)")

    # ── power-user algorithm overrides ────────────────────────────────────────
    p.add_argument("--matcher", choices=["greedy", "hungarian"], default=None,
                   help="Matcher override (default: from strategy config)")
    p.add_argument("--metrics", choices=["basic", "standard", "full"], default=None,
                   help="Metrics level override")
    p.add_argument("--bev-iou-mode", choices=["aabb", "polygon"], default=None,
                   help="BEV IoU mode override")


def _cmd_eval(args: argparse.Namespace) -> int:
    """Delegate to run_perception_eval with config-driven defaults."""
    import scripts.run_perception_eval as _eval
    import yaml

    # Load config files to fill defaults.
    def _yaml(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text()) or {}

    eval_cfg = _yaml(args.eval_config)
    sim_cfg  = _yaml(args.sim_config)

    defaults = eval_cfg.get("defaults", {})
    mining_cfg = eval_cfg.get("mining", {})
    gen_cfg = eval_cfg.get("generator", {})
    sim_export_cfg = sim_cfg.get("export", {})

    # Build a synthetic args namespace that run_perception_eval.main() understands.
    ns = argparse.Namespace(
        # config files
        dataset_config=args.dataset_config,
        eval_config=args.eval_config,
        # dataset
        dataset_name=args.dataset_name,
        dataset_path=None,
        data_root=None,
        version=None,
        # scene selection
        scene_id=None,
        scenes=args.scenes,
        max_frames=args.max_frames,
        # strategy / algorithm
        strategy=args.strategy,
        matcher=args.matcher,
        det_iou_threshold=None,
        trk_iou_threshold=None,
        bev_iou_mode=args.bev_iou_mode,
        center_distance_threshold=None,
        metrics=args.metrics,
        # generator – read from config
        seed=args.seed if args.seed is not None else int(defaults.get("seed", 42)),
        drop_rate=float(gen_cfg.get("drop_rate", 0.10)),
        fp_rate=float(gen_cfg.get("fp_rate", 0.15)),
        translation_noise_std=float(gen_cfg.get("translation_noise_std", 0.35)),
        size_noise_std=float(gen_cfg.get("size_noise_std", 0.08)),
        yaw_noise_std=float(gen_cfg.get("yaw_noise_std", 0.05)),
        # output
        output_dir=args.output_dir,
        run_name=args.run_name,
        topn=int(eval_cfg.get("perception", {}).get("topn_visualizations", 5)),
        topn_per_scenario=int(eval_cfg.get("perception", {}).get("topn_per_scenario", 3)),
        # mining – read from config
        failure_topk_scenes=int(mining_cfg.get("top_k_scenes", 3)),
        failure_topk_frames=int(mining_cfg.get("top_k_frames", 10)),
        failure_w_fn=float(mining_cfg.get("weights", {}).get("fn", 1.0)),
        failure_w_fp=float(mining_cfg.get("weights", {}).get("fp", 0.5)),
        failure_w_map=float(mining_cfg.get("weights", {}).get("map", 2.0)),
        failure_w_idsw=float(mining_cfg.get("weights", {}).get("idsw", 1.0)),
        skip_failure_mining=args.no_mine,
        # export flags
        export_replay=args.export_replay,
        export_replay_gif=args.export_gif,
        replay_gif_fps=3,
        replay_show_trajectories=False,
        replay_view_mode="auto",
        replay_view_half_extent=60.0,
        replay_dpi=150,
        overlay_map=args.overlay_map,
        map_data_root=str(Path(args.dataset_config).parent.parent / "data/nuscenes-mini"),
        export_xosc=args.export_sim,
        xosc_map_file=sim_export_cfg.get("xosc_map_file", ""),
        xosc_scene_graph_file=sim_export_cfg.get("xosc_scene_graph_file", ""),
        # unused flags (compatibility)
        workers=args.workers,
        keep_viz_data=False,
    )
    return _eval.main_with_args(ns)


# ══════════════════════════════════════════════════════════════════════════════
# mine  ─ re-export artifacts from existing failure-mining snapshots
# ══════════════════════════════════════════════════════════════════════════════

def _add_mine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("eval_dir",
                   help="Path to a previous eval run directory (contains failure_mining/)")
    p.add_argument("--export-gif", action="store_true",
                   help="Export animated GIF for each snapshot")
    p.add_argument("--map", action="store_true", dest="overlay_map",
                   help="Overlay lane lines / drivable area")
    p.add_argument("--map-data-root", default="data/nuscenes-mini",
                   help="nuScenes data root for map overlay")
    p.add_argument("--view-mode", choices=["auto", "ego_fixed"], default="auto",
                   help="Viewport mode")
    p.add_argument("--view-half-extent", type=float, default=60.0,
                   help="Half extent (m) for ego_fixed view mode")
    p.add_argument("--dpi", type=int, default=150, help="Output DPI")
    p.add_argument("--fps", type=int, default=3,  help="GIF frame rate")
    p.add_argument("--export-sim", action="store_true",
                   help="Export OpenSCENARIO (.xosc) for each snapshot")


def _cmd_mine(args: argparse.Namespace) -> int:
    """Re-export replay / xosc artifacts from existing failure-mining snapshots."""
    from visualization.replay_scene import export_scene_frames, export_scene_gif
    from simulation.export_openscenario import export_snapshot_to_xosc
    import json

    eval_dir = Path(args.eval_dir)
    snapshots = sorted(eval_dir.rglob("snapshot*.json"))
    if not snapshots:
        print(f"[mine] No snapshot*.json files found under {eval_dir}", file=sys.stderr)
        return 1

    map_root = args.map_data_root if args.overlay_map else None
    print(f"[mine] Found {len(snapshots)} snapshot(s) in {eval_dir}")

    for snap_path in snapshots:
        payload = json.loads(snap_path.read_text())
        scene_id = str(payload.get("scene_id", snap_path.stem))

        if args.export_gif:
            gif_path = snap_path.parent / f"{scene_id}.gif"
            export_scene_gif(
                snapshot_payload=payload,
                output_file=str(gif_path),
                fps=args.fps,
                map_data_root=map_root,
                view_mode=args.view_mode,
                view_half_extent=args.view_half_extent,
                dpi=args.dpi,
                show_progress=True,
            )
            print(f"[mine] GIF → {gif_path}")

        if args.export_sim:
            xosc_path = snap_path.parent / f"{scene_id}.xosc"
            export_snapshot_to_xosc(
                snapshot_payload=payload,
                output_path=str(xosc_path),
            )
            print(f"[mine] XOSC → {xosc_path}")

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# sim validate  ─ OpenSCENARIO file validation
# ══════════════════════════════════════════════════════════════════════════════

def _add_sim_validate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="Path to a .xosc file or directory")
    p.add_argument("--glob", default="*.xosc",
                   help="Glob pattern when path is a directory (default: *.xosc)")
    p.add_argument("--xsd", default=None,
                   help="Path to OpenSCENARIO XSD schema for strict validation")
    p.add_argument("--report", default=None,
                   help="Write JSON validation report to this path")
    p.add_argument("--fail-on-warn", action="store_true",
                   help="Treat warnings as errors (non-zero exit)")
    p.add_argument("--sim-config", default="configs/sim.yaml",
                   help="Simulation config (default: configs/sim.yaml)")


def _cmd_sim_validate(args: argparse.Namespace) -> int:
    from simulation.validation import discover_xosc_files, validate_xosc_paths, dump_validation_report
    import yaml

    # Config-file defaults
    sim_cfg: dict = {}
    cfg_path = Path(args.sim_config)
    if cfg_path.exists():
        sim_cfg = yaml.safe_load(cfg_path.read_text()) or {}
    val_cfg = sim_cfg.get("validation", {})

    xsd_path = args.xsd or val_cfg.get("xsd_path")
    fail_on_warn = args.fail_on_warn or bool(val_cfg.get("fail_on_warn", False))
    report_path = args.report or val_cfg.get("report_path")

    files = discover_xosc_files(args.path, pattern=args.glob)
    if not files:
        print(f"[sim validate] No .xosc files found at {args.path}", file=sys.stderr)
        return 1

    results = validate_xosc_paths(files, xsd_path=xsd_path)
    passed = sum(1 for r in results if r.well_formed and r.schema_valid)
    failed = len(results) - passed

    for r in results:
        status = "PASS" if (r.well_formed and r.schema_valid) else "FAIL"
        warn_str = f"  {len(r.warnings)} warning(s)" if r.warnings else ""
        print(f"  [{status}] {r.path}{warn_str}")
        for e in r.errors:
            print(f"           ERROR: {e}")

    print(f"\n[sim validate] {passed}/{len(results)} passed, {failed} failed")

    if report_path:
        dump_validation_report(results, report_path)
        print(f"[sim validate] Report → {report_path}")

    if failed > 0:
        return 1
    if fail_on_warn and any(r.warnings for r in results):
        return 2
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# sim smoke  ─ esmini headless smoke tests
# ══════════════════════════════════════════════════════════════════════════════

def _add_sim_smoke_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="Path to a .xosc file or directory")
    p.add_argument("--glob", default="*.xosc",
                   help="Glob pattern when path is a directory")
    p.add_argument("--esmini-bin", default=None,
                   help="esmini executable path (overrides sim config)")
    p.add_argument("--timeout-sec", type=float, default=None,
                   help="Per-scenario timeout seconds (overrides sim config)")
    p.add_argument("--treat-timeout-as-pass", action="store_true", default=None,
                   help="Count timeout as pass, not failure (overrides sim config)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate files and print commands without running esmini")
    p.add_argument("--sim-config", default="configs/sim.yaml",
                   help="Simulation config (default: configs/sim.yaml)")


def _cmd_sim_smoke(args: argparse.Namespace) -> int:
    """Delegate to run_esmini_smoke script logic."""
    import yaml
    import subprocess

    sim_cfg: dict = {}
    cfg_path = Path(args.sim_config)
    if cfg_path.exists():
        sim_cfg = yaml.safe_load(cfg_path.read_text()) or {}
    esmini_cfg = sim_cfg.get("esmini", {})

    esmini_bin     = args.esmini_bin or esmini_cfg.get("bin", "esmini")
    timeout_sec    = args.timeout_sec if args.timeout_sec is not None else float(esmini_cfg.get("timeout_sec", 60))
    treat_timeout  = (args.treat_timeout_as_pass
                      if args.treat_timeout_as_pass
                      else bool(esmini_cfg.get("treat_timeout_as_pass", True)))

    cmd = [
        sys.executable, str(_ROOT / "scripts" / "run_esmini_smoke.py"),
        args.path,
        "--glob", args.glob,
        "--esmini-bin", esmini_bin,
        "--timeout-sec", str(timeout_sec),
    ]
    if treat_timeout:
        cmd.append("--treat-timeout-as-pass")
    if args.dry_run:
        cmd.append("--dry-run")

    return subprocess.call(cmd)


# ══════════════════════════════════════════════════════════════════════════════
# viz replay  ─ scenario snapshot replay
# ══════════════════════════════════════════════════════════════════════════════

def _add_viz_replay_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("snapshot", help="Path to a scenario snapshot JSON file")
    p.add_argument("--save-dir", default=None,
                   help="Export per-frame PNG images to this directory")
    p.add_argument("--save-gif", default=None,
                   help="Export animated GIF to this path")
    p.add_argument("--fps", type=int, default=3, help="GIF frame rate (default: 3)")
    p.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150)")
    p.add_argument("--map", action="store_true", dest="overlay_map",
                   help="Overlay lane lines / drivable area")
    p.add_argument("--map-data-root", default="data/nuscenes-mini",
                   help="nuScenes data root for map overlay")
    p.add_argument("--view-mode", choices=["auto", "ego_fixed"], default="auto",
                   help="Viewport mode (default: auto)")
    p.add_argument("--view-half-extent", type=float, default=60.0,
                   help="Half extent (m) for ego_fixed view (default: 60.0)")
    p.add_argument("--trajectories", action="store_true",
                   help="Overlay GT and prediction trajectories")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open an interactive window")


def _cmd_viz_replay(args: argparse.Namespace) -> int:
    import json
    from visualization.replay_scene import export_scene_frames, export_scene_gif

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(f"[viz replay] File not found: {snapshot_path}", file=sys.stderr)
        return 1

    payload = json.loads(snapshot_path.read_text())
    map_root = args.map_data_root if args.overlay_map else None

    if args.save_gif:
        gif_out = export_scene_gif(
            snapshot_payload=payload,
            output_file=str(args.save_gif),
            fps=args.fps,
            map_data_root=map_root,
            view_mode=args.view_mode,
            view_half_extent=args.view_half_extent,
            dpi=args.dpi,
            show_trajectories=args.trajectories,
            show_progress=True,
        )
        print(f"[viz replay] GIF → {gif_out}")

    if args.save_dir:
        frames = export_scene_frames(
            snapshot_payload=payload,
            output_dir=str(args.save_dir),
            map_data_root=map_root,
            view_mode=args.view_mode,
            view_half_extent=args.view_half_extent,
            dpi=args.dpi,
            show_trajectories=args.trajectories,
            show_progress=True,
        )
        print(f"[viz replay] {len(frames)} frame(s) → {args.save_dir}")

    if not args.save_gif and not args.save_dir:
        print("[viz replay] No output specified. Use --save-gif or --save-dir.")
        return 1

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# main dispatch
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adeval",
        description="AD Evaluation Suite – unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  adeval eval\n"
            "  adeval eval --strategy l2_planning --scenes first --export-gif --map\n"
            "  adeval mine outputs/perception/my_run --export-gif --map\n"
            "  adeval sim validate outputs/sim/ --report report.json\n"
            "  adeval sim smoke outputs/sim/ --esmini-bin tools/esmini\n"
            "  adeval viz replay snapshot.json --save-gif replay.gif --map\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # eval ─────────────────────────────────────────────────────────────────────
    eval_p = sub.add_parser("eval", help="Run the perception evaluation pipeline")
    _add_eval_args(eval_p)

    # mine ─────────────────────────────────────────────────────────────────────
    mine_p = sub.add_parser("mine", help="Re-export artifacts from existing failure-mining snapshots")
    _add_mine_args(mine_p)

    # sim ──────────────────────────────────────────────────────────────────────
    sim_p = sub.add_parser("sim", help="Simulation tools (validate | smoke)")
    sim_sub = sim_p.add_subparsers(dest="sim_cmd", metavar="action")
    sim_sub.required = True
    _add_sim_validate_args(sim_sub.add_parser("validate", help="Validate .xosc files"))
    _add_sim_smoke_args(sim_sub.add_parser("smoke", help="Run esmini smoke tests"))

    # viz ──────────────────────────────────────────────────────────────────────
    viz_p = sub.add_parser("viz", help="Visualization tools (replay)")
    viz_sub = viz_p.add_subparsers(dest="viz_cmd", metavar="action")
    viz_sub.required = True
    _add_viz_replay_args(viz_sub.add_parser("replay", help="Replay a scenario snapshot"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "mine":
        return _cmd_mine(args)
    if args.command == "sim":
        if args.sim_cmd == "validate":
            return _cmd_sim_validate(args)
        if args.sim_cmd == "smoke":
            return _cmd_sim_smoke(args)
    if args.command == "viz":
        if args.viz_cmd == "replay":
            return _cmd_viz_replay(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
