# ad-eval-suite

Autonomous driving evaluation toolkit focused on practical iteration speed:

- evaluate perception quality (detection + tracking)
- mine worst scenes and worst frames automatically
- export replay visuals and OpenSCENARIO files for simulation handoff
- validate and smoke-test `.xosc` scenarios

## Feature Showcase

### Failure replay GIF (mined bad case)

![Failure replay GIF](outputs/perception/gif_halfscenes_fullframes/failure_mining/replay_gif/scene-0061.gif)

### Frame-level BEV replay with map overlay

![Frame replay](outputs/perception/slider_scene0061_fullframes/failure_mining/replay/scene-0061/frame_0015.png)

## What This Project Delivers

### 1) Evaluation module (`eval/`)

- detection metrics: Precision / Recall / F1 / AP / mAP
- tracking metrics: MOTA / MOTP / IDF1 / ID switches / MT / ML
- strategy-driven execution using YAML (`raw`, `detection_10cls`, `l2_planning`)

### 2) Mining module (`mining/`)

- ranks worst scenes and frames by weighted error score
- exports snapshot JSON payloads for reproducible debugging
- supports replay + simulation export from mined snapshots

### 3) Simulation module (`simulation/`)

- snapshot to OpenSCENARIO (`.xosc`) export
- XML/XSD validation pipeline
- esmini headless smoke runner for fast scenario sanity checks

### 4) Visualization module (`visualization/`)

- top-down frame rendering
- animated GIF generation
- optional nuScenes map overlay (lane/drivable area)

## Unified CLI (Final)

Main entry: `scripts/adeval.py`

```bash
# See all commands
python scripts/adeval.py --help

# 1) Full perception evaluation
python scripts/adeval.py eval

# 2) Eval with strategy + subset scenes + artifact export
python scripts/adeval.py eval \
    --strategy l2_planning \
    --scenes first \
    --export-gif \
    --map \
    --export-sim \
    --run-name demo_l2

# 3) Re-export artifacts from an existing run
python scripts/adeval.py mine outputs/perception/demo_l2 --export-gif --export-sim

# 4) Validate OpenSCENARIO files
python scripts/adeval.py sim validate outputs/perception/demo_l2/failure_mining/xosc

# 5) Run esmini smoke tests
python scripts/adeval.py sim smoke outputs/perception/demo_l2/failure_mining/xosc --dry-run

# 6) Replay a snapshot directly
python scripts/adeval.py viz replay outputs/perception/demo_l2/failure_mining/snapshots/scene-0061_snapshot.json --save-gif /tmp/scene-0061.gif --map
```

Compatibility note: `scripts/run_perception_eval.py` is kept for direct use and CI compatibility.

## Config Architecture

Three config files are now clearly separated by concern:

- `configs/dataset.yaml`
    - dataset registry
    - active dataset
    - category schema mapping
- `configs/eval.yaml`
    - strategy defaults (matcher, IoU, metrics level)
    - perception/prediction/planning eval settings
    - mining weights and top-k settings
    - synthetic generator settings
- `configs/sim.yaml`
    - simulation export defaults
    - `.xosc` validation defaults
    - esmini smoke defaults

This split keeps dataset, evaluation, and simulation concerns independent.

## How To Run

### 1) Environment

```bash
git clone https://github.com/cola1917/ad-eval-suite.git
cd ad-eval-suite

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Data

Place nuScenes mini under:

```text
data/nuscenes-mini/
    maps/
    samples/
    sweeps/
    v1.0-mini/
```

### 3) First run

```bash
python scripts/adeval.py eval --run-name first_run
```

Outputs will be in:

```text
outputs/perception/first_run/
    report.json
    report.md
    topn/
    failure_mining/
```

## Current Scope Status

- implemented and production-ready: perception eval, mining, visualization, simulation export/validation/smoke
- scaffold-level (placeholder): dedicated prediction/planning/e2e CLI scripts

## Key Folders

```text
configs/         dataset/eval/sim yaml
datasets/        dataset loaders
eval/            metric evaluators
mining/          failure mining and ranking
simulation/      xosc export + validation
visualization/   replay and map overlay
scripts/         adeval CLI and legacy wrappers
tests/           unit + regression tests
```