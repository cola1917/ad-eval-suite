# ad-eval-suite

An **Autonomous Driving Evaluation Suite** for benchmarking AD system components across a 4-stage pipeline: **Perception → Prediction → Planning → End-to-End**.

Built on the [nuScenes](https://www.nuscenes.org/) mini dataset, it provides a modular Python evaluation framework with strategy-based configuration, class-aware matching, and interactive Jupyter notebooks covering every key metric.

---

## 4-Stage Evaluation Pipeline

| Stage | Script | Key Metrics |
|-------|--------|-------------|
| 1. Perception | `scripts/run_perception_eval.py` | Precision, Recall, F1, mAP, MOTA, MOTP, IDF1, IDSW |
| 2. Prediction | `scripts/run_prediction_eval.py` | ADE, FDE, minADE, minFDE, Miss Rate |
| 3. Planning | `scripts/run_planning_eval.py` | Collision Rate, Jerk, Lane Violation, Route Completion |
| 4. End-to-End | `scripts/run_e2e_eval.py` | Open/closed-loop driving behavior metrics |

---

## Getting Started

### GitHub Codespaces (zero-setup)

1. Click the green **Code** button → **Codespaces** tab → **Create codespace on main**.
2. The environment builds automatically and installs all dependencies.
3. Open `Metrics_Introduction.ipynb` to explore all metrics interactively.

### Local Setup

```bash
git clone https://github.com/cola1917/ad-eval-suite.git
cd ad-eval-suite

python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### nuScenes Mini Data

The nuScenes mini split is expected at `data/nuscenes-mini/` with the standard layout:

```
data/nuscenes-mini/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/
```

Datasets and category schemas are configured in `configs/dataset.yaml`; evaluation strategies are configured in `configs/eval.yaml`.

---

## Running Evaluations

### Perception (Detection + Tracking)

```bash
# Default: active dataset + active strategy from YAML
python scripts/run_perception_eval.py

# Choose dataset and strategy from YAML
python scripts/run_perception_eval.py --dataset-name nuscenes_mini --strategy detection_10cls
python scripts/run_perception_eval.py --dataset-name nuscenes_mini --strategy l2_planning

# Temporary CLI overrides (without editing YAML)
python scripts/run_perception_eval.py \
    --dataset-path /mnt/data/custom_nuscenes \
    --version v1.0 \
    --strategy detection_10cls \
    --scenes first \
    --max-frames 50 \
    --metrics full \
    --run-name my_run

# Scene selection supports: first | half | full | explicit list
python scripts/run_perception_eval.py --scenes scene-0061,scene-0103 --max-frames 5
```

Detection and tracking are evaluated together and written to `outputs/perception/<run-name>/`.

`--max-frames` is applied per selected scene (`full` means no frame cap).

### Prediction

```bash
python scripts/run_prediction_eval.py
```

### Planning

```bash
python scripts/run_planning_eval.py
```

### End-to-End

```bash
python scripts/run_e2e_eval.py
```

---

## Eval Strategies

Strategies are defined in `configs/eval.yaml` and select a category schema plus default eval parameters.

| Strategy | Classes | IoU Threshold | Matcher | Description |
|----------|---------|---------------|---------|-------------|
| `raw` | 23 (all nuScenes) | 0.5 | greedy | No remapping; full original taxonomy |
| `detection_10cls` *(default)* | 10 | 0.5 | greedy | Standard nuScenes detection benchmark |
| `l2_planning` | 4 | 0.3 | hungarian | Coarse semantic grouping for planning queries |

The **detection_10cls** schema maps nuScenes sub-categories (e.g. `vehicle.car`, `human.pedestrian.adult`) to 10 canonical classes. The **l2_planning** schema coarsens everything into `vehicle / pedestrian / cyclist / object`.

---

## BEV IoU Modes

The `--bev-iou-mode` flag (or `bev_iou_mode` in `eval.yaml`) selects the Bird's Eye View IoU computation method:

| Mode | Speed | Yaw-aware | Notes |
|------|-------|-----------|-------|
| `aabb` *(default)* | Fast | No | Axis-aligned bounding boxes; NumPy-vectorized |
| `polygon` | Slower | Yes | Rotated polygon IoU via Shapely |

`aabb` is the default and sufficient for most benchmarks. Use `polygon` when yaw accuracy matters.

---

## Metrics Reference

### Perception — Detection

| Metric | Description |
|--------|-------------|
| Precision / Recall / F1 | TP/FP/FN-based per-frame detection quality |
| AP | Area under the 11-point interpolated PR curve |
| mAP | Mean AP across all object classes |
| FP Breakdown | False positives by cause: localization, background, class confusion |
| Distance Buckets | Near / Medium / Far performance split |
| Occlusion Buckets | Performance by annotation visibility level |

### Perception — Tracking (MOT)

$$\text{MOTA} = 1 - \frac{\sum_t(\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}$$

$$\text{MOTP} = \frac{\sum_{i,t} d_{i,t}}{\sum_t c_t}$$

Also reported: IDF1, ID Switch (IDSW), Mostly Tracked (MT ≥ 80%), Mostly Lost (ML ≤ 20%).

### Prediction — Trajectory

$$\text{ADE} = \frac{1}{T}\sum_{t=1}^{T}\sqrt{(\hat{x}_t - x_t)^2 + (\hat{y}_t - y_t)^2}$$

$$\text{FDE} = \sqrt{(\hat{x}_T - x_T)^2 + (\hat{y}_T - y_T)^2}$$

Also supported: minADE, minFDE (best-of-K multimodal prediction), Miss Rate.

### Planning / End-to-End

| Metric | Description |
|--------|-------------|
| Collision Rate | Fraction of timesteps with obstacle proximity below threshold |
| Lane Violation | Fraction of timesteps with lateral offset beyond lane half-width |
| Jerk | Peak $\|\Delta a / \Delta t\|$; comfort threshold ~3 m/s³ |
| Route Completion | Whether ego reaches the goal within distance tolerance |

---

## Notebooks

| Notebook | Location | Content |
|----------|----------|---------|
| Metrics Introduction | `Metrics_Introduction.ipynb` | Full walkthrough of all AD evaluation metrics |
| Perception Metrics | `notebooks/Perception_Metrics.ipynb` | Detection PR curves, mAP, FP breakdown |
| Tracking Metrics | `notebooks/Tracking_Metrics.ipynb` | MOTA, MOTP, IDF1 deep-dive |
| Prediction Metrics | `notebooks/Prediction_Metrics.ipynb` | ADE, FDE, Miss Rate visualizations |

---

## Repository Structure

```
ad-eval-suite/
├── Metrics_Introduction.ipynb      # Top-level metrics reference notebook
├── requirements.txt
├── configs/
│   ├── dataset.yaml                # nuScenes data root, version, category schemas
│   └── eval.yaml                   # Strategies, IoU thresholds, matcher defaults
├── data/
│   └── nuscenes-mini/              # nuScenes mini split (maps, samples, v1.0-mini)
├── datasets/
│   ├── base_dataset.py             # Abstract dataset interface
│   └── nuscenes_loader.py          # nuScenes-devkit loader
├── eval/
│   ├── perception/
│   │   ├── _common.py             # Shared matcher/scenario helper functions
│   │   ├── bucket_metrics.py      # Distance/occlusion bucket and FP breakdown helpers
│   │   ├── detection_eval.py       # Frame-wise detection eval (mAP, PR, FP breakdown)
│   │   ├── tracking_eval.py        # Sequence-wise MOT eval (MOTA, MOTP, IDF1)
│   ├── prediction/
│   │   ├── prediction_eval.py      # Trajectory prediction evaluation
│   │   └── trajectory_metrics.py
│   ├── planning/
│   │   ├── planning_eval.py        # Planning evaluation runner
│   │   └── planning_metrics.py     # Jerk, TTC, route completion
│   └── end_to_end/
│       └── e2e_eval.py             # Open/closed-loop E2E evaluation
├── generators/                     # Synthetic data generators (for testing)
│   ├── detection_generator.py
│   ├── tracking_generator.py
│   ├── trajectory_generator.py
│   └── e2e_generator.py
├── matching/
│   ├── iou_matching.py             # BEV IoU (AABB vectorized + polygon via Shapely)
│   ├── greedy_match.py             # Greedy class-aware GT↔pred assignment
│   └── hungarian.py                # Optimal class-aware assignment (linear_sum_assignment)
├── metrics/
│   ├── precision_recall.py         # Precision, Recall, F1, TP/FP/FN aggregation
│   ├── ap_map.py                   # AP (PR-curve AUC) and mAP
│   ├── tracking_metrics.py         # MOTA, MOTP, IDSW, MT, ML (via motmetrics)
│   └── trajectory_metrics.py       # ADE, FDE, minADE, minFDE, Miss Rate
├── notebooks/
│   ├── Perception_Metrics.ipynb
│   ├── Prediction_Metrics.ipynb
│   └── Tracking_Metrics.ipynb
├── scripts/
│   ├── run_perception_eval.py      # CLI: detection + tracking
│   ├── run_prediction_eval.py      # CLI: ADE/FDE
│   ├── run_planning_eval.py        # CLI: planning metrics
│   └── run_e2e_eval.py             # CLI: end-to-end
└── utils/
    ├── category_remap.py           # Category schema remapping
    ├── distance_bucket.py          # Near / Medium / Far bucketing
    ├── geometry.py                 # BEV projection, coordinate transforms
    ├── iou.py                      # 2D axis-aligned IoU utilities
    └── visualization.py            # BEV plots, PR curves, trajectory overlays
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas`, `scipy` | Numerical computation and matching |
| `matplotlib`, `seaborn` | Visualization |
| `opencv-python-headless`, `shapely` | Image processing and polygon geometry |
| `nuscenes-devkit` | nuScenes dataset API |
| `motmetrics` | MOT metrics (MOTA / MOTP / IDF1) |
| `PyYAML`, `tqdm`, `jupyter`, `gdown` | Configuration, progress, notebooks, data download |

```bash
pip install -r requirements.txt
```