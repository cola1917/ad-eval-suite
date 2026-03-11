# ad-eval-suite

An **Autonomous Driving Evaluation Suite** for benchmarking and analyzing AD system components across a 4-stage evaluation pipeline: **Perception → Prediction → Planning → End-to-End**.

The project includes a comprehensive educational notebook (`Metrics_Introduction.ipynb`) covering all key AD evaluation metrics, along with a modular Python codebase for running evaluations on the [nuScenes](https://www.nuscenes.org/) dataset.

---

## 4-Stage Evaluation Pipeline

| Stage | Module | Key Metrics |
|-------|--------|-------------|
| 1. Perception | `eval/perception/` | mAP, Precision, Recall, MOTA, MOTP, ID Switch |
| 2. Prediction | `eval/prediction/` | ADE, FDE, minADE, minFDE, Miss Rate |
| 3. Planning | `eval/planning/` | Collision Rate, Jerk, Lane Violation, Route Completion |
| 4. End-to-End | `eval/end_to_end/` | Open/closed-loop driving behavior metrics |

---

## Getting Started

### GitHub Codespaces (zero-setup)

1. Click the green **Code** button on the repository homepage.
2. Select the **Codespaces** tab, then click **Create codespace on main**.
3. Wait for the environment to build — dependencies install automatically via `pip install -r requirements.txt`.
4. Open `Metrics_Introduction.ipynb` to explore all metrics interactively.

### Local Setup

```bash
# Clone the repository
git clone https://github.com/cola1917/ad-eval-suite.git
cd ad-eval-suite

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### nuScenes Data

Place the nuScenes mini split under `data/nuscenes-mini/`. Configure the data root and evaluation parameters in `configs/dataset.yaml` and `configs/eval.yaml`.

---

## Metrics Overview

### Perception — Detection

| Metric | Description |
|--------|-------------|
| Precision / Recall / F1 | TP/FP/FN-based detection quality |
| AP | Area under the interpolated PR curve |
| mAP | Mean AP across all object classes |
| Distance Bucketing | Near / Medium / Far performance breakdown |

### Perception — Tracking (MOT)

$$\text{MOTA} = 1 - \frac{\sum_t(\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}$$

$$\text{MOTP} = \frac{\sum_{i,t} d_{i,t}}{\sum_t c_t}$$

Additional: ID Switch (IDSW), Mostly Tracked (MT ≥ 80%), Mostly Lost (ML ≤ 20%).

### Prediction / End-to-End — Trajectory

$$\text{ADE} = \frac{1}{T}\sum_{t=1}^{T}\sqrt{(\hat{x}_t - x_t)^2 + (\hat{y}_t - y_t)^2}$$

$$\text{FDE} = \sqrt{(\hat{x}_T - x_T)^2 + (\hat{y}_T - y_T)^2}$$

Also supported: minADE, minFDE (best-of-K multimodal), Miss Rate.

### Planning / End-to-End — Driving Behavior

| Metric | Description |
|--------|-------------|
| Collision Rate | Fraction of timesteps with obstacle proximity < threshold |
| Lane Violation | Fraction of timesteps with lateral offset > lane half-width |
| Jerk | Peak $\|\Delta a / \Delta t\|$, comfort threshold ~3 m/s³ |
| Route Completion | Whether ego reaches goal within distance tolerance |

---

## Repository Structure

```
ad-eval-suite/
├── Metrics_Introduction.ipynb  # Comprehensive bilingual metrics reference notebook
├── requirements.txt
├── configs/
│   ├── dataset.yaml            # nuScenes data root, split, sensor modalities
│   └── eval.yaml               # IoU thresholds, distance buckets, class names
├── data/
│   └── nuscenes-mini/          # nuScenes mini split data root
├── datasets/
│   ├── base_dataset.py         # Abstract dataset interface
│   └── nuscenes_loader.py      # nuScenes-devkit-based loader
├── eval/
│   ├── perception/
│   │   ├── detection_eval.py   # Frame-wise detection evaluation (mAP, PR)
│   │   ├── tracking_eval.py    # Sequence-wise MOT evaluation (MOTA, MOTP)
│   │   └── metrics.py          # Shared perception metric helpers
│   ├── prediction/
│   │   ├── prediction_eval.py  # Trajectory prediction evaluation
│   │   └── trajectory_metrics.py
│   ├── planning/
│   │   ├── planning_eval.py    # Planning evaluation runner
│   │   └── planning_metrics.py # Jerk, TTC, route completion helpers
│   └── end_to_end/
│       └── e2e_eval.py         # Open/closed-loop E2E evaluation
├── generators/
│   ├── detection_generator.py  # Synthetic detection data for testing
│   ├── tracking_generator.py   # Synthetic tracklet data
│   ├── trajectory_generator.py # Synthetic predicted trajectories
│   └── e2e_generator.py        # Synthetic E2E planning outputs
├── matching/
│   ├── iou_matching.py         # IoU computation utilities
│   ├── greedy_match.py         # Greedy IoU-based GT↔pred assignment
│   └── hungarian.py            # Optimal assignment (Hungarian algorithm)
├── metrics/
│   ├── precision_recall.py     # Precision, Recall, F1, TP/FP/FN
│   ├── ap_map.py               # AP (PR-curve AUC), mAP across classes
│   ├── tracking_metrics.py     # MOTA, MOTP, IDSW, MT, ML (via motmetrics)
│   └── trajectory_metrics.py  # ADE, FDE, minADE, minFDE, Miss Rate
├── notebooks/
│   ├── Metrics_Introduction.ipynb
│   ├── Perception_Metrics.ipynb
│   ├── Prediction_Metrics.ipynb
│   └── Tracking_Metrics.ipynb
├── scripts/
│   ├── run_perception_eval.py  # CLI: detection + tracking evaluation
│   ├── run_tracking_eval.py    # CLI: MOT evaluation
│   ├── run_prediction_eval.py  # CLI: ADE/FDE evaluation
│   └── run_e2e_eval.py         # CLI: end-to-end evaluation
└── utils/
    ├── iou.py                  # 2D axis-aligned and 3D BEV IoU
    ├── geometry.py             # Coordinate transforms, BEV projection
    ├── distance_bucket.py      # Near / Medium / Far object bucketing
    └── visualization.py        # BEV plots, PR curves, trajectory overlays
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas`, `scipy` | Numerical computation |
| `matplotlib`, `seaborn` | Visualization |
| `opencv-python-headless`, `shapely` | Geometry and image processing |
| `nuscenes-devkit` | nuScenes dataset API |
| `motmetrics` | MOT metrics (MOTA / MOTP) |
| `PyYAML`, `tqdm`, `jupyter` | Configuration, progress, notebooks |

Install all dependencies:

```bash
pip install -r requirements.txt
```