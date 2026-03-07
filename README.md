# Autonomous Driving Evaluation Suite (ad-eval-suite)

A modular, 4-stage evaluation framework for autonomous driving algorithms, covering the full pipeline from raw sensor perception to end-to-end system performance.

---

## Architecture Overview

The suite is organized into four sequential evaluation stages, each represented by its own directory:

| Stage | Directory | Focus |
|-------|-----------|-------|
| 1 | `1_perception_eval/` | Sensor processing, object detection, 2D/3D bounding box metrics (e.g., IoU) |
| 2 | `2_prediction_eval/` | Trajectory and motion prediction accuracy (e.g., ADE, FDE) |
| 3 | `3_planning_control_eval/` | Path planning quality and control command evaluation |
| 4 | `4_end_to_end_eval/` | Full-stack scenario testing and composite system metrics |

---

## Getting Started

### Using GitHub Codespaces (Recommended)

1. Click the green **Code** button at the top of this repository.
2. Select the **Codespaces** tab.
3. Click **Create codespace on main**.

The Codespace will automatically:
- Spin up a Python 3 development container.
- Install all required dependencies via `pip install -r requirements.txt`.
- Enable the **Python** and **Jupyter** VS Code extensions.

### Local Setup

```bash
git clone https://github.com/cola1917/ad-eval-suite.git
cd ad-eval-suite
pip install -r requirements.txt
jupyter notebook
```

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

- `numpy` — numerical computing
- `pandas` — data manipulation
- `matplotlib` — visualization
- `jupyter` — interactive notebooks
- `opencv-python-headless` — image processing

---

## Stage 1 — Perception Evaluation

Start with [`1_perception_eval/01_iou_matching.ipynb`](1_perception_eval/01_iou_matching.ipynb) to explore 2D bounding box IoU (Intersection over Union) calculation, a fundamental metric for object detection evaluation.
