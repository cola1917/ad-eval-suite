# ad-eval-suite

An **Autonomous Driving Evaluation Suite** for benchmarking and analyzing AD system components across a 4-stage evaluation pipeline.

## 4-Stage Evaluation Architecture

| Stage | Directory | Description |
|-------|-----------|-------------|
| 1 | `1_perception_eval/` | Perception metrics (e.g., 2D/3D bounding box IoU, detection accuracy) |
| 2 | `2_prediction_eval/` | Prediction metrics (e.g., trajectory forecasting error, ADE/FDE) |
| 3 | `3_planning_control_eval/` | Planning & control metrics (e.g., collision rate, comfort indices) |
| 4 | `4_end_to_end_eval/` | End-to-end system metrics (e.g., intervention rate, miles per disengagement) |

## Getting Started with GitHub Codespaces

This repository is pre-configured with a Dev Container for a zero-setup experience.

1. Click the green **Code** button on the repository homepage.
2. Select the **Codespaces** tab.
3. Click **Create codespace on main**.
4. Wait for the environment to build — dependencies will be installed automatically via `pip install -r requirements.txt`.
5. Open any notebook in the evaluation directories and select the Python 3 kernel to begin.

## Local Setup

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

## Repository Structure

```
ad-eval-suite/
├── .devcontainer/
│   └── devcontainer.json       # Codespace / Dev Container configuration
├── 1_perception_eval/
│   └── 01_iou_matching.ipynb   # 2D Bounding Box IoU calculation notebook
├── 2_prediction_eval/          # Trajectory prediction evaluation (coming soon)
├── 3_planning_control_eval/    # Planning & control evaluation (coming soon)
├── 4_end_to_end_eval/          # End-to-end system evaluation (coming soon)
├── requirements.txt            # Python dependencies
└── README.md
```