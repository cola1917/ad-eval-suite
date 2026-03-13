# Occlusion (ODD) Bucketing Documentation

## Overview

Occlusion bucketing (referred to as ODD - Operational Design Domain bucketing) has been added to the evaluation suite. This feature allows you to:

1. **Evaluate detection performance by occlusion level** (visibility)
2. **Combine occlusion and distance bucketing** for fine-grained analysis
3. **Understand how occlusion affects detection accuracy**

## What is ODD Bucketing?

ODD (Operational Design Domain) bucketing groups objects based on their visibility/occlusion level. In the nuScenes dataset, objects have visibility tokens that represent their occlusion state:

| Visibility Level | ODD Bucket | Description |
|------------------|------------|-------------|
| 4 | `fully_visible` | Object is fully visible |
| 3 | `mostly_visible` | Object is mostly visible with slight occlusion |
| 2 | `partially_occluded` | Object is partially occluded |
| 1 | `mostly_occluded` | Object is mostly occluded, small portion visible |

## New Modules and Functions

### 1. `utils/occlusion_bucket.py`

New module for occlusion-related bucketing functionality:

```python
# Assign a single box to an occlusion bucket
from utils.occlusion_bucket import assign_occlusion_bucket
bucket = assign_occlusion_bucket(box)  # Returns: "fully_visible", "mostly_visible", etc.

# Bucketize all boxes by occlusion level
from utils.occlusion_bucket import bucketize_boxes_by_occlusion
occlusion_buckets = bucketize_boxes_by_occlusion(boxes)
# Returns: {
#     "fully_visible": [...],
#     "mostly_visible": [...],
#     "partially_occluded": [...],
#     "mostly_occluded": [...],
#     "unknown": [...]
# }

# Combined bucketing (occlusion x distance)
from utils.occlusion_bucket import bucketize_boxes_by_distance_and_occlusion
combined = bucketize_boxes_by_distance_and_occlusion(boxes)
# Returns: {
#     "fully_visible": {
#         "near": [...],
#         "medium": [...],
#         "far": [...]
#     },
#     ...
# }

# Get human-readable labels
from utils.occlusion_bucket import occlusion_bucket_labels
labels = occlusion_bucket_labels()
```

### 2. `eval/perception/metrics.py`

Two new functions added for computing metrics by occlusion level:

```python
# Compute metrics bucketed by occlusion only
from eval.perception.metrics import compute_occlusion_bucket_metrics
occ_metrics = compute_occlusion_bucket_metrics(
    gt_boxes=gt_boxes,
    pred_boxes=pred_boxes,
    matcher_fn=matcher_fn,
    iou_threshold=0.5,
    class_aware=True,
)
# Returns: {
#     "fully_visible": {...metrics...},
#     "mostly_visible": {...metrics...},
#     "partially_occluded": {...metrics...},
#     "mostly_occluded": {...metrics...},
#     "unknown": {...metrics...}
# }

# Compute metrics bucketed by both occlusion and distance
from eval.perception.metrics import compute_occlusion_distance_bucket_metrics
combined_metrics = compute_occlusion_distance_bucket_metrics(
    gt_boxes=gt_boxes,
    pred_boxes=pred_boxes,
    matcher_fn=matcher_fn,
    iou_threshold=0.5,
    class_aware=True,
    distance_boundaries=(20.0, 40.0),
)
# Returns: {
#     "fully_visible": {
#         "near": {...metrics...},
#         "medium": {...metrics...},
#         "far": {...metrics...}
#     },
#     ...
# }
```

### 3. `eval/perception/detection_eval.py`

Updated `evaluate_detection_frames()` function with new parameters:

```python
result = evaluate_detection_frames(
    frame_records=frame_records,
    prediction_records=prediction_records,
    iou_threshold=0.5,
    
    # New parameters:
    use_occlusion_bucketing=False,      # Enable occlusion-only bucketing
    use_combined_bucketing=False,       # Enable occlusion x distance bucketing
)

# Result includes new keys when enabled:
if use_occlusion_bucketing:
    result["occlusion_buckets"]       # Metrics by occlusion level
    result["occlusion_bucket_labels"] # Labels for each bucket

if use_combined_bucketing:
    result["occlusion_distance_buckets"] # Nested metrics
    result["combined_bucket_labels"]      # Labels for both dimensions
```

## Usage Examples

### Example 1: Distance Bucketing Only (Existing)

```python
from eval.perception.detection_eval import evaluate_detection_frames

result = evaluate_detection_frames(
    frame_records=frame_records,
    prediction_records=prediction_records,
    iou_threshold=0.5,
    distance_boundaries=(20.0, 40.0),
)

# Results include:
# - result["distance_buckets"]: {"near": {...}, "medium": {...}, "far": {...}}
# - result["distance_bucket_labels"]: Human-readable labels
```

### Example 2: Occlusion Bucketing Only (New)

```python
result = evaluate_detection_frames(
    frame_records=frame_records,
    prediction_records=prediction_records,
    iou_threshold=0.5,
    use_occlusion_bucketing=True,
)

# Print occlusion results
for occ_level, metrics in result["occlusion_buckets"].items():
    label = result["occlusion_bucket_labels"][occ_level]
    print(f"{label}:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
```

### Example 3: Combined Occlusion + Distance Bucketing (New)

```python
result = evaluate_detection_frames(
    frame_records=frame_records,
    prediction_records=prediction_records,
    iou_threshold=0.5,
    distance_boundaries=(20.0, 40.0),
    use_combined_bucketing=True,
)

# Print combined results
occ_labels = result["combined_bucket_labels"]["occlusion"]
dist_labels = result["combined_bucket_labels"]["distance"]

for occ_level, dist_buckets in result["occlusion_distance_buckets"].items():
    print(f"\n{occ_labels[occ_level]}:")
    for dist_name, metrics in dist_buckets.items():
        print(f"  {dist_labels[dist_name]}:")
        print(f"    TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
        print(f"    P={metrics['precision']:.4f} R={metrics['recall']:.4f}")
```

### Example 4: Using Both Distance and Occlusion Bucketing

```python
result = evaluate_detection_frames(
    frame_records=frame_records,
    prediction_records=prediction_records,
    iou_threshold=0.5,
    distance_boundaries=(20.0, 40.0),
    use_occlusion_bucketing=True,
    use_combined_bucketing=True,
)

# Now you have:
# - result["distance_buckets"]: Just distance breakdown
# - result["occlusion_buckets"]: Just occlusion breakdown
# - result["occlusion_distance_buckets"]: Combined breakdown
```

## Running the Examples

Execute the example script to see all three approaches:

```bash
cd /workspaces/ad-eval-suite
python ODD_Bucketing_Example.py
```

## Visibility Tokens in nuScenes Data

The visibility information is stored as part of each annotation's metadata:

```python
# When loading data from nuScenes:
for box in gt_boxes:
    visibility_token = box.get("visibility_token")
    # "4" -> fully_visible
    # "3" -> mostly_visible
    # "2" -> partially_occluded
    # "1" -> mostly_occluded
```

## Key Files Modified

1. **`utils/occlusion_bucket.py`** - New file with occlusion bucketing functions
2. **`eval/perception/metrics.py`** - Added `compute_occlusion_bucket_metrics()` and `compute_occlusion_distance_bucket_metrics()`
3. **`eval/perception/detection_eval.py`** - Updated `evaluate_detection_frames()` with new parameters

## Backward Compatibility

All changes are backward compatible. Existing code continues to work without modifications:

- `use_occlusion_bucketing=False` (default) - Occlusion bucketing disabled
- `use_combined_bucketing=False` (default) - Combined bucketing disabled

## Analysis Benefits

With occlusion bucketing, you can:

1. **Identify robustness to occlusion**: How does your detector perform on occluded objects?
2. **Find challenging scenarios**: Which combinations of distance + occlusion are hardest?
3. **Compare model robustness**: Different detectors may handle occlusion differently
4. **Validate ODD specifications**: Ensure your model works within defined operational domains
5. **Debug detection failures**: Understand if failures are due to distance, occlusion, or both

## Example Output

```
EXAMPLE 3: Combined Occlusion + Distance Bucketing

scene: scene-0001
frames: 5

Combined Buckets (Occlusion x Distance):

  Fully Visible (visibility=4):
    near(<20m):
      TP=15 FP=2 FN=1
      P=0.8824 R=0.9375 F1=0.9090
    medium([20,40)m):
      TP=12 FP=1 FN=2
      P=0.9231 R=0.8571 F1=0.8889
    far(>=40m):
      TP=5 FP=0 FN=3
      P=1.0000 R=0.6250 F1=0.7692

  Mostly Visible (visibility=3):
    near(<20m):
      TP=8 FP=1 FN=2
      P=0.8889 R=0.8000 F1=0.8421
    ...
```

## Contact & Support

For questions or issues with the occlusion bucketing feature, please refer to the individual module docstrings or create an issue in the repository.
