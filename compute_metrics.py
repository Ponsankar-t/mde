# /home/ponsankar/mde/compute_metrics.py
import os
import numpy as np
from PIL import Image

# Configuration
predictions_dir = "/home/ponsankar/mde/val_predictions_depth_prediction"
gt_dir = "/home/ponsankar/dataset/depth_selection/val_selection_cropped/groundtruth_depth"

# Function to load depth map (in meters)
def load_depth_map(file_path):
    return np.array(Image.open(file_path)).astype(np.float32) / 256.0

# Function to compute metrics
def compute_metrics(pred, gt):
    # Mask for valid depths (gt > 0)
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]
    
    if pred.size == 0 or gt.size == 0:
        return None, None, None
    
    # AbsRel
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    
    # Delta_1
    ratio = np.maximum(pred / gt, gt / pred)
    delta_1 = np.mean(ratio < 1.25)
    
    return abs_rel, rmse, delta_1

# Process predictions
results = []
for file_name in sorted(os.listdir(predictions_dir)):
    if file_name.endswith(".png"):
        pred_path = os.path.join(predictions_dir, file_name)
        gt_path = os.path.join(gt_dir, file_name)
        if not os.path.exists(gt_path):
            print(f"Ground truth not found for {file_name}")
            continue
        
        pred_depth = load_depth_map(pred_path)
        gt_depth = load_depth_map(gt_path)
        
        abs_rel, rmse, delta_1 = compute_metrics(pred_depth, gt_depth)
        if abs_rel is not None:
            results.append({"file": file_name, "abs_rel": abs_rel, "rmse": rmse, "delta_1": delta_1})
            print(f"{file_name}: AbsRel={abs_rel:.4f}, RMSE={rmse:.4f}, Delta_1={delta_1:.4f}")
        else:
            print(f"{file_name}: No valid depths")

# Aggregate results
if results:
    abs_rel_avg = np.mean([r["abs_rel"] for r in results])
    rmse_avg = np.mean([r["rmse"] for r in results])
    delta_1_avg = np.mean([r["delta_1"] for r in results])
    print(f"\nAverage Metrics (N={len(results)}):")
    print(f"AbsRel: {abs_rel_avg:.4f}")
    print(f"RMSE: {rmse_avg:.4f}")
    print(f"Delta_1: {delta_1_avg:.4f}")
else:
    print("No valid results computed")