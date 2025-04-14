# /home/ponsankar/mde/predict_distance.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
predictions_dir = "/home/ponsankar/mde/predictions_depth_prediction"  # Update if using depth_completion
output_dir = "/home/ponsankar/mde/distance_results"
os.makedirs(output_dir, exist_ok=True)

# Function to load and convert depth map to meters
def load_depth_map(file_path):
    depth_map = np.array(Image.open(file_path)).astype(np.float32) / 256.0  # Undo scaling
    return depth_map  # In meters

# Function to predict distance at a specific pixel
def get_distance_at_pixel(depth_map, x, y):
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        return depth_map[y, x]
    return None

# Function to predict average distance in a region
def get_average_distance_in_region(depth_map, x_min, y_min, x_max, y_max):
    # Ensure bounds are valid
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(depth_map.shape[1], x_max)
    y_max = min(depth_map.shape[0], y_max)
    if x_max <= x_min or y_max <= y_min:
        return None
    region = depth_map[y_min:y_max, x_min:x_max]
    valid_depths = region[region > 0]  # Ignore zero/invalid depths
    if valid_depths.size > 0:
        return np.mean(valid_depths)
    return None

# Process depth maps
for file_name in sorted(os.listdir(predictions_dir)):
    if file_name.endswith(".png"):
        file_path = os.path.join(predictions_dir, file_name)
        depth_map = load_depth_map(file_path)
        
        # Debug: Check depth map stats
        print(f"{file_name}: Shape={depth_map.shape}, Min={depth_map.min():.2f}, Max={depth_map.max():.2f}, Mean={depth_map.mean():.2f}")
        
        # Distance at center pixel
        center_x, center_y = depth_map.shape[1] // 2, depth_map.shape[0] // 2
        center_distance = get_distance_at_pixel(depth_map, center_x, center_y)
        if center_distance is not None:
            print(f"{file_name}: Distance at center ({center_x}, {center_y}) = {center_distance:.2f} meters")
        else:
            print(f"{file_name}: Center pixel out of bounds or invalid")
        
        # Average distance in region
        region_size = 100
        x_min = center_x - region_size // 2
        x_max = center_x + region_size // 2
        y_min = center_y - region_size // 2
        y_max = center_y + region_size // 2
        avg_distance = get_average_distance_in_region(depth_map, x_min, y_min, x_max, y_max)
        if avg_distance is not None:
            print(f"{file_name}: Average distance in region = {avg_distance:.2f} meters")
        else:
            print(f"{file_name}: No valid depths in region ({x_min}:{x_max}, {y_min}:{y_max})")
        
        # Visualize depth map
        plt.imshow(depth_map, cmap="viridis")
        plt.title(f"{file_name}: Depth Map")
        plt.colorbar(label="Depth (meters)")
        plt.savefig(os.path.join(output_dir, f"{file_name}_visual.png"))
        plt.close()