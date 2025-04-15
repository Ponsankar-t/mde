import cv2
import torch
import numpy as np
from PIL import Image
from model import DepthModel
import time

# Configuration
model_path = "/home/ponsankar/mde/depth_model_depth_prediction.pth"
device = torch.device("cpu")
task = "depth_prediction"
input_size = (1216, 352)  # Model input size (width x height)
max_depth = 80.0  # Model depth range (0–80m)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load model
print("Loading model...")
model = DepthModel(task=task).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
model.eval()

# Preprocess frame
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_resized = frame_pil.resize(input_size, Image.BILINEAR)
    frame_np = np.array(frame_resized).astype(np.float32) / 255.0
    return torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(device), frame_rgb

# Get depth map
def get_depth_map(frame_tensor):
    with torch.no_grad():
        depth_pred = model(frame_tensor, sparse_depth=None)
        return depth_pred.squeeze().cpu().numpy()

# Get distance at center
def get_center_distance(depth_map):
    center_x, center_y = input_size[0] // 2, input_size[1] // 2  # 608, 176
    if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
        return depth_map[center_y, center_x]
    return None

# Main loop
print("Starting live object distance prediction (press 'q' to quit)...")
while True:
    start_time = time.time()
    
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Process frame
    frame_tensor, frame_rgb = preprocess_frame(frame)
    
    # Predict depth
    depth_map = get_depth_map(frame_tensor)
    
    # Debug depth map stats
    print(f"Depth map stats: Max={depth_map.max():.2f}m, Mean={depth_map.mean():.2f}m")
    
    # Get object distance (center pixel)
    distance = get_center_distance(depth_map)
    if distance is not None and distance > 0:
        print(f"Object distance: {distance:.2f}m")
    else:
        print("Object distance: N/A (invalid depth)")
    
    # Visualize
    cv2.imshow("RGB Frame", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    depth_display = (depth_map / max_depth * 255.0).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_VIRIDIS)
    cv2.imshow("Depth Map", depth_colored)
    
    # Accurate FPS
    elapsed_time = time.time() - start_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
    print(f"FPS: {fps:.2f}")
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Stopped prediction")