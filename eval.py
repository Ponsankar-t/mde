# /home/ponsankar/mde/evaluate.py
import os
import torch
from torch.utils.data import DataLoader
from dataset import AutonomousVehicleDataset
from model import DepthModel
from PIL import Image
import numpy as np

# Configuration
task = "depth_prediction"  # Change to "depth_completion" if needed
data_dir = "/home/ponsankar/dataset/depth_selection/val_selection_cropped"
model_path = f"/home/ponsankar/mde/depth_model_{task}.pth"
output_dir = f"/home/ponsankar/mde/val_predictions_{task}"
device = torch.device("cpu")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Validation dataset
val_dataset = AutonomousVehicleDataset(
    data_dir=data_dir,
    mode="val",
    task=task
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load model
model = DepthModel(task=task).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Generate predictions
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        images = batch["image"].to(device)
        sparse_depth = batch.get("sparse_depth", None)
        if sparse_depth is not None:
            sparse_depth = sparse_depth.to(device)
        depth_pred = model(images, sparse_depth)
        depth_pred = depth_pred.squeeze().cpu().numpy()
        
        # Get corresponding ground truth filename
        gt_filename = val_dataset.gt_files[i]  # Matches dataset.py's gt_files
        file_name = gt_filename  # Use ground truth name directly
        
        # Save prediction
        Image.fromarray((depth_pred * 256.0).astype(np.uint16)).save(
            os.path.join(output_dir, file_name)
        )
        print(f"Saved {file_name}")