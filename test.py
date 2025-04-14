# /home/ponsankar/mde/test.py
import os
import torch
from torch.utils.data import DataLoader
from dataset import AutonomousVehicleDataset
from model import DepthModel
from PIL import Image
import numpy as np

# Configuration
task = "depth_prediction"  # Change to "depth_completion" as needed
data_dir = f"/home/ponsankar/dataset/depth_selection/test_{task}_anonymous"
model_path = f"depth_model_{task}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test dataset
test_dataset = AutonomousVehicleDataset(
    data_dir=data_dir,
    mode="test",
    task=task
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = DepthModel(task=task).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Inference
os.makedirs(f"predictions_{task}", exist_ok=True)
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images = batch["image"].to(device)
        sparse_depth = batch.get("sparse_depth", None)
        if sparse_depth is not None:
            sparse_depth = sparse_depth.to(device)
        depth_pred = model(images, sparse_depth)
        depth_pred = depth_pred.squeeze().cpu().numpy()
        # Save prediction as PNG
        Image.fromarray((depth_pred * 256.0).astype(np.uint16)).save(
            f"predictions_{task}/pred_{i:06d}.png"
        )