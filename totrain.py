# /home/ponsankar/mde/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AutonomousVehicleDataset
from model import DepthModel
import sys
import time

# Hyperparameters
batch_size = 2
epochs = 30  # Increased to ensure convergence
learning_rate = 0.0005  # Higher for faster learning
device = torch.device("cpu")
task = "depth_prediction"
data_dir = "/home/ponsankar/dataset/depth_selection/val_selection_cropped"

# Dataset
print(f"Loading dataset from {data_dir}...", flush=True)
dataset = AutonomousVehicleDataset(
    data_dir=data_dir,
    mode="val",
    task=task
)
print(f"Dataset size: {len(dataset)} samples", flush=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Model
print("Initializing model...", flush=True)
model = DepthModel(task=task).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print(f"Starting training on {device}...", flush=True)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        sparse_depth = batch.get("sparse_depth", None)
        if sparse_depth is not None:
            sparse_depth = sparse_depth.to(device)

        optimizer.zero_grad()
        depth_pred = model(images, sparse_depth)
        if depth_pred.size() != depth_gt.size():
            depth_pred = torch.nn.functional.interpolate(
                depth_pred, size=depth_gt.size()[2:], mode="bilinear", align_corners=False
            )
        
        # Mask valid depths
        mask = depth_gt > 0
        if mask.sum() > 0:
            loss = criterion(depth_pred[mask], depth_gt[mask])
        else:
            loss = torch.tensor(0.0, device=device)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}", flush=True)

    avg_loss = running_loss / len(dataloader)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s", flush=True)

    # Save checkpoint
    torch.save(model.state_dict(), f"depth_model_{task}_epoch{epoch+1}.pth")

# Save final model
print("Saving final model...", flush=True)
torch.save(model.state_dict(), f"depth_model_{task}.pth")
print("Training complete!", flush=True)