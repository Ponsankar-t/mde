# /home/ponsankar/mde/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AutonomousVehicleDataset
from model import DepthModel

# Hyperparameters
batch_size = 8
epochs = 20
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = "depth_prediction"  # Change to "depth_completion" as needed
data_dir = "/home/ponsankar/dataset/depth_selection/val_selection_cropped"

# Dataset and DataLoader
dataset = AutonomousVehicleDataset(
    data_dir=data_dir,
    mode="val",
    task=task
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = DepthModel(task=task).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images = batch["image"].to(device)
        depth_gt = batch["depth_gt"].to(device)  # (B, 1, H, W)
        sparse_depth = batch.get("sparse_depth", None)
        if sparse_depth is not None:
            sparse_depth = sparse_depth.to(device)

        # Forward
        optimizer.zero_grad()
        depth_pred = model(images, sparse_depth)
        # Ensure same size
        if depth_pred.size() != depth_gt.size():
            depth_pred = torch.nn.functional.interpolate(
                depth_pred, size=depth_gt.size()[2:], mode="bilinear", align_corners=False
            )
        loss = criterion(depth_pred, depth_gt)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), f"depth_model_{task}.pth")